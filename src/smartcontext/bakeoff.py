"""Local-model bake-off -- compares candidate Ollama models on the actual
filtering task, end to end, against a fixed judge.

Unlike ``sweep.py`` (an offline token-count proxy), this hits the live
Claude API three times per (model, capture) pair:

1. **Baseline** -- the original, unpruned payload, once per capture, shared
   across every candidate. This is the reference answer.
2. **Candidate** -- the payload after that candidate model decided what to
   keep, sent to the same target model the capture was originally recorded
   against.
3. **Judge** -- one fixed model (``JUDGE_MODEL``, no extended thinking)
   compares the candidate answer to the baseline and scores whether pruning
   lost anything material.

The judge is deliberately never one of the candidates: grading your own
output (or a sibling's) is a conflict of interest, and a fixed model is
what makes scores comparable *across* candidates -- the one thing that has
to hold for a bake-off to mean anything. (``temperature`` is intentionally
omitted -- ``JUDGE_MODEL`` rejects it outright with a 400.)

This spends real money on every run. Callers should show ``estimate_plan()``
before spending it -- see ``cli.cmd_bakeoff``.
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .local_model import LocalModel
from .pruner import Pruner, _recent_task_text, estimate_payload_tokens
from .store import Store
from .tokens import price_per_token, relative_input_cost

# Fixed and never a candidate -- see module docstring on why consistency
# requires this. No thinking: we want a repeatable score, not the judge's
# most thoughtful take. No `temperature` either -- this model rejects it
# outright ("`temperature` is deprecated for this model", a hard 400), so
# repeatability rests on the fixed model + prompt alone.
JUDGE_MODEL = "claude-sonnet-5"

# Anthropic's output pricing runs close to 5x the input rate across every
# current tier (opus $5/$25, sonnet $3/$15) -- used only to size the
# pre-flight cost estimate, never billed dollars.
_OUTPUT_PRICE_MULTIPLIER = 5
# Rough guess for a typical assistant turn, used only for the cost estimate.
_ASSUMED_OUTPUT_TOKENS = 600
_JUDGE_MAX_TOKENS = 200

JUDGE_PROMPT = """You are grading whether a context-pruning system damaged an answer's quality.

TASK THE ASSISTANT WAS ASKED TO DO:
{task}

REFERENCE ANSWER (produced with the full, unpruned context):
{baseline}

CANDIDATE ANSWER (produced after context was pruned by "{model}"):
{candidate}

Rate the candidate against the reference on a 1-5 scale:
5 = equivalent quality, nothing material lost
4 = minor omissions that don't affect correctness
3 = noticeable gaps but the core answer still holds
2 = missing information that changes the answer
1 = wrong or unusable because needed context was dropped

Reply with ONLY a JSON object, no prose outside it: \
{{"score": <1-5 int>, "verdict": "pass" or "fail", "rationale": "<one sentence>"}}. \
"pass" means score >= 3."""


@dataclass
class CandidateRun:
    model: str
    capture_index: int
    scan_scope: str = "tail"
    trim_latency_ms: float | None = None
    tokens_before: int = 0
    tokens_after: int = 0
    blocks_filtered: int = 0
    local_model_used: bool = False
    judge_score: int | None = None
    judge_verdict: str | None = None
    judge_rationale: str | None = None
    error: str | None = None


@dataclass
class ModelSummary:
    model: str
    scan_scope: str
    runs: int
    errors: int
    avg_trim_latency_ms: float | None
    avg_reduction_pct: float
    avg_judge_score: float | None
    pass_rate: float | None


def estimate_plan(
    payloads: list[dict[str, Any]], models: list[str], scopes: list[str] | None = None
) -> dict[str, Any]:
    """Pre-flight call count and rough dollar estimate, before spending anything."""
    scopes = list(scopes) if scopes else ["tail"]
    # One candidate per (model, scan_scope) pair -- that is what gets scored.
    n, m = len(payloads), len(models) * len(scopes)
    baseline_calls = n
    candidate_calls = n * m
    judge_calls = n * m

    cost = 0.0
    for payload in payloads:
        tokens_in = estimate_payload_tokens(payload)
        target_price = price_per_token(payload.get("model"))
        # Baseline: one full-context call.
        cost += tokens_in * target_price + _ASSUMED_OUTPUT_TOKENS * target_price * _OUTPUT_PRICE_MULTIPLIER
        # Candidate: assume pruning saves nothing for the estimate (worst case);
        # actual runs will usually cost less than this.
        cost += m * (tokens_in * target_price + _ASSUMED_OUTPUT_TOKENS * target_price * _OUTPUT_PRICE_MULTIPLIER)
        judge_price = price_per_token(JUDGE_MODEL)
        judge_tokens_in = min(tokens_in, 3000) // 3 + 4000  # task + both answers, truncated
        cost += m * (judge_tokens_in * judge_price + _JUDGE_MAX_TOKENS * judge_price * _OUTPUT_PRICE_MULTIPLIER)

    return {
        "captures": n,
        "models": len(models),
        "scopes": len(scopes),
        "candidates": m,
        "baseline_calls": baseline_calls,
        "candidate_calls": candidate_calls,
        "judge_calls": judge_calls,
        "total_live_calls": baseline_calls + candidate_calls + judge_calls,
        "est_cost_usd": round(cost, 2),
    }


def _extract_text(message: Any) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


# Captures come from real Claude Code traffic, which routinely turns on context
# editing (`context_management` in the body). That field only exists on the beta
# messages endpoint -- the stable one rejects it outright.
_CONTEXT_MANAGEMENT_BETA = "context-management-2025-06-27"


async def _get_message(client: Any, payload: dict[str, Any]) -> Any:
    request = {k: v for k, v in payload.items() if k != "stream"}
    if "context_management" in request:
        async with client.beta.messages.stream(**request, betas=[_CONTEXT_MANAGEMENT_BETA]) as stream:
            return await stream.get_final_message()
    async with client.messages.stream(**request) as stream:
        return await stream.get_final_message()


async def _get_completion(client: Any, payload: dict[str, Any]) -> str:
    return _extract_text(await _get_message(client, payload))


def _usage_summary(message: Any) -> dict[str, int]:
    """The three input-side counters `relative_input_cost` needs, defaulting to
    0 for any the API omitted."""
    usage = getattr(message, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        "cache_creation_input_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
    }


def _parse_verdict(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "score" not in parsed:
        return None
    return parsed


async def _judge(client: Any, model: str, task: str, baseline: str, candidate: str) -> dict[str, Any] | None:
    prompt = JUDGE_PROMPT.format(
        task=task[:2500], baseline=baseline[:4000], candidate=candidate[:4000], model=model,
    )
    message = await client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=_JUDGE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_verdict(_extract_text(message))


async def run_bakeoff(
    payloads: list[dict[str, Any]],
    models: list[str],
    settings: Settings,
    client: Any,
    scopes: list[str] | None = None,
    on_event: Callable[[str], None] | None = None,
) -> list[ModelSummary]:
    notify = on_event or (lambda _msg: None)
    scopes = list(scopes) if scopes else [settings.scan_scope]

    baselines: list[str | None] = []
    for i, payload in enumerate(payloads):
        notify(f"baseline {i + 1}/{len(payloads)}")
        try:
            baselines.append(await _get_completion(client, payload))
        except Exception as exc:  # noqa: BLE001 - one bad capture shouldn't kill the run
            notify(f"baseline {i + 1} failed: {exc}")
            baselines.append(None)

    runs: list[CandidateRun] = []
    for model in models:
        local = LocalModel(model=model, base=settings.ollama_base, timeout_s=settings.local_timeout_s)
        for scope in scopes:
            # A per-candidate copy so one scope's setting never leaks into the
            # next candidate (or back into the caller's settings).
            scope_settings = replace(settings, scan_scope=scope)
            label = f"{model} ({scope})"
            for i, payload in enumerate(payloads):
                baseline = baselines[i]
                run = CandidateRun(model=model, capture_index=i, scan_scope=scope)
                if baseline is None:
                    run.error = "no baseline answer to compare against"
                    runs.append(run)
                    continue

                with tempfile.TemporaryDirectory() as tmp:
                    store = Store(Path(tmp) / "bakeoff.db")
                    pruner = Pruner(scope_settings, store, local)
                    started = time.perf_counter()
                    try:
                        result = await pruner.prune(
                            payload, session_key=f"bakeoff-{model}-{scope}-{i}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        store.close()
                        run.error = f"prune failed: {exc}"
                        runs.append(run)
                        notify(f"{label} [{i + 1}/{len(payloads)}]: prune failed")
                        continue
                    run.trim_latency_ms = round((time.perf_counter() - started) * 1000, 1)
                    store.close()

                run.tokens_before = result.est_before
                run.tokens_after = result.est_after
                run.blocks_filtered = result.blocks_filtered
                run.local_model_used = result.local_model_used

                try:
                    candidate_answer = await _get_completion(client, result.payload)
                except Exception as exc:  # noqa: BLE001
                    run.error = f"candidate answer failed: {exc}"
                    runs.append(run)
                    notify(f"{label} [{i + 1}/{len(payloads)}]: answer call failed")
                    continue

                try:
                    verdict = await _judge(
                        client, label, _recent_task_text(payload), baseline, candidate_answer
                    )
                except Exception as exc:  # noqa: BLE001
                    verdict = None
                    run.error = f"judge call failed: {exc}"

                if verdict is not None:
                    run.judge_score = verdict.get("score")
                    run.judge_verdict = verdict.get("verdict")
                    run.judge_rationale = verdict.get("rationale")
                elif not run.error:
                    run.error = "judge returned unparseable output"

                runs.append(run)
                notify(
                    f"{label} [{i + 1}/{len(payloads)}]: "
                    f"score={run.judge_score} verdict={run.judge_verdict}"
                )

    return summarize(runs, models, scopes)


def summarize(
    runs: list[CandidateRun], models: list[str], scopes: list[str] | None = None
) -> list[ModelSummary]:
    scopes = list(scopes) if scopes else sorted({r.scan_scope for r in runs}) or ["tail"]
    summaries = []
    for model in models:
        for scope in scopes:
            rows = [r for r in runs if r.model == model and r.scan_scope == scope]
            errors = [r for r in rows if r.error]
            scored = [r for r in rows if r.judge_score is not None]
            timed = [r for r in rows if r.trim_latency_ms is not None]
            reductions = [
                1 - r.tokens_after / r.tokens_before for r in rows if r.tokens_before
            ]
            summaries.append(ModelSummary(
                model=model,
                scan_scope=scope,
                runs=len(rows),
                errors=len(errors),
                avg_trim_latency_ms=round(sum(r.trim_latency_ms for r in timed) / len(timed), 1) if timed else None,
                avg_reduction_pct=round(100 * sum(reductions) / len(reductions), 1) if reductions else 0.0,
                avg_judge_score=round(sum(r.judge_score for r in scored) / len(scored), 2) if scored else None,
                pass_rate=round(sum(1 for r in scored if r.judge_verdict == "pass") / len(scored), 2) if scored else None,
            ))
    return summaries


def format_table(summaries: list[ModelSummary]) -> str:
    header = (
        f"{'model':<28}  {'scope':<5}  {'runs':>5}  {'errs':>5}  {'avg score':>9}  "
        f"{'pass%':>6}  {'reduction':>9}  {'trim ms':>8}"
    )
    lines = [header, "-" * len(header)]
    for s in summaries:
        score = f"{s.avg_judge_score:.2f}" if s.avg_judge_score is not None else "—"
        pass_pct = f"{s.pass_rate * 100:.0f}%" if s.pass_rate is not None else "—"
        trim = f"{s.avg_trim_latency_ms:.0f}" if s.avg_trim_latency_ms is not None else "—"
        lines.append(
            f"{s.model:<28}  {s.scan_scope:<5}  {s.runs:>5}  {s.errors:>5}  {score:>9}  "
            f"{pass_pct:>6}  {s.avg_reduction_pct:>8}%  {trim:>8}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------- replay
#
# The bake-off above scores answer quality on one-shot payloads, which is the
# wrong instrument for cache economics: every call starts cold, so the
# cache-write penalty that makes wide pruning expensive never appears, and
# `full` will always look strictly better than it is.
#
# A replay walks one real conversation turn by turn, in order, against a warm
# prefix, and reads the actual `usage` counters back from the API. That is the
# only way to see whether wider pruning earns back the prefix it invalidates.
# The `off` arm sends every turn unmodified -- the do-nothing baseline the
# whole cache-safety argument is measured against.


@dataclass
class ReplayTurn:
    scan_scope: str
    turn_index: int
    tokens_before: int = 0
    tokens_after: int = 0
    blocks_filtered: int = 0
    input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    error: str | None = None


@dataclass
class ReplaySummary:
    model: str
    scan_scope: str
    turns: int
    errors: int
    avg_reduction_pct: float
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    relative_input_cost: float
    est_cost_usd: float


def estimate_replay_plan(
    session: list[dict[str, Any]], models: list[str], scopes: list[str] | None = None
) -> dict[str, Any]:
    """Pre-flight for a replay. Worst case: assumes no cache hits at all, which
    is exactly what the replay exists to disprove."""
    scopes = list(scopes) if scopes else ["off", "tail", "full"]
    arms = len(models) * len(scopes)

    cost = 0.0
    for payload in session:
        tokens_in = estimate_payload_tokens(payload)
        price = price_per_token(payload.get("model"))
        cost += arms * (
            tokens_in * price + _ASSUMED_OUTPUT_TOKENS * price * _OUTPUT_PRICE_MULTIPLIER
        )

    return {
        "turns": len(session),
        "models": len(models),
        "scopes": len(scopes),
        "arms": arms,
        "total_live_calls": len(session) * arms,
        "est_cost_usd": round(cost, 2),
    }


async def run_replay(
    session: list[dict[str, Any]],
    models: list[str],
    settings: Settings,
    client: Any,
    scopes: list[str] | None = None,
    ttl: str = "5m",
    cooldown_s: float = 0.0,
    on_event: Callable[[str], None] | None = None,
) -> list[ReplaySummary]:
    notify = on_event or (lambda _msg: None)
    scopes = list(scopes) if scopes else ["off", "tail", "full"]

    total_arms = len(models) * len(scopes)
    arm_index = 0

    summaries: list[ReplaySummary] = []
    for model in models:
        local = LocalModel(model=model, base=settings.ollama_base, timeout_s=settings.local_timeout_s)
        for scope in scopes:
            label = f"{model} ({scope})"
            turns: list[ReplayTurn] = []

            with tempfile.TemporaryDirectory() as tmp:
                # One store for the whole lineage, deliberately. Decisions
                # memoise by content hash, and that memo is precisely what
                # keeps a filtered prefix byte-stable from one turn to the
                # next. A fresh store per turn would re-decide every turn and
                # thrash the very prefix this measurement is about.
                store = Store(Path(tmp) / "replay.db")
                arm_settings = settings if scope == "off" else replace(settings, scan_scope=scope)
                pruner = Pruner(arm_settings, store, local)

                for i, payload in enumerate(session):
                    turn = ReplayTurn(scan_scope=scope, turn_index=i)
                    sent = payload
                    try:
                        if scope == "off":
                            turn.tokens_before = estimate_payload_tokens(payload)
                            turn.tokens_after = turn.tokens_before
                        else:
                            result = await pruner.prune(
                                payload, session_key=f"replay-{model}-{scope}"
                            )
                            sent = result.payload
                            turn.tokens_before = result.est_before
                            turn.tokens_after = result.est_after
                            turn.blocks_filtered = result.blocks_filtered
                    except Exception as exc:  # noqa: BLE001 - one bad turn shouldn't kill the arm
                        turn.error = f"prune failed: {exc}"
                        turns.append(turn)
                        notify(f"{label} turn {i + 1}/{len(session)}: prune failed")
                        continue

                    try:
                        message = await _get_message(client, sent)
                    except Exception as exc:  # noqa: BLE001
                        turn.error = f"call failed: {exc}"
                        turns.append(turn)
                        notify(f"{label} turn {i + 1}/{len(session)}: call failed: {exc}")
                        continue

                    usage = _usage_summary(message)
                    turn.input_tokens = usage["input_tokens"]
                    turn.cache_read_input_tokens = usage["cache_read_input_tokens"]
                    turn.cache_creation_input_tokens = usage["cache_creation_input_tokens"]
                    turns.append(turn)
                    notify(
                        f"{label} turn {i + 1}/{len(session)}: "
                        f"fresh={turn.input_tokens} read={turn.cache_read_input_tokens} "
                        f"write={turn.cache_creation_input_tokens}"
                    )

                store.close()

            summaries.append(summarize_replay(model, scope, turns, session, ttl))

            # Arms must not share upstream cache entries. In tail mode the
            # conversation history is byte-identical across arms, so without a
            # gap the first arm pays every cold-start write and the rest read
            # those entries back for free -- which reads as a saving that isn't
            # real. Waiting past the cache TTL makes each arm start cold and
            # pay its own way.
            arm_index += 1
            if cooldown_s > 0 and arm_index < total_arms:
                notify(
                    f"cooling down {cooldown_s:.0f}s so the next arm starts "
                    "against an expired cache"
                )
                await asyncio.sleep(cooldown_s)

    return summaries


def summarize_replay(
    model: str,
    scope: str,
    turns: list[ReplayTurn],
    session: list[dict[str, Any]],
    ttl: str = "5m",
) -> ReplaySummary:
    ok = [t for t in turns if not t.error]
    reductions = [1 - t.tokens_after / t.tokens_before for t in ok if t.tokens_before]
    totals = {
        "input_tokens": sum(t.input_tokens for t in ok),
        "cache_read_input_tokens": sum(t.cache_read_input_tokens for t in ok),
        "cache_creation_input_tokens": sum(t.cache_creation_input_tokens for t in ok),
    }
    rel = relative_input_cost(totals, ttl)
    price = price_per_token(session[0].get("model")) if session else 0.0

    return ReplaySummary(
        model=model,
        scan_scope=scope,
        turns=len(turns),
        errors=len(turns) - len(ok),
        avg_reduction_pct=round(100 * sum(reductions) / len(reductions), 1) if reductions else 0.0,
        input_tokens=totals["input_tokens"],
        cache_read_tokens=totals["cache_read_input_tokens"],
        cache_write_tokens=totals["cache_creation_input_tokens"],
        relative_input_cost=round(rel, 1),
        est_cost_usd=round(rel * price, 4),
    )


def format_replay_table(summaries: list[ReplaySummary]) -> str:
    header = (
        f"{'model':<24}  {'scope':<5}  {'turns':>5}  {'errs':>4}  {'reduction':>9}  "
        f"{'fresh':>11}  {'cache rd':>11}  {'cache wr':>11}  {'rel cost':>12}  {'est $':>9}"
    )
    lines = [header, "-" * len(header)]
    for s in summaries:
        lines.append(
            f"{s.model:<24}  {s.scan_scope:<5}  {s.turns:>5}  {s.errors:>4}  "
            f"{s.avg_reduction_pct:>8}%  {s.input_tokens:>11,}  {s.cache_read_tokens:>11,}  "
            f"{s.cache_write_tokens:>11,}  {s.relative_input_cost:>12,.0f}  {s.est_cost_usd:>9.4f}"
        )
    return "\n".join(lines)
