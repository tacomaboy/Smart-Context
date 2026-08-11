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

import json
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .local_model import LocalModel
from .pruner import Pruner, _recent_task_text, estimate_payload_tokens
from .store import Store
from .tokens import price_per_token

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
    runs: int
    errors: int
    avg_trim_latency_ms: float | None
    avg_reduction_pct: float
    avg_judge_score: float | None
    pass_rate: float | None


def estimate_plan(payloads: list[dict[str, Any]], models: list[str]) -> dict[str, Any]:
    """Pre-flight call count and rough dollar estimate, before spending anything."""
    n, m = len(payloads), len(models)
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
        "models": m,
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


async def _get_completion(client: Any, payload: dict[str, Any]) -> str:
    request = {k: v for k, v in payload.items() if k != "stream"}
    if "context_management" in request:
        async with client.beta.messages.stream(**request, betas=[_CONTEXT_MANAGEMENT_BETA]) as stream:
            message = await stream.get_final_message()
    else:
        async with client.messages.stream(**request) as stream:
            message = await stream.get_final_message()
    return _extract_text(message)


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
    on_event: Callable[[str], None] | None = None,
) -> list[ModelSummary]:
    notify = on_event or (lambda _msg: None)

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
        for i, payload in enumerate(payloads):
            baseline = baselines[i]
            run = CandidateRun(model=model, capture_index=i)
            if baseline is None:
                run.error = "no baseline answer to compare against"
                runs.append(run)
                continue

            with tempfile.TemporaryDirectory() as tmp:
                store = Store(Path(tmp) / "bakeoff.db")
                pruner = Pruner(settings, store, local)
                started = time.perf_counter()
                try:
                    result = await pruner.prune(payload, session_key=f"bakeoff-{model}-{i}")
                except Exception as exc:  # noqa: BLE001
                    store.close()
                    run.error = f"prune failed: {exc}"
                    runs.append(run)
                    notify(f"{model} [{i + 1}/{len(payloads)}]: prune failed")
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
                notify(f"{model} [{i + 1}/{len(payloads)}]: answer call failed")
                continue

            try:
                verdict = await _judge(client, model, _recent_task_text(payload), baseline, candidate_answer)
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
            notify(f"{model} [{i + 1}/{len(payloads)}]: score={run.judge_score} verdict={run.judge_verdict}")

    return summarize(runs, models)


def summarize(runs: list[CandidateRun], models: list[str]) -> list[ModelSummary]:
    summaries = []
    for model in models:
        rows = [r for r in runs if r.model == model]
        errors = [r for r in rows if r.error]
        scored = [r for r in rows if r.judge_score is not None]
        timed = [r for r in rows if r.trim_latency_ms is not None]
        reductions = [
            1 - r.tokens_after / r.tokens_before for r in rows if r.tokens_before
        ]
        summaries.append(ModelSummary(
            model=model,
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
        f"{'model':<28}  {'runs':>5}  {'errs':>5}  {'avg score':>9}  "
        f"{'pass%':>6}  {'reduction':>9}  {'trim ms':>8}"
    )
    lines = [header, "-" * len(header)]
    for s in summaries:
        score = f"{s.avg_judge_score:.2f}" if s.avg_judge_score is not None else "—"
        pass_pct = f"{s.pass_rate * 100:.0f}%" if s.pass_rate is not None else "—"
        trim = f"{s.avg_trim_latency_ms:.0f}" if s.avg_trim_latency_ms is not None else "—"
        lines.append(
            f"{s.model:<28}  {s.runs:>5}  {s.errors:>5}  {score:>9}  "
            f"{pass_pct:>6}  {s.avg_reduction_pct:>8}%  {trim:>8}"
        )
    return "\n".join(lines)
