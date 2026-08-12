"""Tests for the live model bake-off.

No real Ollama or Anthropic calls here -- LocalModel is monkeypatched (same
trick as test_sweep.py) and the Anthropic client is a small fake that records
what it was asked and returns scripted answers. What's worth checking:
call-count/cost estimates before any money is spent, robust judge-JSON
parsing, that a failed baseline skips every candidate for that capture
instead of guessing, and that per-model aggregation is arithmetically right.
"""

from __future__ import annotations

import pytest

from smartcontext import bakeoff as bakeoff_mod
from smartcontext.bakeoff import (
    CandidateRun,
    ReplayTurn,
    _parse_verdict,
    estimate_plan,
    estimate_replay_plan,
    format_replay_table,
    format_table,
    run_bakeoff,
    run_replay,
    summarize,
    summarize_replay,
)
from smartcontext.config import Settings
from smartcontext.local_model import LocalDecision

BIG = "\n".join(f"line {i} some reasonably wordy filler content here" for i in range(400))


def make_payload(text: str = BIG, model: str = "claude-opus-5") -> dict:
    return {
        "model": model,
        "max_tokens": 1024,
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Find the bug."}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_abc123", "name": "read", "input": {}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_abc123", "content": text}
            ]},
        ],
    }


class FakeLocal:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def select_chunks(self, task, chunks, keep_at_least=1):
        return LocalDecision(keep=[0], model="fake")

    async def available(self):
        return True


@pytest.fixture(autouse=True)
def fake_local_model(monkeypatch):
    monkeypatch.setattr(bakeoff_mod, "LocalModel", FakeLocal)


@pytest.fixture
def settings(tmp_path):
    s = Settings(mode="prune", data_dir=tmp_path)
    s.min_block_chars = 1000
    s.keep_budget_chars = 1200
    s.chunk_chars = 500
    return s


class FakeBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [FakeBlock(text)]


class FakeStream:
    def __init__(self, text: str, error: Exception | None) -> None:
        self._text = text
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get_final_message(self):
        if self._error:
            raise self._error
        return FakeMessage(self._text)


class FakeMessages:
    """Records every call; stream() calls are numbered so a test can fail a specific one."""

    def __init__(self, answer_text: str = "ANSWER", judge_text: str = '{"score": 4, "verdict": "pass", "rationale": "fine"}',
                 fail_stream_at: set[int] | None = None) -> None:
        self.answer_text = answer_text
        self.judge_text = judge_text
        self.fail_stream_at = fail_stream_at or set()
        self.stream_calls: list[dict] = []
        self.create_calls: list[dict] = []

    def stream(self, **kwargs):
        index = len(self.stream_calls)
        self.stream_calls.append(kwargs)
        error = RuntimeError("boom") if index in self.fail_stream_at else None
        return FakeStream(self.answer_text, error)

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return FakeMessage(self.judge_text)


class FakeClient:
    def __init__(self, **kwargs) -> None:
        self.messages = FakeMessages(**kwargs)


# --------------------------------------------------------------- estimate_plan


def test_estimate_plan_counts_scale_with_captures_and_models():
    payloads = [make_payload(), make_payload()]
    models = ["m1", "m2", "m3"]

    plan = estimate_plan(payloads, models)

    assert plan["baseline_calls"] == 2
    assert plan["candidate_calls"] == 6
    assert plan["judge_calls"] == 6
    assert plan["total_live_calls"] == 14
    assert plan["est_cost_usd"] > 0


def test_estimate_plan_scales_with_scan_scopes():
    """Each (model, scope) pair is a separate candidate, so listing both scopes
    doubles the billed candidate and judge calls."""
    payloads = [make_payload(), make_payload()]
    models = ["m1", "m2"]

    tail_only = estimate_plan(payloads, models, ["tail"])
    both = estimate_plan(payloads, models, ["tail", "full"])

    assert tail_only["candidates"] == 2
    assert both["candidates"] == 4
    # Baselines are per-capture and shared across candidates, so they don't scale.
    assert both["baseline_calls"] == tail_only["baseline_calls"]
    assert both["candidate_calls"] == 2 * tail_only["candidate_calls"]
    assert both["judge_calls"] == 2 * tail_only["judge_calls"]
    assert both["est_cost_usd"] > tail_only["est_cost_usd"]


def test_summarize_separates_scan_scopes():
    """tail and full runs of the same model must not be averaged together --
    that is the whole point of the comparison."""
    runs = [
        CandidateRun(model="m", capture_index=0, scan_scope="tail", judge_score=5,
                     judge_verdict="pass", tokens_before=1000, tokens_after=900),
        CandidateRun(model="m", capture_index=0, scan_scope="full", judge_score=3,
                     judge_verdict="fail", tokens_before=1000, tokens_after=400),
    ]

    summaries = summarize(runs, ["m"], ["tail", "full"])

    assert len(summaries) == 2
    tail = next(s for s in summaries if s.scan_scope == "tail")
    full = next(s for s in summaries if s.scan_scope == "full")
    assert tail.runs == 1 and full.runs == 1
    assert tail.avg_judge_score == pytest.approx(5.0)
    assert full.avg_judge_score == pytest.approx(3.0)
    # full trims harder, which is exactly the trade being measured.
    assert full.avg_reduction_pct > tail.avg_reduction_pct


def test_estimate_plan_is_free_of_side_effects():
    """Must not spend anything or touch the network -- pure arithmetic."""
    payloads = [make_payload()]
    plan1 = estimate_plan(payloads, ["m1"])
    plan2 = estimate_plan(payloads, ["m1"])
    assert plan1 == plan2


# ---------------------------------------------------------------- parse_verdict


def test_parse_verdict_reads_clean_json():
    verdict = _parse_verdict('{"score": 5, "verdict": "pass", "rationale": "great"}')
    assert verdict == {"score": 5, "verdict": "pass", "rationale": "great"}


def test_parse_verdict_tolerates_surrounding_prose():
    verdict = _parse_verdict('Sure, here is my rating:\n{"score": 2, "verdict": "fail", "rationale": "lost detail"}\nDone.')
    assert verdict["score"] == 2


def test_parse_verdict_rejects_garbage():
    assert _parse_verdict("I refuse to output JSON.") is None
    assert _parse_verdict("") is None
    assert _parse_verdict("{not valid json}") is None


def test_parse_verdict_rejects_json_without_score_field():
    assert _parse_verdict('{"verdict": "pass"}') is None


# ------------------------------------------------------------------ run_bakeoff


async def test_run_bakeoff_scores_every_model_against_the_shared_baseline(settings):
    payloads = [make_payload()]
    client = FakeClient()

    summaries = await run_bakeoff(payloads, ["model-a", "model-b"], settings, client)

    assert len(client.messages.stream_calls) == 3  # 1 baseline + 2 candidates
    assert len(client.messages.create_calls) == 2  # 1 judge per candidate
    assert {s.model for s in summaries} == {"model-a", "model-b"}
    for s in summaries:
        assert s.runs == 1
        assert s.errors == 0
        assert s.avg_judge_score == pytest.approx(4.0)
        assert s.pass_rate == pytest.approx(1.0)
        assert s.avg_reduction_pct > 0  # BIG content is well over min_block_chars


async def test_run_bakeoff_skips_candidates_when_baseline_fails(settings):
    payloads = [make_payload()]
    client = FakeClient(fail_stream_at={0})  # the one baseline call

    summaries = await run_bakeoff(payloads, ["model-a", "model-b"], settings, client)

    # No candidate or judge calls should have been made for a capture with no reference.
    assert len(client.messages.stream_calls) == 1
    assert len(client.messages.create_calls) == 0
    for s in summaries:
        assert s.runs == 1
        assert s.errors == 1
        assert s.avg_judge_score is None


async def test_run_bakeoff_records_unparseable_judge_output_as_error(settings):
    payloads = [make_payload()]
    client = FakeClient(judge_text="not json at all")

    summaries = await run_bakeoff(payloads, ["model-a"], settings, client)

    assert summaries[0].errors == 1
    assert summaries[0].avg_judge_score is None


# -------------------------------------------------------------------- summarize


def test_summarize_averages_only_over_scored_runs():
    runs = [
        CandidateRun(model="m", capture_index=0, judge_score=5, judge_verdict="pass",
                     trim_latency_ms=100.0, tokens_before=1000, tokens_after=500),
        CandidateRun(model="m", capture_index=1, judge_score=3, judge_verdict="pass",
                     trim_latency_ms=300.0, tokens_before=1000, tokens_after=500),
        CandidateRun(model="m", capture_index=2, error="prune failed"),
    ]

    summaries = summarize(runs, ["m"])

    s = summaries[0]
    assert s.runs == 3
    assert s.errors == 1
    assert s.avg_judge_score == pytest.approx(4.0)
    assert s.pass_rate == pytest.approx(1.0)
    assert s.avg_trim_latency_ms == pytest.approx(200.0)
    assert s.avg_reduction_pct == pytest.approx(50.0)


def test_summarize_handles_a_model_with_no_successful_runs():
    runs = [CandidateRun(model="m", capture_index=0, error="prune failed")]
    s = summarize(runs, ["m"])[0]
    assert s.avg_judge_score is None
    assert s.pass_rate is None
    assert s.avg_trim_latency_ms is None
    assert s.avg_reduction_pct == 0.0


# ------------------------------------------------------------------ format_table


def test_format_table_renders_dashes_for_missing_values():
    from smartcontext.bakeoff import ModelSummary

    summaries = [
        ModelSummary(model="good-model", scan_scope="tail", runs=2, errors=0,
                     avg_trim_latency_ms=120.0, avg_reduction_pct=60.0,
                     avg_judge_score=4.5, pass_rate=1.0),
        ModelSummary(model="broken-model", scan_scope="full", runs=1, errors=1,
                     avg_trim_latency_ms=None, avg_reduction_pct=0.0,
                     avg_judge_score=None, pass_rate=None),
    ]

    table = format_table(summaries)

    assert "good-model" in table
    assert "broken-model" in table
    assert "4.50" in table
    assert "—" in table
    # The scope axis has to be visible, or tail/full rows are indistinguishable.
    assert "tail" in table
    assert "full" in table


# --------------------------------------------------------------------- replay


class CountingLocal(FakeLocal):
    """Counts select_chunks calls so a test can prove decisions were memoised."""

    calls = 0

    async def select_chunks(self, task, chunks, keep_at_least=1):
        type(self).calls += 1
        return LocalDecision(keep=[0], model="fake")


def test_estimate_replay_plan_scales_with_arms():
    session = [make_payload(), make_payload(), make_payload()]

    one = estimate_replay_plan(session, ["m1"], ["off"])
    three = estimate_replay_plan(session, ["m1"], ["off", "tail", "full"])

    assert one["arms"] == 1
    assert three["arms"] == 3
    assert one["total_live_calls"] == 3
    assert three["total_live_calls"] == 9
    assert three["est_cost_usd"] > one["est_cost_usd"]


async def test_replay_shares_one_store_across_turns(settings, monkeypatch):
    """The whole point of a replay is a stable prefix. Decisions memoise by
    content hash, so an identical block appearing in consecutive turns must be
    decided exactly once -- re-deciding would rewrite the prefix each turn and
    destroy the cache behaviour being measured."""
    CountingLocal.calls = 0
    monkeypatch.setattr(bakeoff_mod, "LocalModel", CountingLocal)

    session = [make_payload(), make_payload()]
    client = FakeClient()

    summaries = await run_replay(session, ["m1"], settings, client, scopes=["tail"])

    assert len(summaries) == 1
    assert summaries[0].turns == 2
    assert summaries[0].errors == 0
    assert CountingLocal.calls == 1


async def test_replay_off_arm_sends_payloads_unmodified(settings):
    """The baseline arm must be a true do-nothing control."""
    session = [make_payload(), make_payload()]
    client = FakeClient()

    summaries = await run_replay(session, ["m1"], settings, client, scopes=["off"])

    assert summaries[0].scan_scope == "off"
    assert summaries[0].avg_reduction_pct == 0.0
    for sent in client.messages.stream_calls:
        assert sent["messages"] == session[0]["messages"]


def test_summarize_replay_prices_reads_and_writes():
    """rel cost = fresh x1.0 + cache read x0.1 + cache write x1.25 (5m)."""
    turns = [
        ReplayTurn(scan_scope="tail", turn_index=0, tokens_before=1000, tokens_after=800,
                   input_tokens=100, cache_read_input_tokens=1000,
                   cache_creation_input_tokens=200),
    ]

    s = summarize_replay("m1", "tail", turns, [make_payload()], ttl="5m")

    assert s.input_tokens == 100
    assert s.cache_read_tokens == 1000
    assert s.cache_write_tokens == 200
    assert s.relative_input_cost == pytest.approx(100 + 100 + 250)
    assert s.avg_reduction_pct == pytest.approx(20.0)


def test_summarize_replay_charges_more_for_1h_writes():
    turns = [ReplayTurn(scan_scope="full", turn_index=0, cache_creation_input_tokens=1000)]

    five_min = summarize_replay("m1", "full", turns, [make_payload()], ttl="5m")
    one_hour = summarize_replay("m1", "full", turns, [make_payload()], ttl="1h")

    assert one_hour.relative_input_cost > five_min.relative_input_cost


def test_format_replay_table_shows_every_arm():
    summaries = [
        summarize_replay("m1", scope, [ReplayTurn(scan_scope=scope, turn_index=0)], [make_payload()])
        for scope in ("off", "tail", "full")
    ]

    table = format_replay_table(summaries)

    for scope in ("off", "tail", "full"):
        assert scope in table
    assert "cache wr" in table
