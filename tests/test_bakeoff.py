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
    _parse_verdict,
    estimate_plan,
    format_table,
    run_bakeoff,
    summarize,
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
        ModelSummary(model="good-model", runs=2, errors=0, avg_trim_latency_ms=120.0,
                     avg_reduction_pct=60.0, avg_judge_score=4.5, pass_rate=1.0),
        ModelSummary(model="broken-model", runs=1, errors=1, avg_trim_latency_ms=None,
                     avg_reduction_pct=0.0, avg_judge_score=None, pass_rate=None),
    ]

    table = format_table(summaries)

    assert "good-model" in table
    assert "broken-model" in table
    assert "4.50" in table
    assert "—" in table
