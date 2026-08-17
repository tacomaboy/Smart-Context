"""Tests for the offline threshold sweep.

The sweep replays captures through the real Pruner without touching the
network or the live store, so the main things worth checking are: capture
loading tolerates junk files, each (min_chars, keep_budget) point gets its
own isolated store (no cross-setting memoisation), and the reported numbers
line up with what the fake local model actually decided.
"""

from __future__ import annotations

import json

import pytest

from smartcontext.config import Settings
from smartcontext.local_model import LocalDecision
from smartcontext.pruner import estimate_payload_tokens
from smartcontext import sweep as sweep_mod
from smartcontext.sweep import (
    SweepPoint,
    format_table,
    load_capture_sessions,
    load_captures,
    run_sweep,
)

BIG = "\n".join(f"line {i} some reasonably wordy filler content here" for i in range(400))


def make_payload(text: str = BIG) -> dict:
    return {
        "model": "claude-opus-5",
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
    """Keeps only the first chunk, regardless of what settings constructed it."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def select_chunks(self, task, chunks, keep_at_least=1):
        return LocalDecision(keep=[0], model="fake")

    async def available(self):
        return True


@pytest.fixture
def base_settings(tmp_path):
    return Settings(mode="prune", data_dir=tmp_path, chunk_chars=500)


# ------------------------------------------------------------------ load_captures


def test_load_captures_reads_json_files(tmp_path):
    (tmp_path / "one.json").write_text(json.dumps(make_payload()), encoding="utf-8")
    (tmp_path / "two.json").write_text(json.dumps(make_payload("short")), encoding="utf-8")

    payloads = load_captures(tmp_path)

    assert len(payloads) == 2


def test_load_capture_sessions_groups_by_session_in_time_order(tmp_path):
    """Capture filenames carry {time_ns}_{session_key}, so real conversation
    lineage is recoverable -- that is what makes a warm-prefix replay possible."""
    def write(ts: int, session: str, marker: str) -> None:
        (tmp_path / f"{ts}_{session}.json").write_text(
            json.dumps(make_payload(marker)), encoding="utf-8"
        )

    # Written out of order on purpose; grouping must sort by timestamp.
    write(300, "sessA", "a-third")
    write(100, "sessA", "a-first")
    write(200, "sessA", "a-second")
    write(150, "sessB", "b-first")
    write(250, "sessB", "b-second")
    write(999, "solo", "only-one")

    sessions = load_capture_sessions(tmp_path)

    # Longest session first; the single-capture session is dropped entirely.
    assert [len(s) for s in sessions] == [3, 2]

    def marker(payload):
        return payload["messages"][-1]["content"][0]["content"]

    assert [marker(p) for p in sessions[0]] == ["a-first", "a-second", "a-third"]
    assert [marker(p) for p in sessions[1]] == ["b-first", "b-second"]


def test_load_capture_sessions_ignores_unparseable_names(tmp_path):
    (tmp_path / "no-timestamp_sessA.json").write_text(json.dumps(make_payload()), encoding="utf-8")
    (tmp_path / "nounderscore.json").write_text(json.dumps(make_payload()), encoding="utf-8")

    assert load_capture_sessions(tmp_path) == []


def test_load_captures_skips_malformed_files(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps(make_payload()), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")

    payloads = load_captures(tmp_path)

    assert len(payloads) == 1


def test_load_captures_empty_dir_returns_empty_list(tmp_path):
    assert load_captures(tmp_path) == []


# ----------------------------------------------------------------------- run_sweep


@pytest.fixture(autouse=True)
def fake_local_model(monkeypatch):
    monkeypatch.setattr(sweep_mod, "LocalModel", FakeLocal)


async def test_run_sweep_preserves_settings_it_does_not_vary(base_settings, monkeypatch):
    """The sweep varies min_block_chars and keep_budget_chars. Every other setting
    -- scan_scope especially -- must survive into the Pruner, or the sweep reports
    numbers for a configuration nobody asked for."""
    base_settings.scan_scope = "full"
    seen: list[str] = []

    real_pruner = sweep_mod.Pruner

    class RecordingPruner(real_pruner):
        def __init__(self, s, store, local):
            seen.append(s.scan_scope)
            super().__init__(s, store, local)

    monkeypatch.setattr(sweep_mod, "Pruner", RecordingPruner)

    await run_sweep([make_payload()], base_settings, [1000, 2000])

    assert seen == ["full", "full"]


async def test_run_sweep_produces_one_point_per_min_chars_value(base_settings):
    payloads = [make_payload()]

    points = await run_sweep(payloads, base_settings, [500, 4000])

    assert [p.min_block_chars for p in points] == [500, 4000]


async def test_run_sweep_crosses_min_chars_with_keep_budget(base_settings):
    payloads = [make_payload()]

    points = await run_sweep(payloads, base_settings, [500, 4000], keep_budget_values=[100, 200])

    combos = {(p.min_block_chars, p.keep_budget_chars) for p in points}
    assert combos == {(500, 100), (500, 200), (4000, 100), (4000, 200)}


async def test_run_sweep_defaults_keep_budget_to_base_settings(base_settings):
    payloads = [make_payload()]

    points = await run_sweep(payloads, base_settings, [500])

    assert points[0].keep_budget_chars == base_settings.keep_budget_chars


async def test_run_sweep_below_min_chars_leaves_payload_untouched(base_settings):
    """A block smaller than min_block_chars is never filtered, so before == after."""
    payloads = [make_payload("short block, well under any threshold")]

    points = await run_sweep(payloads, base_settings, [4000])

    assert points[0].tokens_before == points[0].tokens_after
    assert points[0].blocks_filtered == 0


async def test_run_sweep_above_min_chars_reduces_tokens(base_settings):
    payloads = [make_payload()]
    before = estimate_payload_tokens(payloads[0])

    points = await run_sweep(payloads, base_settings, [500])

    point = points[0]
    assert point.tokens_before == before
    assert point.blocks_filtered == 1
    assert point.tokens_after < point.tokens_before
    assert point.local_model_used == 1


async def test_run_sweep_isolates_stores_across_settings(base_settings):
    """Same content hash, two different keep_budget values -- each point must
    make its own filtering decision instead of reusing a cached one computed
    under a different budget."""
    payloads = [make_payload()]

    points = await run_sweep(payloads, base_settings, [500], keep_budget_values=[100, 100_000])

    tight, loose = points
    assert tight.tokens_after <= loose.tokens_after


# ------------------------------------------------------------------- reduction_pct


def test_reduction_pct_computed_from_before_and_after():
    point = SweepPoint(
        min_block_chars=500, keep_budget_chars=1500, payload_count=1,
        tokens_before=1000, tokens_after=250, blocks_filtered=1,
        local_model_used=1, fallback_count=0,
    )
    assert point.reduction_pct == 75.0


def test_reduction_pct_zero_before_is_zero_not_a_crash():
    point = SweepPoint(
        min_block_chars=500, keep_budget_chars=1500, payload_count=0,
        tokens_before=0, tokens_after=0, blocks_filtered=0,
        local_model_used=0, fallback_count=0,
    )
    assert point.reduction_pct == 0.0


# --------------------------------------------------------------------- format_table


def test_format_table_includes_every_point():
    points = [
        SweepPoint(500, 1500, 3, 1000, 800, 2, 2, 0),
        SweepPoint(4000, 1500, 3, 1000, 950, 1, 1, 1),
    ]

    table = format_table(points)

    assert "500" in table
    assert "4000" in table
    assert table.count("\n") >= 3  # header + separator + one row per point
