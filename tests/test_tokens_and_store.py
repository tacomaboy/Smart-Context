from __future__ import annotations

import pytest

from smartcontext.store import Store, session_key_for
from smartcontext.tokens import block_text, price_per_token, relative_input_cost, usage_summary


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


# ------------------------------------------------------------------ tokens


def test_usage_summary_totals_all_three_input_buckets():
    """input_tokens is only the uncached remainder; reading it alone badly
    understates a cached conversation."""
    summary = usage_summary(
        {
            "input_tokens": 500,
            "cache_read_input_tokens": 40_000,
            "cache_creation_input_tokens": 1_000,
            "output_tokens": 300,
        }
    )
    assert summary["total_prompt_tokens"] == 41_500


def test_usage_summary_tolerates_missing_fields():
    assert usage_summary(None)["total_prompt_tokens"] == 0
    assert usage_summary({})["output_tokens"] == 0


def test_pruning_that_busts_the_cache_costs_more_than_not_pruning():
    """The core economic claim behind the whole design, as an executable check.

    100k cached tokens read back cost 10k units. Cutting the context in half but
    re-writing it from scratch costs 62.5k units -- over six times more.
    """
    cached = usage_summary({"input_tokens": 0, "cache_read_input_tokens": 100_000})
    pruned_uncached = usage_summary({"cache_creation_input_tokens": 50_000})

    assert relative_input_cost(cached) == pytest.approx(10_000)
    assert relative_input_cost(pruned_uncached) == pytest.approx(62_500)
    assert relative_input_cost(pruned_uncached) > relative_input_cost(cached)


def test_one_hour_ttl_writes_cost_double():
    usage = usage_summary({"cache_creation_input_tokens": 1_000})
    assert relative_input_cost(usage, ttl="5m") == pytest.approx(1_250)
    assert relative_input_cost(usage, ttl="1h") == pytest.approx(2_000)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-opus-5", 5.00 / 1_000_000),
        ("claude-sonnet-4-6-20261001", 3.00 / 1_000_000),
        ("claude-haiku-4-5", 1.00 / 1_000_000),
        ("claude-fable-5", 10.00 / 1_000_000),
        ("some-unrecognised-model", 3.00 / 1_000_000),
        (None, 3.00 / 1_000_000),
    ],
)
def test_price_per_token_matches_by_substring(model, expected):
    assert price_per_token(model) == pytest.approx(expected)


@pytest.mark.parametrize(
    "block,expected",
    [
        ("plain", "plain"),
        ({"type": "text", "text": "hello"}, "hello"),
        ({"type": "tool_result", "content": "output"}, "output"),
        ({"type": "tool_result", "content": [{"type": "text", "text": "a"}]}, "a"),
        ({"type": "image", "source": {}}, ""),
        (None, ""),
    ],
)
def test_block_text(block, expected):
    assert block_text(block) == expected


# ------------------------------------------------------------------- store


def test_chunk_roundtrip(store):
    handle = store.put_chunk("the quick brown fox", session_key="s1", token_est=5)
    chunk = store.get_chunk(handle)
    assert chunk is not None
    assert chunk.content == "the quick brown fox"


def test_identical_content_is_stored_once(store):
    a = store.put_chunk("same bytes", tool_use_id="t1")
    b = store.put_chunk("same bytes", tool_use_id="t1")
    assert a == b


def test_search_finds_stored_text(store):
    store.put_chunk("the migration failed on the billing table", session_key="s1")
    store.put_chunk("unrelated notes about the weather", session_key="s1")

    hits = store.search("billing")
    assert hits and "billing" in hits[0].content


def test_search_survives_malformed_fts_query(store):
    """Model-authored queries contain unbalanced quotes often enough to matter."""
    store.put_chunk("some content about parsers", session_key="s1")
    assert store.search('unbalanced " quote AND') is not None


def test_decision_roundtrip(store):
    store.put_decision("hash123", [0, 2], ["sc_aaa", "sc_bbb"])
    assert store.get_decision("hash123") == ([0, 2], ["sc_aaa", "sc_bbb"])


def test_missing_decision_is_none(store):
    assert store.get_decision("nope") is None


def test_reset_clears_requests_but_keeps_chunks_by_default(store):
    store.log_request({"ts": 1.0, "mode": "prune", "model": "claude-opus-5"})
    handle = store.put_chunk("kept unless explicitly wiped", session_key="s1")

    store.reset()

    assert store.stats() == {"requests": 0}
    assert store.get_chunk(handle) is not None


def test_reset_can_also_wipe_chunks(store):
    store.log_request({"ts": 1.0, "mode": "prune", "model": "claude-opus-5"})
    handle = store.put_chunk("gone after a full wipe", session_key="s1")
    store.put_decision("hash123", [0], ["sc_aaa"])

    store.reset(wipe_chunks=True)

    assert store.stats() == {"requests": 0}
    assert store.get_chunk(handle) is None
    assert store.get_decision("hash123") is None


def test_stats_reports_cache_hit_ratio(store):
    store.log_request(
        {
            "ts": 1.0, "mode": "prune", "model": "claude-opus-5",
            "input_tokens": 1_000, "cache_read_tokens": 9_000,
            "cache_write_tokens": 0, "output_tokens": 100,
            "relative_input_cost": 1900.0, "blocks_filtered": 1,
        }
    )
    stats = store.stats()
    assert stats["requests"] == 1
    assert stats["cache_hit_ratio"] == pytest.approx(0.9)


def test_stats_reports_tokens_saved_and_reduction_and_cost(store):
    store.log_request(
        {
            "ts": 1.0, "mode": "prune", "model": "claude-opus-5",
            "est_tokens_before": 1_000, "est_tokens_after": 250,
            "latency_local_ms": 100.0,
        }
    )
    store.log_request(
        {
            "ts": 2.0, "mode": "prune", "model": "claude-opus-5",
            "est_tokens_before": 1_000, "est_tokens_after": 1_000,
            "latency_local_ms": 300.0,
        }
    )

    stats = store.stats()

    assert stats["tokens_saved"] == 750
    assert stats["avg_reduction_pct"] == pytest.approx((0.75 + 0.0) / 2)
    assert stats["est_prompt_cost_before_usd"] == pytest.approx(2_000 * (5.00 / 1_000_000), abs=1e-4)
    assert stats["est_prompt_cost_after_usd"] == pytest.approx(1_250 * (5.00 / 1_000_000), abs=1e-4)
    assert stats["est_cost_savings_usd"] == pytest.approx(750 * (5.00 / 1_000_000), abs=1e-4)
    assert stats["est_cost_savings_pct"] == pytest.approx(750 / 2_000, abs=1e-4)
    assert stats["avg_trim_time_ms"] == pytest.approx(200.0)


def test_stats_shadow_mode_requests_excluded_from_reduction_average(store):
    """Shadow mode never filters anything -- averaging it in would understate
    how effective pruning actually is when it's turned on."""
    store.log_request(
        {
            "ts": 1.0, "mode": "shadow", "model": "claude-opus-5",
            "est_tokens_before": 1_000, "est_tokens_after": 1_000,
        }
    )

    stats = store.stats()

    assert stats["avg_reduction_pct"] == 0.0


def test_stats_tolerates_missing_latency_and_token_estimates(store):
    store.log_request({"ts": 1.0, "mode": "prune", "model": "claude-opus-5"})

    stats = store.stats()

    assert stats["tokens_saved"] == 0
    assert stats["avg_reduction_pct"] == 0.0
    assert stats["est_prompt_cost_before_usd"] == 0.0
    assert stats["est_prompt_cost_after_usd"] == 0.0
    assert stats["est_cost_savings_usd"] == 0.0
    assert stats["est_cost_savings_pct"] == 0.0
    assert stats["avg_trim_time_ms"] is None


def test_request_detail_includes_saved_prune_blocks(store):
    request_id = store.log_request(
        {
            "ts": 1.0,
            "mode": "prune",
            "model": "claude-sonnet-5",
            "prune_details": [
                {
                    "message_index": 2,
                    "block_index": 1,
                    "tool_use_id": "tool_123",
                    "before_text": "line one\nline two\nline three\n",
                    "after_text": "line one\n\n[smart-context: 2 of 3 sections elided...]",
                    "handles": ["sc_abc"],
                    "before_tokens_est": 30,
                    "after_tokens_est": 12,
                }
            ],
        }
    )

    detail = store.request_detail(request_id)

    assert detail is not None
    assert detail["id"] == request_id
    assert len(detail["prunes"]) == 1
    assert detail["prunes"][0]["tool_use_id"] == "tool_123"
    assert detail["prunes"][0]["handles"] == ["sc_abc"]


def test_request_detail_includes_latest_user_turn_preview(store):
    request_id = store.log_request(
        {
            "ts": 1.0,
            "mode": "prune",
            "model": "claude-sonnet-5",
            "latest_user_turn": "Hey what's up",
            "latest_user_message_index": 7,
            "latest_user_text_block_index": 0,
        }
    )

    detail = store.request_detail(request_id)

    assert detail is not None
    assert detail["latest_user_turn"] == "Hey what's up"
    assert detail["latest_user_message_index"] == 7
    assert detail["latest_user_text_block_index"] == 0


def test_request_detail_handles_older_databases_without_prune_table(store):
    request_id = store.log_request(
        {
            "ts": 1.0,
            "mode": "prune",
            "model": "claude-sonnet-5",
            "est_tokens_before": 100,
            "est_tokens_after": 50,
        }
    )
    store._conn.execute("DROP TABLE request_prunes")

    detail = store.request_detail(request_id)

    assert detail is not None
    assert detail["id"] == request_id
    assert detail["prunes"] == []


def test_timeline_includes_estimated_prompt_costs(store):
    store.log_request(
        {
            "ts": 1.0,
            "mode": "prune",
            "model": "claude-sonnet-5",
            "est_tokens_before": 1_000,
            "est_tokens_after": 250,
        }
    )

    row = store.timeline(1)[0]

    assert row["est_prompt_cost_before_usd"] == pytest.approx(1_000 * (3.00 / 1_000_000), abs=1e-4)
    assert row["est_prompt_cost_after_usd"] == pytest.approx(250 * (3.00 / 1_000_000), abs=1e-4)
    assert row["est_cost_savings_usd"] == pytest.approx(750 * (3.00 / 1_000_000), abs=1e-4)
    assert row["est_cost_savings_pct"] == pytest.approx(0.75, abs=1e-4)


def test_timeline_exposes_cost_price_basis(store):
    store.log_request(
        {
            "ts": 1.0,
            "mode": "prune",
            "model": "claude-opus-5",
            "est_tokens_before": 10,
            "est_tokens_after": 5,
        }
    )

    row = store.timeline(1)[0]

    assert row["est_price_family"] == "opus"
    assert row["est_input_price_per_mtoken_usd"] == 5.0


def test_session_key_is_stable_as_the_conversation_grows(store):
    base = {
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "first question"}],
    }
    grown = {
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "an answer"},
            {"role": "user", "content": "a follow up"},
        ],
    }
    assert session_key_for(base) == session_key_for(grown)


def test_session_key_differs_across_conversations():
    a = {"system": "s", "messages": [{"role": "user", "content": "one"}]}
    b = {"system": "s", "messages": [{"role": "user", "content": "two"}]}
    assert session_key_for(a) != session_key_for(b)


def test_opening_a_pre_latency_column_db_migrates_in_place(tmp_path):
    """A store created before latency_local_ms existed must not crash on open,
    and logging a request afterwards should pick up the new column."""
    import sqlite3

    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL, session_key TEXT, model TEXT, mode TEXT,
            streaming INTEGER, est_tokens_before INTEGER, est_tokens_after INTEGER,
            blocks_filtered INTEGER, local_model_used INTEGER, status INTEGER,
            input_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER,
            output_tokens INTEGER, relative_input_cost REAL, note TEXT
        )
        """
    )
    conn.commit()
    conn.close()

    store = Store(db_path)
    try:
        store.log_request(
            {
                "ts": 1.0,
                "mode": "prune",
                "model": "claude-opus-5",
                "latency_local_ms": 42.0,
                "latest_user_turn": "hello",
                "latest_user_message_index": 0,
                "latest_user_text_block_index": None,
            }
        )
        stats = store.stats()
        req = store.request_detail(1)
        assert stats["avg_trim_time_ms"] == pytest.approx(42.0)
        assert req is not None and req["latest_user_turn"] == "hello"
        assert req["latest_user_message_index"] == 0
    finally:
        store.close()
