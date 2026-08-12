"""Tests for the request-mutation invariants.

These are the tests that matter. A context proxy that saves tokens but
occasionally corrupts a request is worse than no proxy at all.
"""

from __future__ import annotations

import pytest

from smartcontext.config import Settings
from smartcontext.local_model import LocalDecision, _parse_indices
from smartcontext.pruner import Pruner, split_chunks
from smartcontext.store import Store

BIG = "\n".join(f"line {i} some reasonably wordy filler content here" for i in range(400))


class FakeLocal:
    """Keeps only the first chunk, and counts calls so we can prove memoisation."""

    def __init__(self, decision: list[int] | None = None, fail: bool = False) -> None:
        self.decision = decision
        self.fail = fail
        self.calls = 0

    async def select_chunks(self, task, chunks, keep_at_least=1):
        self.calls += 1
        if self.fail:
            return None
        keep = self.decision if self.decision is not None else [0]
        return LocalDecision(keep=[i for i in keep if i < len(chunks)], model="fake")

    async def available(self):
        return not self.fail


@pytest.fixture
def settings(tmp_path):
    s = Settings(mode="prune", data_dir=tmp_path)
    s.min_block_chars = 1000
    s.keep_budget_chars = 1200
    s.chunk_chars = 500
    return s


@pytest.fixture
def store(settings):
    s = Store(settings.db_path)
    yield s
    s.close()


def payload_with_tool_result(text: str = BIG, **block_extra) -> dict:
    block = {"type": "tool_result", "tool_use_id": "toolu_abc123", "content": text}
    block.update(block_extra)
    return {
        "model": "claude-opus-5",
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Find the bug."}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_abc123", "name": "read", "input": {}}
            ]},
            {"role": "user", "content": [block]},
        ],
    }


async def test_oversized_tool_result_is_shrunk(settings, store):
    pruner = Pruner(settings, store, FakeLocal())
    result = await pruner.prune(payload_with_tool_result(), "sess1")

    assert result.modified
    assert result.est_after < result.est_before
    assert result.handles


async def test_tool_result_block_is_never_removed(settings, store):
    """Dropping a tool_result whose tool_use survives is a hard 400 upstream."""
    pruner = Pruner(settings, store, FakeLocal())
    result = await pruner.prune(payload_with_tool_result(), "sess1")

    blocks = result.payload["messages"][-1]["content"]
    assert len(blocks) == 1
    assert blocks[0]["type"] == "tool_result"
    assert blocks[0]["tool_use_id"] == "toolu_abc123"


async def test_is_error_flag_survives(settings, store):
    pruner = Pruner(settings, store, FakeLocal())
    result = await pruner.prune(payload_with_tool_result(is_error=True), "sess1")

    assert result.payload["messages"][-1]["content"][0]["is_error"] is True


async def test_earlier_non_filterable_content_is_untouched(settings, store):
    """Nothing about the earlier messages qualifies (too small / wrong type) --
    they should come through byte-identical, not because of message position."""
    original = payload_with_tool_result()
    earlier = [m for m in original["messages"][:-1]]

    pruner = Pruner(settings, store, FakeLocal())
    result = await pruner.prune(original, "sess1")

    assert result.payload["messages"][:-1] == earlier
    assert result.payload["system"] == original["system"]


async def test_earlier_oversized_tool_result_is_not_filtered(settings, store):
    """Only the newest user turn is eligible; historical turns stay untouched
    to preserve cacheable prefixes."""
    payload = payload_with_tool_result()
    # Push the big block into history and end on a plain user question.
    payload["messages"].append({"role": "assistant", "content": [{"type": "text", "text": "ok"}]})
    payload["messages"].append({"role": "user", "content": [{"type": "text", "text": "and now?"}]})

    pruner = Pruner(settings, store, FakeLocal())
    result = await pruner.prune(payload, "sess1")

    assert not result.modified
    assert result.payload == payload


async def test_full_scope_filters_earlier_turns(settings, store):
    """scan_scope='full' restores whole-history filtering: the very block that
    tail mode leaves alone is trimmed here."""
    settings.scan_scope = "full"
    payload = payload_with_tool_result()
    payload["messages"].append({"role": "assistant", "content": [{"type": "text", "text": "ok"}]})
    payload["messages"].append({"role": "user", "content": [{"type": "text", "text": "and now?"}]})
    big_index = len(payload["messages"]) - 3

    pruner = Pruner(settings, store, FakeLocal())
    result = await pruner.prune(payload, "sess1")

    assert result.modified
    assert result.blocks_filtered == 1
    # The historical block was rewritten; the original payload is untouched.
    assert result.payload["messages"][big_index] != payload["messages"][big_index]
    assert result.est_after < result.est_before


async def test_full_scope_still_skips_earlier_cache_control(settings, store):
    """Widening the scope must not override the cache_control guard on history."""
    settings.scan_scope = "full"
    payload = {
        "model": "claude-opus-5",
        "system": "You are a helpful assistant.",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_old", "content": BIG, "cache_control": {"type": "ephemeral"}},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": [{"type": "text", "text": "and now?"}]},
        ],
    }
    pruner = Pruner(settings, store, FakeLocal())
    result = await pruner.prune(payload, "sess1")

    assert result.payload["messages"][0]["content"][0]["content"] == BIG


def payload_with_recalled_block(tool_name: str, text: str = BIG) -> dict:
    """A turn where Claude asked for elided context and got it back."""
    return {
        "model": "claude-opus-5",
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "What was in that file?"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_recall", "name": tool_name,
                 "input": {"handle": "sc_abc123"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_recall", "content": text},
            ]},
        ],
    }


@pytest.mark.parametrize(
    "tool_name",
    ["context_get", "context_recall", "context_recent", "mcp__smart-context__context_get"],
)
async def test_recalled_context_is_never_re_trimmed(settings, store, tool_name):
    """Claude asking for elided text must receive it. Trimming the recall result
    is a loop: it hands back another handle instead of the content."""
    pruner = Pruner(settings, store, FakeLocal())
    payload = payload_with_recalled_block(tool_name)

    result = await pruner.prune(payload, "sess1")

    assert not result.modified
    assert result.payload["messages"][-1]["content"][0]["content"] == BIG


async def test_recall_guard_does_not_spare_ordinary_tool_results(settings, store):
    """The guard keys on the originating tool, not on size -- a normal oversized
    tool_result in the same shape must still be filtered."""
    pruner = Pruner(settings, store, FakeLocal())
    payload = payload_with_recalled_block("read_file")

    result = await pruner.prune(payload, "sess1")

    assert result.modified
    assert result.blocks_filtered == 1


async def test_fails_open_when_local_model_is_down(settings, store):
    """Ollama being unreachable must degrade to passthrough, never to an error."""
    pruner = Pruner(settings, store, FakeLocal(fail=True))
    payload = payload_with_tool_result()
    result = await pruner.prune(payload, "sess1")

    assert not result.modified
    assert result.payload == payload


async def test_cache_control_blocks_in_newest_turn_can_still_be_filtered(settings, store):
    """Newest-turn cache_control blocks can be trimmed and should keep the marker."""
    pruner = Pruner(settings, store, FakeLocal())
    payload = payload_with_tool_result(cache_control={"type": "ephemeral", "ttl": "1h"})
    result = await pruner.prune(payload, "sess1")

    assert result.modified
    block = result.payload["messages"][-1]["content"][0]
    assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_cache_control_blocks_in_earlier_turns_are_still_skipped(settings, store):
    """Older cache breakpoints remain off-limits."""
    payload = {
        "model": "claude-opus-5",
        "system": "You are a helpful assistant.",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_old", "content": BIG, "cache_control": {"type": "ephemeral"}},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": [{"type": "text", "text": "and now?"}]},
        ],
    }
    pruner = Pruner(settings, store, FakeLocal())
    result = await pruner.prune(payload, "sess1")

    assert not result.modified


async def test_small_blocks_are_left_alone(settings, store):
    local = FakeLocal()
    pruner = Pruner(settings, store, local)
    result = await pruner.prune(payload_with_tool_result("tiny output"), "sess1")

    assert not result.modified
    assert local.calls == 0


async def test_decision_is_memoised_so_the_prefix_stays_stable(settings, store):
    """The client re-sends the original bytes every turn. If we judged them
    differently each time, the prefix would change and the cache would miss."""
    local = FakeLocal()
    pruner = Pruner(settings, store, local)

    first = await pruner.prune(payload_with_tool_result(), "sess1")
    second = await pruner.prune(payload_with_tool_result(), "sess1")

    assert local.calls == 1, "second pass should reuse the stored decision"
    assert first.payload["messages"][-1] == second.payload["messages"][-1]


async def test_elided_text_is_recoverable(settings, store):
    pruner = Pruner(settings, store, FakeLocal())
    result = await pruner.prune(payload_with_tool_result(), "sess1")

    chunk = store.get_chunk(result.handles[0])
    assert chunk is not None
    assert chunk.content in BIG


async def test_elision_marker_names_a_handle(settings, store):
    pruner = Pruner(settings, store, FakeLocal())
    result = await pruner.prune(payload_with_tool_result(), "sess1")

    text = result.payload["messages"][-1]["content"][0]["content"][0]["text"]
    assert "smart-context" in text
    assert result.handles[0] in text


async def test_kept_text_is_verbatim_not_regenerated(settings, store):
    """The local model returns indices, never prose -- so nothing it says can
    end up in the request as if it were real file content."""
    pruner = Pruner(settings, store, FakeLocal(decision=[0, 1]))
    result = await pruner.prune(payload_with_tool_result(), "sess1")

    text = result.payload["messages"][-1]["content"][0]["content"][0]["text"]
    kept = text.split("\n\n[smart-context:")[0]
    assert kept in BIG


async def test_assistant_tail_does_not_block_filtering_elsewhere(settings, store):
    """If the latest message is assistant-only, we still scan backward to the
    newest user turn and filter there when needed."""
    payload = payload_with_tool_result()
    payload["messages"].append({"role": "assistant", "content": [{"type": "text", "text": "hi"}]})

    pruner = Pruner(settings, store, FakeLocal())
    result = await pruner.prune(payload, "sess1")
    assert result.modified


async def test_oversized_text_block_is_shrunk(settings, store):
    payload = {
        "model": "claude-opus-5",
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": BIG}]},
        ],
    }
    pruner = Pruner(settings, store, FakeLocal())
    result = await pruner.prune(payload, "sess1")

    assert result.modified
    text = result.payload["messages"][-1]["content"][0]["text"]
    assert "smart-context" in text
    assert result.handles[0] in text


async def test_thinking_blocks_are_never_touched(settings, store):
    """A thinking block's signature covers its exact bytes; rewriting it is a
    hard 400 upstream, so it must never be considered filterable."""
    payload = {
        "model": "claude-opus-5",
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": BIG, "signature": "sig123"}
            ]},
            {"role": "user", "content": [{"type": "text", "text": "and now?"}]},
        ],
    }
    pruner = Pruner(settings, store, FakeLocal())
    result = await pruner.prune(payload, "sess1")

    assert not result.modified
    assert result.payload == payload


def test_split_chunks_preserves_all_content():
    chunks = split_chunks(BIG, 500)
    assert len(chunks) > 1
    assert "".join(chunks) == BIG


def test_split_chunks_handles_one_oversized_line():
    line = "x" * 5000
    assert "".join(split_chunks(line, 500)) == line


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("[0, 2]", [0, 2]),
        ("Sure! [1]", [1]),
        ("```json\n[0,1,2]\n```", [0, 1, 2]),
        ("{\"keep\": [0, 2]}", [0, 2]),
        ("```json\n{\"indices\": [1, 3]}\n```", [1, 3]),
        ("0, 2", [0, 2]),
        ("2", [2]),
        ("[0, 0, 1]", [0, 1]),          # de-duplicated
        ("[99]", []),                    # out of range dropped
        ("[true, 1]", [1]),              # booleans are not indices
        ("no array here", None),
        ("", None),
    ],
)
def test_parse_indices(raw, expected):
    assert _parse_indices(raw, upper=5) == expected
