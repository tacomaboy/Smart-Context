"""Tests for the synthetic session generator.

Its whole value is that the answers are known and the filler is worthless. If
either property breaks, the sessions stop measuring pruning and start measuring
noise -- which is exactly what the real captures were doing.
"""

from __future__ import annotations

import json

from smartcontext.synth import (
    CLAUDE_MD,
    FACTS,
    INTERSTITIAL_TURNS,
    LISTEN_PORT,
    PROBES,
    build_session,
    probe_for_payload,
    score_answer,
    write_session,
)
from smartcontext.tokens import block_text


def test_every_probe_has_its_fact_planted_earlier():
    """A probe only tests pruning if the answer is somewhere behind it."""
    payloads = build_session(filler_chars=1000)
    final = payloads[-1]["messages"]

    # Only probes answered by a fact the user stated. Probes fed by CLAUDE.md
    # or by tool output have their own tests.
    for probe in [p for p in PROBES if p.depends_on in FACTS]:
        fact_positions = [
            i for i, m in enumerate(final)
            if any(probe.depends_on in block_text(b) for b in m.get("content", [])
                   if isinstance(b, dict))
        ]
        question_positions = [
            i for i, m in enumerate(final)
            if any(probe.question in block_text(b) for b in m.get("content", [])
                   if isinstance(b, dict))
        ]
        assert fact_positions, f"fact for {probe.question!r} never stated"
        assert question_positions, f"probe {probe.question!r} never asked"
        assert min(fact_positions) < min(question_positions)


def test_every_payload_is_a_valid_request():
    """Roles must strictly alternate and tools must be declared. Two assistant
    messages in a row is a 400, which silently turned a whole bakeoff into
    errors -- every payload after the first was rejected before it was scored."""
    for i, payload in enumerate(build_session(filler_chars=500)):
        roles = [m["role"] for m in payload["messages"]]
        assert roles[0] == "user", f"payload {i} must start with a user turn"
        assert roles[-1] == "user", f"payload {i} must end asking something"
        for a, b in zip(roles, roles[1:]):
            assert a != b, f"payload {i} has two {a} messages in a row: {roles}"

        # tool_use / tool_result blocks require the tool to be declared.
        uses_tools = any(
            isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
            for m in payload["messages"] for b in m["content"]
        )
        if uses_tools:
            assert payload.get("tools"), f"payload {i} uses tools without declaring them"

        # Every tool_result must match a tool_use, or the request is rejected.
        use_ids = {
            b["id"] for m in payload["messages"] for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_use"
        }
        result_ids = {
            b["tool_use_id"] for m in payload["messages"] for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"
        }
        assert result_ids <= use_ids, f"payload {i} has an orphaned tool_result"


def test_claude_md_is_planted_once_and_is_large_enough_to_be_trimmed():
    """The open question is whether losing CLAUDE.md costs anything. It can only
    be answered if the block is big enough to clear the trimming thresholds and
    is never re-sent fresh."""
    payloads = build_session(filler_chars=500)
    final = payloads[-1]["messages"]

    occurrences = [
        m_index for m_index, m in enumerate(final)
        for b in m["content"]
        if isinstance(b, dict) and "# claudeMd" in block_text(b)
    ]
    assert len(occurrences) == 1, "CLAUDE.md must appear exactly once, as the real client sends it"
    assert occurrences[0] < 3, "it belongs at the start, so history carries it forward"
    # Sized on purpose: above 2,000 so that threshold trims it, below 8,000 so
    # that threshold spares it. The probes then show whether losing it matters.
    assert 2000 < len(CLAUDE_MD) < 8000, f"CLAUDE.md is {len(CLAUDE_MD)} chars"


def test_instruction_probes_depend_on_claude_md_not_on_a_stated_fact():
    """These probes fail only if the CLAUDE.md block was destroyed."""
    instruction_probes = [p for p in PROBES if "CLAUDE.md" in p.depends_on]
    assert instruction_probes, "no probe tests the instruction block"
    for probe in instruction_probes:
        assert probe.expected.lower() in CLAUDE_MD.lower()
        assert all(probe.expected not in fact for fact in FACTS)


def test_history_grows_and_carries_everything_forward():
    """Mirrors a real client re-sending full history, which is what makes pruning
    decisions compound across turns."""
    payloads = build_session(filler_chars=1000)

    lengths = [len(p["messages"]) for p in payloads]
    assert lengths == sorted(lengths)
    assert len(payloads) == len(FACTS) + INTERSTITIAL_TURNS + len(PROBES)
    assert len(payloads[-1]["messages"]) > len(payloads[0]["messages"])


def tool_results_by_tool(payload) -> dict[str, list[str]]:
    """Tool results grouped by the tool that produced them."""
    names = {
        b["id"]: b["name"]
        for m in payload["messages"] for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_use"
    }
    out: dict[str, list[str]] = {}
    for m in payload["messages"]:
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                out.setdefault(names.get(b["tool_use_id"], "?"), []).append(block_text(b))
    return out


def test_build_filler_is_large_and_contains_no_answers():
    """Build logs are the safe-to-discard case: the pruner needs bulk to work
    on, and dropping every byte of it must never cost a correct answer."""
    payloads = build_session(filler_chars=8000)
    fillers = tool_results_by_tool(payloads[-1])["run_build"]

    assert fillers, "no build filler generated"
    assert all(len(f) >= 8000 for f in fillers)
    for probe in PROBES:
        assert all(probe.expected.lower() not in f.lower() for f in fillers)


def test_config_dump_hides_a_needed_fact_deep_inside_it():
    """The hard case: tool output that looks like discardable noise but isn't.
    The fact sits past the halfway mark, so keeping the first slice isn't enough
    -- the pruner has to actually select it."""
    payloads = build_session(filler_chars=8000)
    dumps = tool_results_by_tool(payloads[-1])["read_file"]

    assert len(dumps) == 1
    dump = dumps[0]
    assert LISTEN_PORT in dump
    assert dump.index(LISTEN_PORT) > len(dump) * 0.5, "fact must not sit near the top"

    # And it must exist nowhere else, or the probe proves nothing about pruning.
    others = [t for name, ts in tool_results_by_tool(payloads[-1]).items()
              if name != "read_file" for t in ts]
    assert all(LISTEN_PORT not in t for t in others)
    assert all(LISTEN_PORT not in f for f in FACTS)
    assert LISTEN_PORT not in CLAUDE_MD


def test_probes_end_in_prose_questions_not_tool_calls():
    """Real captures end in tool calls, so replayed answers come back empty and
    the judge scores the emptiness. Synthetic turns must ask plain questions."""
    payloads = build_session(filler_chars=1000)
    for payload in payloads[-len(PROBES):]:
        last = payload["messages"][-1]
        assert last["role"] == "user"
        assert last["content"][0]["type"] == "text"


def test_probe_for_payload_identifies_only_probe_turns():
    """Identified structurally, not by position -- turns get inserted."""
    payloads = build_session(filler_chars=500)
    found = [probe_for_payload(p) for p in payloads]

    assert [f.expected for f in found if f] == [p.expected for p in PROBES]
    assert sum(1 for f in found if f is None) == len(FACTS) + INTERSTITIAL_TURNS
    # Probes come last, so the fact is always behind the question.
    assert all(f is not None for f in found[-len(PROBES):])


def test_score_answer_checks_ground_truth():
    probe = PROBES[0]
    assert score_answer("Your favorite day is Friday.", probe)
    assert score_answer("friday", probe)
    assert not score_answer("Your favorite day is Tuesday.", probe)
    assert not score_answer("", probe)


def test_write_session_produces_loadable_captures(tmp_path):
    written = write_session(tmp_path, session_key="synthtest0000000", filler_chars=1000)

    assert len(written) == len(FACTS) + INTERSTITIAL_TURNS + len(PROBES)
    for path in written:
        assert path.stem.endswith("_synthtest0000000")
        json.loads(path.read_text(encoding="utf-8"))  # parses

    from smartcontext.sweep import load_capture_sessions_keyed

    sessions = load_capture_sessions_keyed(tmp_path)
    assert len(sessions) == 1
    key, payloads = sessions[0]
    assert key == "synthtest0000000"
    assert len(payloads) == len(written)
