"""Offline threshold sweep -- replay captured payloads through the pruner
under a range of settings, without touching the network or the live store.

This is a cost *proxy*, not a real dollar figure: Anthropic's prompt cache
can't be replayed offline (it lives on their infra, keyed by exact prior
requests), so a sweep can't reproduce real cache-hit/cache-write billing.
What it can measure honestly is the one thing that transfers directly to
cost regardless of caching: resulting prompt token count. Once you've picked
a setting here, confirm it with one live run and `smart-context stats`.

Each sweep point gets its own throwaway Store. Prune decisions are memoised
by content hash alone (see pruner.py), so sharing a store across settings
that differ in ``keep_budget_chars`` would silently reuse a stale decision
computed under a different budget -- isolation avoids that.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .local_model import LocalModel
from .pruner import Pruner, estimate_payload_tokens
from .store import Store


@dataclass
class SweepPoint:
    min_block_chars: int
    keep_budget_chars: int
    payload_count: int
    tokens_before: int
    tokens_after: int
    blocks_filtered: int
    local_model_used: int
    fallback_count: int
    notes: dict[str, int] = field(default_factory=dict)

    @property
    def reduction_pct(self) -> float:
        if not self.tokens_before:
            return 0.0
        return round(100 * (1 - self.tokens_after / self.tokens_before), 1)


def load_captures(captures_dir: Path) -> list[dict[str, Any]]:
    payloads = []
    for f in sorted(captures_dir.glob("*.json")):
        try:
            payloads.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
    return payloads


def load_capture_sessions_keyed(captures_dir: Path) -> list[tuple[str, list[dict[str, Any]]]]:
    """``load_capture_sessions``, but keeping each session's key.

    The key is what makes a replay reproducible -- positional indexes shift as
    new captures land, and ties in turn count make them ambiguous.
    """
    sessions: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for f in sorted(captures_dir.glob("*.json")):
        ts_text, _, session_key = f.stem.partition("_")
        if not session_key:
            continue
        try:
            ts = int(ts_text)
        except ValueError:
            continue
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        sessions.setdefault(session_key, []).append((ts, payload))

    grouped = [
        (key, [payload for _ts, payload in sorted(turns, key=lambda item: item[0])])
        for key, turns in sessions.items()
        if len(turns) > 1
    ]
    # Longest first, then by key so ties are stable across runs.
    grouped.sort(key=lambda item: (-len(item[1]), item[0]))
    return grouped


def load_capture_sessions(captures_dir: Path) -> list[list[dict[str, Any]]]:
    """Captures grouped back into the conversations they came from.

    Capture files are named ``{time_ns}_{session_key}.json``, so the lineage of
    a real conversation -- one prefix growing turn by turn -- is already on
    disk. Replaying a group in order is the only way to observe prompt-cache
    behaviour; a single one-shot payload can never show it.

    Returns one list of payloads per session, each ordered oldest-first, longest
    session first. Single-capture sessions are dropped -- one turn cannot
    demonstrate a warm prefix.
    """
    return [payloads for _key, payloads in load_capture_sessions_keyed(captures_dir)]


async def run_sweep(
    payloads: list[dict[str, Any]],
    base_settings: Settings,
    min_chars_values: list[int],
    keep_budget_values: list[int] | None = None,
) -> list[SweepPoint]:
    keep_budget_values = keep_budget_values or [base_settings.keep_budget_chars]
    # Shared across configs: it holds no per-setting state, just an HTTP
    # client to Ollama, so reusing it doesn't leak between sweep points.
    local = LocalModel(
        model=base_settings.local_model,
        base=base_settings.ollama_base,
        timeout_s=base_settings.local_timeout_s,
    )

    points: list[SweepPoint] = []
    for min_chars in min_chars_values:
        for keep_budget in keep_budget_values:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                store = Store(tmp_path / "sweep.db")
                # Derive from the caller's settings rather than rebuilding one
                # field by field: a hand-copied list silently drops any setting
                # added later (scan_scope did exactly that), and the sweep then
                # measures a configuration nobody asked for.
                settings = replace(
                    base_settings,
                    mode="prune",
                    data_dir=tmp_path,
                    min_block_chars=min_chars,
                    keep_budget_chars=keep_budget,
                )
                pruner = Pruner(settings, store, local)

                before_total = after_total = filtered_total = 0
                used_local_count = fallback_count = 0
                notes: dict[str, int] = {}
                for i, payload in enumerate(payloads):
                    before = estimate_payload_tokens(payload)
                    result = await pruner.prune(payload, session_key=f"sweep-{min_chars}-{keep_budget}-{i}")
                    before_total += before
                    after_total += result.est_after
                    filtered_total += result.blocks_filtered
                    used_local_count += int(result.local_model_used)
                    if result.blocks_filtered == 0 and result.note == "local model unavailable or kept everything":
                        fallback_count += 1
                    notes[result.note] = notes.get(result.note, 0) + 1
                store.close()

            points.append(SweepPoint(
                min_block_chars=min_chars,
                keep_budget_chars=keep_budget,
                payload_count=len(payloads),
                tokens_before=before_total,
                tokens_after=after_total,
                blocks_filtered=filtered_total,
                local_model_used=used_local_count,
                fallback_count=fallback_count,
                notes=notes,
            ))
    return points


def format_table(points: list[SweepPoint]) -> str:
    header = f"{'min_chars':>10}  {'keep_budget':>11}  {'before':>10}  {'after':>10}  {'reduction':>9}  {'blocks':>7}  {'fallbacks':>9}"
    lines = [header, "-" * len(header)]
    for p in points:
        lines.append(
            f"{p.min_block_chars:>10}  {p.keep_budget_chars:>11}  {p.tokens_before:>10}  "
            f"{p.tokens_after:>10}  {p.reduction_pct:>8}%  {p.blocks_filtered:>7}  {p.fallback_count:>9}"
        )
    return "\n".join(lines)
