"""Full-history filtering -- the only place a request is ever modified.

Three invariants, in priority order. Correctness beats savings every time:

1. **Structure preserved.** A ``tool_result`` block is never deleted, only
   shrunk. Dropping one whose ``tool_use`` still exists is a hard API error, so
   the block, its ``tool_use_id``, and its ``is_error`` flag always survive.
2. **Nothing is lost.** Elided text is written to the local store first and
   replaced by a handle Claude can recall on demand.
3. **Never touch what the API will reject or invalidate structurally.**
   ``thinking``/``redacted_thinking`` blocks carry a cryptographic signature
   over their exact bytes -- rewriting one is a hard 400 from the API, so
   they are excluded by construction (not in ``_FILTERABLE_TYPES``).

Every message in the conversation is scanned, not just the newest turn --
this trades the upstream prompt cache for maximum context reduction. Once any
message is rewritten, every request from that point on re-writes the cached
prefix (a cache **write**, billed at 1.25x) instead of reading it (0.1x).
That is an intentional, accepted cost, not an oversight -- see ``smart-context
stats`` / ``relative_input_cost`` to track it.

Blocks carrying ``cache_control`` are still skipped outright -- those mark
cache breakpoints the client placed, and rewriting one changes bytes the
client is relying on to stay put.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .local_model import LocalModel
from .store import Store
from .tokens import estimate_tokens

log = logging.getLogger("smartcontext.pruner")


@dataclass
class PruneResult:
    payload: dict[str, Any]
    est_before: int
    est_after: int
    blocks_filtered: int = 0
    local_model_used: bool = False
    handles: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    @property
    def modified(self) -> bool:
        return self.blocks_filtered > 0


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:32]


def split_chunks(text: str, chunk_chars: int) -> list[str]:
    """Split on line boundaries, packing up to ``chunk_chars`` per chunk.

    Line-aligned so that kept chunks re-join into readable text rather than
    slicing through the middle of a token or an identifier.
    """
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in lines:
        # A single oversized line becomes its own chunk rather than being split.
        if size + len(line) > chunk_chars and buf:
            chunks.append("".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line)
    if buf:
        chunks.append("".join(buf))
    return chunks or ([text] if text else [])


def estimate_payload_tokens(payload: dict[str, Any]) -> int:
    """Rough size of the whole request. Threshold use only."""
    from .tokens import block_text

    total = 0
    system = payload.get("system")
    if isinstance(system, str):
        total += estimate_tokens(system)
    elif isinstance(system, list):
        total += sum(estimate_tokens(block_text(b)) for b in system)

    for message in payload.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            total += sum(estimate_tokens(block_text(b)) for b in content)
    return total


def _recent_task_text(payload: dict[str, Any], limit: int = 2500) -> str:
    """What the model is currently being asked -- the relevance yardstick.

    Uses the system prompt tail plus the most recent non-tool user text.
    """
    from .tokens import block_text

    parts: list[str] = []
    system = payload.get("system")
    if isinstance(system, str):
        parts.append(system[-800:])
    elif isinstance(system, list):
        parts.append(" ".join(block_text(b) for b in system)[-800:])

    for message in reversed(payload.get("messages") or []):
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block_text(block))
        if sum(len(p) for p in parts) > limit:
            break
    return " ".join(parts)[:limit]


class Pruner:
    def __init__(self, settings: Settings, store: Store, local: LocalModel) -> None:
        self.settings = settings
        self.store = store
        self.local = local

    async def prune(self, payload: dict[str, Any], session_key: str) -> PruneResult:
        est_before = estimate_payload_tokens(payload)
        result = PruneResult(payload=payload, est_before=est_before, est_after=est_before)

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            result.note = "no messages"
            return result

        # (message_index, block_index) for every oversized, filterable block
        # in every message -- not just the newest turn.
        targets: list[tuple[int, int]] = []
        for m_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for b_index, block in enumerate(content):
                if _is_filterable(block, self.settings.min_block_chars):
                    targets.append((m_index, b_index))

        if not targets:
            result.note = "no oversized filterable blocks"
            return result

        task = _recent_task_text(payload)
        new_payload = copy.deepcopy(payload)
        new_messages = new_payload["messages"]

        for m_index, b_index in targets:
            content = new_messages[m_index]["content"]
            block = content[b_index]
            replaced = await self._filter_block(block, task, session_key)
            if replaced is None:
                continue
            content[b_index] = replaced.block
            result.blocks_filtered += 1
            result.local_model_used = result.local_model_used or replaced.used_local
            result.handles.extend(replaced.handles)
            result.details.append(
                {
                    "message_index": m_index,
                    "block_index": b_index,
                    "tool_use_id": replaced.tool_use_id,
                    "before_text": replaced.before_text,
                    "after_text": replaced.after_text,
                    "handles": replaced.handles,
                    "before_tokens_est": replaced.before_tokens_est,
                    "after_tokens_est": replaced.after_tokens_est,
                }
            )

        if result.blocks_filtered == 0:
            result.note = "local model unavailable or kept everything"
            return result

        result.payload = new_payload
        result.est_after = estimate_payload_tokens(new_payload)
        result.note = f"filtered {result.blocks_filtered} block(s)"
        return result

    async def _filter_block(self, block: dict[str, Any], task: str, session_key: str) -> "_Replaced | None":
        from .tokens import block_text

        text = block_text(block)
        if not text:
            return None

        chunks = split_chunks(text, self.settings.chunk_chars)
        if len(chunks) < 2:
            return None

        digest = _content_hash(text)
        cached = self.store.get_decision(digest)
        used_local = False

        if cached is not None:
            keep, handles = cached
        else:
            decision = await self.local.select_chunks(task, chunks)
            if decision is None:
                return None  # fail open: leave the block untouched
            used_local = True
            keep = _trim_to_budget(decision.keep, chunks, self.settings.keep_budget_chars)
            dropped = [i for i in range(len(chunks)) if i not in keep]
            if not dropped:
                # Nothing to gain; still memoise so we do not re-ask next turn.
                self.store.put_decision(digest, list(range(len(chunks))), [])
                return None
            handles = [
                self.store.put_chunk(
                    chunks[i],
                    session_key=session_key,
                    tool_use_id=block.get("tool_use_id"),
                    origin="tool_result",
                    token_est=estimate_tokens(chunks[i]),
                )
                for i in dropped
            ]
            # Store the full block too, so recall can return the whole thing.
            self.store.put_chunk(
                text,
                session_key=session_key,
                tool_use_id=block.get("tool_use_id"),
                origin="tool_result_full",
                token_est=estimate_tokens(text),
            )
            self.store.put_decision(digest, keep, handles)

        if not keep and not handles:
            return None

        kept_text = "".join(chunks[i] for i in sorted(keep))
        dropped_count = len(chunks) - len(keep)
        if dropped_count <= 0:
            return None

        footer = (
            f"\n\n[smart-context: {dropped_count} of {len(chunks)} sections elided to save context. "
            f"Full text is stored locally. Retrieve it with the context_recall tool using "
            f"handle {handles[0] if handles else 'n/a'}"
            + (f" (+{len(handles) - 1} more)" if len(handles) > 1 else "")
            + ".]"
        )

        new_block = dict(block)
        after_text = kept_text + footer

        if block.get("type") == "text":
            # Plain text blocks carry their payload directly under "text",
            # not nested in a "content" list like tool_result does.
            new_block["text"] = after_text
        else:
            new_block["content"] = [{"type": "text", "text": after_text}]
        return _Replaced(
            block=new_block,
            handles=handles,
            used_local=used_local,
            tool_use_id=block.get("tool_use_id"),
            before_text=text,
            after_text=after_text,
            before_tokens_est=estimate_tokens(text),
            after_tokens_est=estimate_tokens(after_text),
        )


@dataclass
class _Replaced:
    block: dict[str, Any]
    handles: list[str]
    used_local: bool
    tool_use_id: str | None
    before_text: str
    after_text: str
    before_tokens_est: int
    after_tokens_est: int


# thinking/redacted_thinking are deliberately excluded: they carry a
# cryptographic signature over their exact bytes, and rewriting one is a hard
# 400 from the API. image/document/tool_use blocks are excluded because they
# either aren't text or would break tool call replay if shrunk.
_FILTERABLE_TYPES = {"tool_result", "text"}


def _is_filterable(block: Any, min_chars: int) -> bool:
    from .tokens import block_text

    if not isinstance(block, dict):
        return False
    if block.get("type") not in _FILTERABLE_TYPES:
        return False
    # Never disturb a cache breakpoint the client placed.
    if block.get("cache_control"):
        return False
    return len(block_text(block)) >= min_chars


def _trim_to_budget(keep: list[int], chunks: list[str], budget_chars: int) -> list[int]:
    """Honour the local model's ranking, but cap total kept size.

    ``keep`` arrives most-relevant-first; we take from the front until the
    budget is spent, then re-sort into document order for readability.
    """
    out: list[int] = []
    size = 0
    for index in keep:
        chunk_len = len(chunks[index])
        if out and size + chunk_len > budget_chars:
            continue
        out.append(index)
        size += chunk_len
    return sorted(out)
