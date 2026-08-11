"""Local SQLite store: everything pruned stays retrievable.

This is what makes pruning *recoverable* rather than lossy. Content removed from
a request is written here first and replaced in-band with a short handle, so
Claude can ask for it back through the recall MCP tool.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .tokens import price_basis_for_model, price_per_token

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    handle        TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    session_key   TEXT,
    tool_use_id   TEXT,
    origin        TEXT,
    content       TEXT NOT NULL,
    token_est     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunks_tool_use ON chunks(tool_use_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content, handle UNINDEXED, tokenize='porter'
);

-- Pruning decisions are memoised by content hash alone (NOT by the evolving
-- task text). The client keeps its own full copy of every tool result and
-- re-sends it verbatim on every turn; if we judged the same bytes differently
-- from one turn to the next, the prompt prefix would change and every request
-- would miss the cache. Deciding once, and identically forever after, is what
-- makes filtering cache-safe.
CREATE TABLE IF NOT EXISTS prune_decisions (
    content_hash TEXT PRIMARY KEY,
    created_at   REAL NOT NULL,
    kept_json    TEXT NOT NULL,
    handles_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS requests (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  REAL NOT NULL,
    session_key         TEXT,
    model               TEXT,
    mode                TEXT,
    streaming           INTEGER,
    est_tokens_before   INTEGER,
    est_tokens_after    INTEGER,
    blocks_filtered     INTEGER,
    local_model_used    INTEGER,
    status              INTEGER,
    input_tokens        INTEGER,
    cache_read_tokens   INTEGER,
    cache_write_tokens  INTEGER,
    output_tokens       INTEGER,
    relative_input_cost REAL,
    note                TEXT,
    latest_user_turn    TEXT,
    latest_user_message_index INTEGER,
    latest_user_text_block_index INTEGER,
    tool_trim_tokens_saved_est INTEGER,
    latency_local_ms    REAL
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts DESC);

CREATE TABLE IF NOT EXISTS request_prunes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id        INTEGER NOT NULL,
    message_index     INTEGER NOT NULL,
    block_index       INTEGER NOT NULL,
    tool_use_id       TEXT,
    before_text       TEXT NOT NULL,
    after_text        TEXT NOT NULL,
    handles_json      TEXT NOT NULL,
    before_tokens_est INTEGER NOT NULL,
    after_tokens_est  INTEGER NOT NULL,
    FOREIGN KEY (request_id) REFERENCES requests(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_request_prunes_request ON request_prunes(request_id);
"""


@dataclass
class Chunk:
    handle: str
    created_at: float
    session_key: str | None
    tool_use_id: str | None
    origin: str | None
    content: str
    token_est: int

    def preview(self, limit: int = 240) -> str:
        flat = " ".join(self.content.split())
        return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def make_handle(content: str, salt: str = "") -> str:
    digest = hashlib.sha256((salt + content).encode("utf-8", "replace")).hexdigest()
    return f"sc_{digest[:12]}"


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._conn:
            self._conn.executescript(SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after a store was first created on disk."""
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(requests)")}
        if "latency_local_ms" not in existing:
            self._conn.execute("ALTER TABLE requests ADD COLUMN latency_local_ms REAL")
        if "latest_user_turn" not in existing:
            self._conn.execute("ALTER TABLE requests ADD COLUMN latest_user_turn TEXT")
        if "latest_user_message_index" not in existing:
            self._conn.execute("ALTER TABLE requests ADD COLUMN latest_user_message_index INTEGER")
        if "latest_user_text_block_index" not in existing:
            self._conn.execute("ALTER TABLE requests ADD COLUMN latest_user_text_block_index INTEGER")
        if "tool_trim_tokens_saved_est" not in existing:
            self._conn.execute("ALTER TABLE requests ADD COLUMN tool_trim_tokens_saved_est INTEGER")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()

    # ---------------------------------------------------------------- chunks

    def put_chunk(
        self,
        content: str,
        *,
        session_key: str | None = None,
        tool_use_id: str | None = None,
        origin: str | None = None,
        token_est: int = 0,
    ) -> str:
        handle = make_handle(content, salt=tool_use_id or "")
        with self._conn:
            existing = self._conn.execute(
                "SELECT 1 FROM chunks WHERE handle = ?", (handle,)
            ).fetchone()
            if existing:
                return handle
            self._conn.execute(
                "INSERT INTO chunks (handle, created_at, session_key, tool_use_id, origin, content, token_est) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (handle, time.time(), session_key, tool_use_id, origin, content, token_est),
            )
            self._conn.execute(
                "INSERT INTO chunks_fts (content, handle) VALUES (?, ?)", (content, handle)
            )
        return handle

    def get_chunk(self, handle: str) -> Chunk | None:
        row = self._conn.execute("SELECT * FROM chunks WHERE handle = ?", (handle,)).fetchone()
        return _row_to_chunk(row) if row else None

    def search(self, query: str, limit: int = 5, session_key: str | None = None) -> list[Chunk]:
        """Full-text search over everything pruned. Falls back to LIKE if FTS
        rejects the query (unbalanced quotes and bare operators are common in
        model-authored queries and would otherwise raise)."""
        try:
            rows = self._conn.execute(
                "SELECT c.* FROM chunks_fts f JOIN chunks c ON c.handle = f.handle "
                "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            like = f"%{query.strip()}%"
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (like, limit),
            ).fetchall()

        chunks = [_row_to_chunk(r) for r in rows]
        if session_key:
            preferred = [c for c in chunks if c.session_key == session_key]
            if preferred:
                return preferred
        return chunks

    def recent(self, limit: int = 20, session_key: str | None = None) -> list[Chunk]:
        if session_key:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE session_key = ? ORDER BY created_at DESC LIMIT ?",
                (session_key, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM chunks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    # ------------------------------------------------------------- decisions

    def get_decision(self, content_hash: str) -> tuple[list[int], list[str]] | None:
        row = self._conn.execute(
            "SELECT kept_json, handles_json FROM prune_decisions WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["kept_json"]), json.loads(row["handles_json"])
        except json.JSONDecodeError:
            return None

    def put_decision(self, content_hash: str, kept: list[int], handles: list[str]) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO prune_decisions (content_hash, created_at, kept_json, handles_json) "
                "VALUES (?, ?, ?, ?)",
                (content_hash, time.time(), json.dumps(kept), json.dumps(handles)),
            )

    # -------------------------------------------------------------- requests

    def log_request(self, record: dict[str, Any]) -> int:
        columns = (
            "ts", "session_key", "model", "mode", "streaming",
            "est_tokens_before", "est_tokens_after", "blocks_filtered",
            "local_model_used", "status", "input_tokens", "cache_read_tokens",
            "cache_write_tokens", "output_tokens", "relative_input_cost", "note",
            "latest_user_turn",
            "latest_user_message_index",
            "latest_user_text_block_index",
            "tool_trim_tokens_saved_est",
            "latency_local_ms",
        )
        values = [record.get(c) for c in columns]
        placeholders = ", ".join("?" for _ in columns)
        with self._conn:
            cursor = self._conn.execute(
                f"INSERT INTO requests ({', '.join(columns)}) VALUES ({placeholders})", values
            )
            request_id = int(cursor.lastrowid)
            for detail in record.get("prune_details") or []:
                self._conn.execute(
                    "INSERT INTO request_prunes (request_id, message_index, block_index, tool_use_id, before_text, after_text, handles_json, before_tokens_est, after_tokens_est) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        request_id,
                        int(detail.get("message_index") or 0),
                        int(detail.get("block_index") or 0),
                        detail.get("tool_use_id"),
                        detail.get("before_text") or "",
                        detail.get("after_text") or "",
                        json.dumps(detail.get("handles") or []),
                        int(detail.get("before_tokens_est") or 0),
                        int(detail.get("after_tokens_est") or 0),
                    ),
                )
        return request_id

    def reset(self, *, wipe_chunks: bool = False) -> None:
        """Clear logged request history (and optionally the recall store too).

        Chunks are left in place by default -- they're the actual recoverable
        content behind every ``context_recall`` handle, not just a stat.
        """
        with self._conn:
            self._conn.execute("DELETE FROM requests")
            if wipe_chunks:
                self._conn.execute("DELETE FROM chunks")
                self._conn.execute("DELETE FROM chunks_fts")
                self._conn.execute("DELETE FROM prune_decisions")
                self._conn.execute("DELETE FROM request_prunes")

    def stats(self, limit: int = 200) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT * FROM requests ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        if not rows:
            return {"requests": 0}

        def total(col: str) -> int:
            return sum(int(r[col] or 0) for r in rows)

        prompt_total = total("input_tokens") + total("cache_read_tokens") + total("cache_write_tokens")
        cache_read = total("cache_read_tokens")

        # Per-request reduction, cost, and trim-time figures. Reduction% and cost
        # savings are pre-cache token-count estimates (est_tokens_before/after),
        # not billed dollars -- see tokens.price_per_token for the caveat.
        tokens_saved = 0
        cost_savings = 0.0
        cost_before = 0.0
        cost_after = 0.0
        reduction_pcts: list[float] = []
        trim_times: list[float] = []
        for r in rows:
            before = int(r["est_tokens_before"] or 0)
            after = int(r["est_tokens_after"] or 0)
            tool_saved = int(r["tool_trim_tokens_saved_est"] or 0)
            if before:
                saved = max(0, before - after) + tool_saved
                tokens_saved += saved
                unit_price = price_per_token(r["model"])
                cost_before += (before + tool_saved) * unit_price
                cost_after += after * unit_price
                cost_savings += saved * unit_price
                if r["mode"] == "prune":
                    reduction_pcts.append(1 - after / before)
            if r["latency_local_ms"] is not None:
                trim_times.append(float(r["latency_local_ms"]))

        return {
            "requests": len(rows),
            "modes": sorted({r["mode"] for r in rows if r["mode"]}),
            "total_prompt_tokens": prompt_total,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": total("cache_write_tokens"),
            "uncached_input_tokens": total("input_tokens"),
            "output_tokens": total("output_tokens"),
            "cache_hit_ratio": round(cache_read / prompt_total, 4) if prompt_total else 0.0,
            "relative_input_cost": round(sum(float(r["relative_input_cost"] or 0) for r in rows), 1),
            "blocks_filtered": total("blocks_filtered"),
            "chunks_stored": self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "tokens_saved": tokens_saved,
            "avg_reduction_pct": round(sum(reduction_pcts) / len(reduction_pcts), 4) if reduction_pcts else 0.0,
            "est_prompt_cost_before_usd": round(cost_before, 4),
            "est_prompt_cost_after_usd": round(cost_after, 4),
            "est_cost_savings_usd": round(cost_savings, 4),
            "est_cost_savings_pct": round(cost_savings / cost_before, 4) if cost_before else 0.0,
            "avg_trim_time_ms": round(sum(trim_times) / len(trim_times), 1) if trim_times else None,
        }

    def elided_tokens(self) -> int:
        """Estimated tokens held locally instead of being sent.

        Counts only the dropped chunks. The 'tool_result_full' rows are whole
        copies kept so recall can return an entire block, and including them
        would double-count.
        """
        row = self._conn.execute(
            "SELECT COALESCE(SUM(token_est), 0) FROM chunks WHERE origin = 'tool_result'"
        ).fetchone()
        return int(row[0] or 0)

    def timeline(self, limit: int = 40) -> list[dict[str, Any]]:
        """Recent requests oldest-first, shaped for charting."""
        rows = self._conn.execute(
            "SELECT id, ts, model, mode, status, est_tokens_before, est_tokens_after, "
            "blocks_filtered, input_tokens, cache_read_tokens, cache_write_tokens, "
            "output_tokens, relative_input_cost, note, tool_trim_tokens_saved_est "
            "FROM requests ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        timeline = []
        for row in reversed(rows):
            item = {k: row[k] for k in row.keys()}
            family, unit_price = price_basis_for_model(item.get("model"))
            before = int(item.get("est_tokens_before") or 0)
            after = int(item.get("est_tokens_after") or 0)
            item["est_price_family"] = family
            item["est_input_price_per_mtoken_usd"] = round(unit_price * 1_000_000, 2)
            item["est_prompt_cost_before_usd"] = round(before * unit_price, 4)
            item["est_prompt_cost_after_usd"] = round(after * unit_price, 4)
            item["est_cost_savings_usd"] = round(max(0, before - after) * unit_price, 4)
            item["est_cost_savings_pct"] = round(max(0, before - after) / before, 4) if before else 0.0
            timeline.append(item)
        return timeline

    def iter_requests(self, limit: int = 50) -> Iterator[dict[str, Any]]:
        for row in self._conn.execute(
            "SELECT * FROM requests ORDER BY ts DESC LIMIT ?", (limit,)
        ):
            yield {k: row[k] for k in row.keys()}

    def request_detail(self, request_id: int) -> dict[str, Any] | None:
        request_row = self._conn.execute(
            "SELECT * FROM requests WHERE id = ?",
            (request_id,),
        ).fetchone()
        if request_row is None:
            return None

        try:
            prune_rows = self._conn.execute(
                "SELECT * FROM request_prunes WHERE request_id = ? ORDER BY message_index, block_index, id",
                (request_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            prune_rows = []
        prunes = []
        for row in prune_rows:
            prunes.append(
                {
                    "id": row["id"],
                    "message_index": row["message_index"],
                    "block_index": row["block_index"],
                    "tool_use_id": row["tool_use_id"],
                    "before_text": row["before_text"],
                    "after_text": row["after_text"],
                    "handles": json.loads(row["handles_json"]),
                    "before_tokens_est": row["before_tokens_est"],
                    "after_tokens_est": row["after_tokens_est"],
                }
            )

        request = {k: request_row[k] for k in request_row.keys()}
        request["prunes"] = prunes
        return request


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        handle=row["handle"],
        created_at=row["created_at"],
        session_key=row["session_key"],
        tool_use_id=row["tool_use_id"],
        origin=row["origin"],
        content=row["content"],
        token_est=row["token_est"],
    )


def session_key_for(payload: dict[str, Any]) -> str:
    """Stable-ish identifier for a conversation.

    Hashes the *first* user message plus the system prompt: both are fixed for
    the life of a conversation, so every turn maps to the same key.
    """
    system = payload.get("system")
    system_text = json.dumps(system, sort_keys=True) if system is not None else ""
    first_user = ""
    for message in payload.get("messages") or []:
        if message.get("role") == "user":
            first_user = json.dumps(message.get("content"), sort_keys=True)
            break
    digest = hashlib.sha256((system_text + " " + first_user).encode("utf-8", "replace"))
    return digest.hexdigest()[:16]
