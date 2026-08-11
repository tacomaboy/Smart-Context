"""Anthropic-API-compatible reverse proxy.

Point ``ANTHROPIC_BASE_URL`` at this and every Claude API client -- Claude Code,
the SDKs, the ``ant`` CLI -- routes through it.

Two properties the whole design hangs on:

* **Requests only.** We modify the outbound request and never the response, so
  streaming relays byte-for-byte. No SSE parsing, no re-framing, no chance of
  corrupting a ``tool_use`` block on the way back.
* **Fail open.** Any error in the local pipeline -- Ollama down, malformed body,
  a bug in the pruner -- falls back to forwarding the original bytes untouched.
  A context optimiser that can take your Claude access down with it is a bad
  trade at any savings.
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .config import Settings
from .local_model import LocalModel
from .pruner import Pruner, estimate_payload_tokens
from .store import Store, session_key_for
from .tokens import estimate_tokens, relative_input_cost, usage_summary

log = logging.getLogger("smartcontext.proxy")

# Hop-by-hop headers, plus ones httpx must recompute for the rewritten body.
_SKIP_REQUEST_HEADERS = {
    "host", "content-length", "connection", "keep-alive", "transfer-encoding",
    "upgrade", "proxy-authorization", "proxy-connection", "te", "trailer",
    "accept-encoding",
}
_SKIP_RESPONSE_HEADERS = {
    "content-length", "content-encoding", "connection", "keep-alive",
    "transfer-encoding", "upgrade", "trailer",
}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await app.state.client.aclose()
        app.state.store.close()

    app = FastAPI(title="smart-context", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.settings = settings
    app.state.store = Store(settings.db_path)
    app.state.local = LocalModel(
        model=settings.local_model,
        base=settings.ollama_base,
        timeout_s=settings.local_timeout_s,
    )
    app.state.pruner = Pruner(settings, app.state.store, app.state.local)
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.upstream_timeout_s, connect=10.0),
        follow_redirects=False,
    )

    # ------------------------------------------------------------ control

    @app.get("/_smartcontext/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "mode": settings.mode,
            "upstream": settings.upstream,
            "local_model": settings.local_model,
            "local_model_available": await app.state.local.available(),
            "db": str(settings.db_path),
        }

    @app.get("/_smartcontext/stats")
    async def stats() -> dict[str, Any]:
        return app.state.store.stats()

    @app.get("/_smartcontext/requests")
    async def requests_log(limit: int = 25) -> dict[str, Any]:
        return {"requests": list(app.state.store.iter_requests(limit))}

    @app.get("/_smartcontext/requests/{request_id}")
    async def request_detail(request_id: int) -> Response:
        detail = app.state.store.request_detail(request_id)
        if detail is None:
            return JSONResponse(status_code=404, content={"ok": False, "error": "not found"})
        return JSONResponse(content=detail)

    @app.post("/_smartcontext/reset")
    async def reset(wipe_chunks: bool = False) -> dict[str, Any]:
        app.state.store.reset(wipe_chunks=wipe_chunks)
        return {"ok": True, "wiped_chunks": wipe_chunks}

    @app.get("/_smartcontext/dashboard-data")
    async def dashboard_data() -> dict[str, Any]:
        store: Store = app.state.store
        stats = store.stats()
        elided = store.elided_tokens()
        sent = stats.get("total_prompt_tokens", 0)
        return {
            "mode": settings.mode,
            "local_model": settings.local_model,
            "local_model_available": await app.state.local.available(),
            "stats": stats,
            # Exact, from response usage objects.
            "sent_tokens": sent,
            # Estimated, from our own chunk sizing. Labelled as such in the UI.
            "elided_tokens_est": elided,
            "effective_multiplier": round((sent + elided) / sent, 2) if sent else None,
            "timeline": store.timeline(40),
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        page = Path(__file__).parent / "static" / "dashboard.html"
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/_smartcontext/recall")
    async def recall(q: str = "", handle: str = "", limit: int = 5) -> dict[str, Any]:
        store: Store = app.state.store
        if handle:
            chunk = store.get_chunk(handle)
            return {"found": bool(chunk), "content": chunk.content if chunk else None}
        results = store.search(q, limit=limit) if q else store.recent(limit=limit)
        return {
            "results": [
                {"handle": c.handle, "preview": c.preview(), "tokens_est": c.token_est}
                for c in results
            ]
        }

    # -------------------------------------------------------------- proxy

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        raw = await request.body()
        started = time.perf_counter()

        payload: dict[str, Any] | None
        try:
            parsed = json.loads(raw)
            payload = parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None

        if payload is None:
            # Not something we understand -- forward verbatim.
            return await _forward(app, request, raw, streaming=False, record=None)

        session_key = session_key_for(payload)
        if settings.capture:
            _capture_payload(settings, session_key, payload)
        streaming = bool(payload.get("stream"))
        latest_user_turn = _latest_user_turn_info(payload)
        record: dict[str, Any] = {
            "ts": time.time(),
            "session_key": session_key,
            "model": payload.get("model"),
            "mode": settings.mode,
            "streaming": int(streaming),
            "est_tokens_before": estimate_payload_tokens(payload),
            "est_tokens_after": None,
            "blocks_filtered": 0,
            "local_model_used": 0,
            "note": "",
            "latest_user_turn": latest_user_turn["text"] if latest_user_turn else None,
            "latest_user_message_index": latest_user_turn["message_index"] if latest_user_turn else None,
            "latest_user_text_block_index": latest_user_turn["text_block_index"] if latest_user_turn else None,
            "tool_trim_tokens_saved_est": 0,
        }

        body = raw
        retry_original_body: bytes | None = None
        if settings.mode == "prune":
            try:
                working = payload
                tools_changed = False
                if settings.trim_tools:
                    latest_text = latest_user_turn["text"] if latest_user_turn else ""
                    tools_tokens_before = _estimate_tools_tokens(payload)
                    working, before_tools, after_tools, trim_method = await _trim_tools_catalog(
                        working,
                        local=app.state.local,
                        max_tools=settings.max_tools,
                        latest_user_text=latest_text,
                    )
                    tools_changed = before_tools > after_tools
                    tools_tokens_after = _estimate_tools_tokens(working)
                    record["tool_trim_tokens_saved_est"] = max(0, tools_tokens_before - tools_tokens_after)
                    if tools_changed:
                        record["note"] = f"tools trimmed {before_tools}->{after_tools} ({trim_method})"

                result = await app.state.pruner.prune(working, session_key)
                record["est_tokens_after"] = result.est_after
                record["blocks_filtered"] = result.blocks_filtered
                record["local_model_used"] = int(result.local_model_used)
                record["prune_details"] = result.details
                if result.note:
                    record["note"] = f"{record.get('note', '')} | {result.note}".strip(" |")
                if result.modified or tools_changed:
                    forward_payload = result.payload if result.modified else working
                    body = json.dumps(forward_payload).encode("utf-8")
                if tools_changed and settings.trim_tools_retry_missing:
                    retry_payload = result.payload if result.modified else working
                    if isinstance(payload.get("tools"), list):
                        retry_payload = dict(retry_payload)
                        retry_payload["tools"] = payload["tools"]
                        retry_original_body = json.dumps(retry_payload).encode("utf-8")
            except Exception as exc:  # noqa: BLE001 - never break the request
                log.exception("pruning failed; forwarding original request")
                record["note"] = f"prune error, passed through: {exc}"
        else:
            record["est_tokens_after"] = record["est_tokens_before"]
            record["note"] = "shadow mode: measured only"

        record["latency_local_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return await _forward(
            app,
            request,
            body,
            streaming=streaming,
            record=record,
            retry_original_body=retry_original_body,
        )

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    )
    async def passthrough(request: Request, path: str) -> Response:
        raw = await request.body()
        return await _forward(app, request, raw, streaming=False, record=None)

    return app


def _latest_user_turn_info(payload: dict[str, Any], limit: int = 1600) -> dict[str, Any] | None:
    """Best-effort text preview and location of the newest user turn."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None

    for message_index in range(len(messages) - 1, -1, -1):
        message = messages[message_index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        text, text_block_index = _content_to_text(content)
        if not text:
            return None
        text = text.strip()
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        return {
            "text": text,
            "message_index": message_index,
            "text_block_index": text_block_index,
        }
    return None


def _content_to_text(content: Any) -> tuple[str, int | None]:
    if isinstance(content, str):
        return content, None
    if isinstance(content, list):
        parts: list[str] = []
        first_text_block_index: int | None = None
        for i, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                if first_text_block_index is None:
                    first_text_block_index = i
                parts.append(block["text"])
        if parts:
            return "\n\n".join(parts), first_text_block_index
    try:
        return json.dumps(content, ensure_ascii=False), None
    except Exception:
        return "", None


def _recent_tool_use_names(payload: dict[str, Any], limit: int = 32) -> set[str]:
    names: set[str] = set()
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return names

    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip().lower())
                if len(names) >= limit:
                    return names
    return names


async def _trim_tools_catalog(
    payload: dict[str, Any],
    *,
    local: LocalModel,
    max_tools: int,
    latest_user_text: str,
) -> tuple[dict[str, Any], int, int, str]:
    """Trim `tools` to a subset likely relevant to this turn.

    Primary path uses the local model; fallback uses a deterministic heuristic.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return payload, 0, 0, "none"
    if len(tools) <= max_tools:
        return payload, len(tools), len(tools), "none"

    # Local-model ranking first: summarize each tool as one chunk and ask for
    # the indices most likely needed for this turn.
    chunks: list[str] = []
    for i, tool in enumerate(tools):
        if isinstance(tool, dict):
            name = str(tool.get("name") or f"tool_{i}")
            desc = str(tool.get("description") or "")[:400]
            schema = json.dumps(tool.get("input_schema") or tool.get("parameters") or {}, ensure_ascii=False)[:1200]
            chunks.append(f"name: {name}\ndescription: {desc}\nschema: {schema}")
        else:
            chunks.append(str(tool)[:1800])

    decision = await local.select_chunks(
        task=(latest_user_text or "Pick tools relevant to the current user request."),
        chunks=chunks,
        keep_at_least=min(max_tools, len(chunks)),
    )
    if decision is not None and decision.keep:
        keep_local = sorted(idx for idx in decision.keep if 0 <= idx < len(tools))[:max_tools]
        if keep_local:
            trimmed = dict(payload)
            trimmed["tools"] = [tools[i] for i in keep_local]
            return trimmed, len(tools), len(keep_local), "local"

    # Fallback: deterministic heuristic if local selection is unavailable.
    recent_used = _recent_tool_use_names(payload)
    user_text = (latest_user_text or "").lower()

    ranked: list[tuple[int, int]] = []
    for idx, tool in enumerate(tools):
        score = 0
        name = ""
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            name = tool["name"].strip()
        low = name.lower()

        if low and low in recent_used:
            score += 100
        if low and user_text:
            relaxed = re.sub(r"[^a-z0-9]+", " ", low).strip()
            if low in user_text or (relaxed and relaxed in user_text):
                score += 10

        ranked.append((score, idx))

    ranked.sort(key=lambda pair: (-pair[0], pair[1]))
    keep_idx = sorted(idx for _, idx in ranked[:max_tools])

    trimmed = dict(payload)
    trimmed["tools"] = [tools[i] for i in keep_idx]
    return trimmed, len(tools), len(keep_idx), "heuristic"


def _outbound_headers(request: Request) -> dict[str, str]:
    return {
        k: v for k, v in request.headers.items()
        if k.lower() not in _SKIP_REQUEST_HEADERS
    }


def _estimate_tools_tokens(payload: dict[str, Any]) -> int:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return 0
    total = 0
    for tool in tools:
        try:
            text = json.dumps(tool, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(tool)
        total += estimate_tokens(text)
    return total


def _inbound_headers(resp: httpx.Response) -> dict[str, str]:
    return {
        k: v for k, v in resp.headers.items()
        if k.lower() not in _SKIP_RESPONSE_HEADERS
    }


def _capture_payload(settings: Settings, session_key: str, payload: dict[str, Any]) -> None:
    """Write the raw, unredacted request to disk for offline sweep fixtures.

    Best-effort only -- a capture failure must never affect the live request,
    same fail-open rule as the pruner itself.
    """
    try:
        settings.captures_dir.mkdir(parents=True, exist_ok=True)
        name = f"{time.time_ns()}_{session_key}.json"
        (settings.captures_dir / name).write_text(
            json.dumps(payload), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001 - capture is diagnostic, never load-bearing
        log.exception("failed to capture request payload")


async def _forward(
    app: FastAPI,
    request: Request,
    body: bytes,
    *,
    streaming: bool,
    record: dict[str, Any] | None,
    retry_original_body: bytes | None = None,
) -> Response:
    settings: Settings = app.state.settings
    client: httpx.AsyncClient = app.state.client
    store: Store = app.state.store

    url = f"{settings.upstream}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    headers = _outbound_headers(request)

    if streaming:
        return await _forward_streaming(client, store, request, url, headers, body, record)

    try:
        resp = await client.request(request.method, url, headers=headers, content=body)
    except httpx.HTTPError as exc:
        log.error("upstream request failed: %s", exc)
        if record is not None:
            record["status"] = 502
            record["note"] = f"{record.get('note', '')} | upstream error: {exc}".strip(" |")
            store.log_request(record)
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "api_error", "message": f"upstream unreachable: {exc}"}},
        )

    # One-shot fallback: if tools were trimmed and upstream rejects with a
    # missing-tool style error, retry once with the original tool catalog.
    if retry_original_body is not None and _is_missing_tool_error(resp):
        try:
            retry_resp = await client.request(request.method, url, headers=headers, content=retry_original_body)
            if record is not None:
                record["note"] = (
                    f"{record.get('note', '')} | auto-retried with full tools after missing-tool error"
                ).strip(" |")
            resp = retry_resp
        except httpx.HTTPError as exc:
            log.warning("retry with full tools failed: %s", exc)
            if record is not None:
                record["note"] = f"{record.get('note', '')} | full-tools retry failed: {exc}".strip(" |")

    if record is not None:
        record["status"] = resp.status_code
        _record_usage_from_json(record, resp.content)
        store.log_request(record)

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=_inbound_headers(resp),
        media_type=resp.headers.get("content-type"),
    )


def _is_missing_tool_error(resp: httpx.Response) -> bool:
    if resp.status_code not in {400, 404, 409, 422}:
        return False
    try:
        payload = resp.json()
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False

    message = ""
    if isinstance(payload.get("error"), dict):
        message = str(payload["error"].get("message") or "")
    if not message:
        message = str(payload.get("message") or "")
    m = message.lower()
    if "tool" not in m:
        return False
    return (
        "not available" in m
        or "not found" in m
        or "unknown tool" in m
        or "tool not" in m
    )


async def _forward_streaming(
    client: httpx.AsyncClient,
    store: Store,
    request: Request,
    url: str,
    headers: dict[str, str],
    body: bytes,
    record: dict[str, Any] | None,
):
    """Relay SSE untouched, sniffing usage off a copy as it passes."""
    seen: dict[str, Any] = {}

    async def stream():
        status = 200
        try:
            async with client.stream(
                request.method, url, headers=headers, content=body
            ) as resp:
                status = resp.status_code
                async for raw_line in resp.aiter_lines():
                    if raw_line.startswith("data:"):
                        _sniff_usage(raw_line[5:].strip(), seen)
                    # aiter_lines strips the newline; SSE framing needs it back.
                    yield (raw_line + "\n").encode("utf-8")
        except httpx.HTTPError as exc:
            log.error("upstream stream failed: %s", exc)
            status = 502
            payload = json.dumps(
                {"type": "error", "error": {"type": "api_error", "message": str(exc)}}
            )
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
        finally:
            if record is not None:
                record["status"] = status
                _apply_usage(record, seen)
                store.log_request(record)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


def _sniff_usage(data: str, seen: dict[str, Any]) -> None:
    """Accumulate usage fields from message_start / message_delta events.

    ``message_start`` carries input and cache counts; ``message_delta`` carries
    the final output count. Merging both gives the full picture.
    """
    if not data or data == "[DONE]":
        return
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return
    if not isinstance(event, dict):
        return

    usage = event.get("usage")
    if usage is None and isinstance(event.get("message"), dict):
        usage = event["message"].get("usage")
    if isinstance(usage, dict):
        for key, value in usage.items():
            if isinstance(value, int) and value:
                seen[key] = value


def _record_usage_from_json(record: dict[str, Any], content: bytes) -> None:
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if isinstance(parsed, dict):
        _apply_usage(record, parsed.get("usage") or {})


def _apply_usage(record: dict[str, Any], usage: dict[str, Any]) -> None:
    summary = usage_summary(usage)
    record["input_tokens"] = summary["input_tokens"]
    record["cache_read_tokens"] = summary["cache_read_input_tokens"]
    record["cache_write_tokens"] = summary["cache_creation_input_tokens"]
    record["output_tokens"] = summary["output_tokens"]
    record["relative_input_cost"] = round(relative_input_cost(summary), 2)
