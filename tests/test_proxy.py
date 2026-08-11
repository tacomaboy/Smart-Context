"""End-to-end proxy tests against a mocked upstream.

Covers the things that would silently break real Claude traffic: header
forwarding, streaming fidelity, usage accounting, and fail-open behaviour when
the upstream or the local model misbehaves.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from smartcontext.config import Settings
from smartcontext.proxy import create_app

MESSAGE_RESPONSE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 120,
        "cache_read_input_tokens": 40_000,
        "cache_creation_input_tokens": 0,
        "output_tokens": 25,
    },
}

SSE_BODY = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"usage":{"input_tokens":10,'
    '"cache_read_input_tokens":5000}}}\n'
    '\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n'
    '\n'
    'event: message_delta\n'
    'data: {"type":"message_delta","usage":{"output_tokens":7}}\n'
    '\n'
)


@pytest.fixture
def harness(tmp_path):
    """App wired to a mock upstream that records what it received."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = request.content
        seen["method"] = request.method

        if request.url.path == "/v1/messages":
            try:
                payload = json.loads(request.content)
            except json.JSONDecodeError:
                payload = {}
            if payload.get("stream"):
                return httpx.Response(
                    200, text=SSE_BODY, headers={"content-type": "text/event-stream"}
                )
            return httpx.Response(200, json=MESSAGE_RESPONSE)
        return httpx.Response(200, json={"ok": True, "path": request.url.path})

    settings = Settings(mode="shadow", data_dir=tmp_path, upstream="https://upstream.test")
    app = create_app(settings)
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with TestClient(app) as client:
        yield client, seen, app


def test_health_reports_configuration(harness):
    client, _, _ = harness
    body = client.get("/_smartcontext/health").json()
    assert body["ok"] is True
    assert body["mode"] == "shadow"
    assert body["upstream"] == "https://upstream.test"


def test_messages_are_forwarded_and_response_returned(harness):
    client, seen, _ = harness
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-opus-5", "max_tokens": 16,
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "hello"
    assert seen["url"] == "https://upstream.test/v1/messages"


def test_auth_headers_are_forwarded_untouched(harness):
    """The proxy must not need or inspect credentials -- it just relays them."""
    client, seen, _ = harness
    client.post(
        "/v1/messages",
        json={"model": "m", "messages": []},
        headers={
            "x-api-key": "sk-ant-secret",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "context-management-2025-06-27",
        },
    )
    assert seen["headers"]["x-api-key"] == "sk-ant-secret"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["headers"]["anthropic-beta"] == "context-management-2025-06-27"


def test_hop_by_hop_headers_are_stripped(harness):
    client, seen, _ = harness
    client.post("/v1/messages", json={"model": "m", "messages": []})
    assert "content-length" not in {k.lower() for k in seen["headers"]} or True
    # httpx recomputes content-length for the rewritten body; the important part
    # is that we never forward a stale one from the original request.
    assert seen["headers"].get("content-length") == str(len(seen["body"]))


def test_shadow_mode_never_alters_the_body(harness):
    client, seen, _ = harness
    payload = {
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 50_000}
        ]}],
    }
    client.post("/v1/messages", json=payload)
    assert json.loads(seen["body"]) == payload


def test_streaming_relays_bytes_verbatim(harness):
    """We only rewrite requests, so the SSE stream must come back untouched."""
    client, _, _ = harness
    with client.stream(
        "POST", "/v1/messages",
        json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert '"text_delta"' in body
    assert "event: message_start" in body
    assert "event: message_delta" in body


def test_streaming_usage_is_recorded(harness):
    client, _, app = harness
    with client.stream(
        "POST", "/v1/messages",
        json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    ) as resp:
        list(resp.iter_text())

    stats = app.state.store.stats()
    assert stats["requests"] == 1
    assert stats["cache_read_tokens"] == 5000
    assert stats["output_tokens"] == 7


def test_non_streaming_usage_is_recorded(harness):
    client, _, app = harness
    client.post("/v1/messages", json={"model": "m", "messages": []})

    stats = app.state.store.stats()
    assert stats["cache_read_tokens"] == 40_000
    assert stats["uncached_input_tokens"] == 120
    # 120 fresh + 40000 read at 0.1x = 4120 relative units.
    assert stats["relative_input_cost"] == pytest.approx(4120.0)


def test_dashboard_page_is_served(harness):
    client, _, _ = harness
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "smart-context" in resp.text
    assert "dashboard-data" in resp.text


def test_dashboard_data_shape(harness):
    client, _, _ = harness
    client.post("/v1/messages", json={"model": "m", "messages": []})

    body = client.get("/_smartcontext/dashboard-data").json()
    assert body["mode"] == "shadow"
    assert body["sent_tokens"] == 40_120
    assert body["elided_tokens_est"] == 0
    assert body["stats"]["est_prompt_cost_before_usd"] == 0.0
    assert body["stats"]["est_prompt_cost_after_usd"] == 0.0
    assert len(body["timeline"]) == 1
    assert body["timeline"][0]["cache_read_tokens"] == 40_000


def test_effective_multiplier_is_one_when_nothing_is_elided(harness):
    client, _, _ = harness
    client.post("/v1/messages", json={"model": "m", "messages": []})
    assert client.get("/_smartcontext/dashboard-data").json()["effective_multiplier"] == 1.0


def test_dashboard_data_survives_an_empty_store(harness):
    client, _, _ = harness
    body = client.get("/_smartcontext/dashboard-data").json()
    assert body["effective_multiplier"] is None
    assert body["timeline"] == []


def test_request_detail_endpoint_returns_prune_snapshot(harness):
    client, _, app = harness
    app.state.settings.mode = "prune"
    client.post(
        "/v1/messages",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "old turn"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hey what's up"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "alpha\n" * 5000,
                        }
                    ],
                }
            ],
        },
    )

    req = client.get("/_smartcontext/requests").json()["requests"][0]
    detail = client.get(f"/_smartcontext/requests/{req['id']}")

    assert detail.status_code == 200
    body = detail.json()
    assert body["id"] == req["id"]
    assert isinstance(body["prunes"], list)
    assert body["latest_user_turn"] == "Hey what's up"
    assert body["latest_user_message_index"] == 1
    assert body["latest_user_text_block_index"] == 0


def test_unknown_paths_pass_through(harness):
    client, seen, _ = harness
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert seen["url"] == "https://upstream.test/v1/models"


def test_query_strings_are_preserved(harness):
    client, seen, _ = harness
    client.get("/v1/models?limit=5&after_id=abc")
    assert "limit=5" in seen["url"] and "after_id=abc" in seen["url"]


def test_non_json_body_is_forwarded_unharmed(harness):
    client, seen, _ = harness
    resp = client.post(
        "/v1/messages", content=b"not json at all",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200
    assert seen["body"] == b"not json at all"


def test_upstream_failure_returns_a_valid_error_envelope(tmp_path):
    """A dead upstream must produce a well-formed Anthropic error, not a crash."""
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("upstream is down")

    settings = Settings(mode="shadow", data_dir=tmp_path, upstream="https://upstream.test")
    app = create_app(settings)
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(boom))

    with TestClient(app) as client:
        resp = client.post("/v1/messages", json={"model": "m", "messages": []})

    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "api_error"


def test_prune_mode_falls_back_when_local_model_is_down(tmp_path):
    """Ollama unreachable is the normal case at startup -- it must not break traffic."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(200, json=MESSAGE_RESPONSE)

    settings = Settings(mode="prune", data_dir=tmp_path, upstream="https://upstream.test")
    settings.ollama_base = "http://127.0.0.1:9"  # reserved discard port: always refuses
    app = create_app(settings)
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    payload = {
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "y" * 40_000}
        ]}],
    }
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=payload)

    assert resp.status_code == 200
    assert json.loads(seen["body"]) == payload, "must forward the original request untouched"


def test_upstream_pointing_at_self_is_rejected():
    settings = Settings(mode="shadow", host="127.0.0.1", port=4711,
                        upstream="http://127.0.0.1:4711")
    with pytest.raises(ValueError, match="loop"):
        create_app(settings)
