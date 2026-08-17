"""Tests for RealUsageClient (the Usage & Cost Admin API wrapper).

No real network calls here -- the client's internal ``httpx.AsyncClient`` is
swapped for a fake that records requests and returns scripted responses shaped
exactly like Anthropic's documented schema (same trick as test_bakeoff.py's
fake Anthropic client). What's worth checking: the no-admin-key short circuit
never touches the network, aggregation math (cache hit ratio, cents->dollars,
per-model summing) is right against a scripted response, pagination merges
pages, both HTTP errors and transport exceptions fail open instead of
raising, and the in-memory cache actually avoids refetching within its TTL.
"""

from __future__ import annotations

import pytest

from smartcontext.real_usage import RealUsageClient


class FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}

    def json(self) -> dict:
        return self._body


class FakeAsyncClient:
    """Queues per-path responses; records every call for assertions."""

    def __init__(self, queued: dict[str, list[FakeResponse]]) -> None:
        self._queued = queued
        self.calls: list[dict] = []

    async def get(self, url: str, headers: dict | None = None, params: dict | None = None) -> FakeResponse:
        self.calls.append({"url": url, "headers": headers, "params": params})
        for path, responses in self._queued.items():
            if path in url:
                if not responses:
                    raise AssertionError(f"no more queued responses for {path}")
                return responses.pop(0)
        raise AssertionError(f"unexpected URL {url}")

    async def aclose(self) -> None:
        pass


def make_client(queued: dict[str, list[FakeResponse]], admin_api_key: str | None = "sk-ant-admin01-test") -> RealUsageClient:
    client = RealUsageClient(admin_api_key)
    client._client = FakeAsyncClient(queued)  # type: ignore[assignment]
    return client


USAGE_BUCKET = {
    "starting_at": "2026-08-10T00:00:00Z",
    "ending_at": "2026-08-11T00:00:00Z",
    "results": [
        {
            "model": "claude-opus-5",
            "uncached_input_tokens": 1000,
            "cache_read_input_tokens": 8000,
            "cache_creation": {"ephemeral_5m_input_tokens": 500, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 200,
        },
        {
            "model": "claude-sonnet-5",
            "uncached_input_tokens": 300,
            "cache_read_input_tokens": 700,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 50,
        },
    ],
}

COST_BUCKET = {
    "starting_at": "2026-08-10T00:00:00Z",
    "ending_at": "2026-08-11T00:00:00Z",
    "results": [
        {"model": "claude-opus-5", "amount": "1000", "description": "Claude Opus 5 - Input"},
        {"model": "claude-sonnet-5", "amount": "250", "description": "Claude Sonnet 5 - Input"},
    ],
}


async def test_no_admin_key_makes_no_network_call():
    client = make_client({}, admin_api_key=None)
    result = await client.fetch(days=7)
    assert result == {"available": False, "reason": "no_admin_key"}


async def test_successful_fetch_aggregates_correctly():
    queued = {
        "usage_report/messages": [FakeResponse(200, {"data": [USAGE_BUCKET], "has_more": False, "next_page": None})],
        "cost_report": [FakeResponse(200, {"data": [COST_BUCKET], "has_more": False, "next_page": None})],
    }
    client = make_client(queued)
    result = await client.fetch(days=7)

    assert result["available"] is True
    assert result["totals"] == {
        "uncached_input_tokens": 1300,
        "cache_read_input_tokens": 8700,
        "cache_creation_tokens": 500,
        "output_tokens": 250,
    }
    # 8700 / (8700 + 500 + 1300) = 0.8288...
    assert result["cache_hit_ratio"] == pytest.approx(8700 / 10500)
    # amount is in cents: 1000 + 250 = 1250 cents = $12.50
    assert result["real_cost_usd"] == pytest.approx(12.50)

    by_model = {row["model"]: row for row in result["by_model"]}
    assert by_model["claude-opus-5"]["cache_read_input_tokens"] == 8000
    assert by_model["claude-sonnet-5"]["uncached_input_tokens"] == 300

    cost_by_model = {row["model"]: row["usd"] for row in result["cost_by_model"]}
    assert cost_by_model["claude-opus-5"] == pytest.approx(10.0)
    assert cost_by_model["claude-sonnet-5"] == pytest.approx(2.5)

    assert len(result["daily"]) == 1
    assert result["daily"][0]["date"] == "2026-08-10"


async def test_pagination_merges_pages():
    bucket_1 = {**USAGE_BUCKET, "starting_at": "2026-08-10T00:00:00Z"}
    bucket_2 = {**USAGE_BUCKET, "starting_at": "2026-08-11T00:00:00Z"}
    queued = {
        "usage_report/messages": [
            FakeResponse(200, {"data": [bucket_1], "has_more": True, "next_page": "page_2"}),
            FakeResponse(200, {"data": [bucket_2], "has_more": False, "next_page": None}),
        ],
        "cost_report": [FakeResponse(200, {"data": [COST_BUCKET], "has_more": False, "next_page": None})],
    }
    client = make_client(queued)
    result = await client.fetch(days=7)

    assert result["available"] is True
    assert len(result["daily"]) == 2
    # Second page's params must carry the cursor from the first page's next_page.
    usage_calls = [c for c in client._client.calls if "usage_report/messages" in c["url"]]
    assert usage_calls[1]["params"]["page"] == "page_2"


async def test_http_error_fails_open():
    queued = {
        "usage_report/messages": [FakeResponse(401, {})],
        "cost_report": [],
    }
    client = make_client(queued)
    result = await client.fetch(days=7)
    assert result["available"] is False
    assert "401" in result["reason"]


async def test_transport_exception_fails_open():
    class ExplodingClient(FakeAsyncClient):
        async def get(self, *args, **kwargs):
            raise ConnectionError("upstream unreachable")

    client = RealUsageClient("sk-ant-admin01-test")
    client._client = ExplodingClient({})  # type: ignore[assignment]
    result = await client.fetch(days=7)
    assert result["available"] is False
    assert "upstream unreachable" in result["reason"]


async def test_cache_reuses_result_within_ttl():
    queued = {
        "usage_report/messages": [FakeResponse(200, {"data": [USAGE_BUCKET], "has_more": False, "next_page": None})],
        "cost_report": [FakeResponse(200, {"data": [COST_BUCKET], "has_more": False, "next_page": None})],
    }
    client = make_client(queued)
    first = await client.fetch(days=7)
    second = await client.fetch(days=7)

    assert first is second  # served from cache, not refetched
    assert len(client._client.calls) == 2  # one usage + one cost call, not four


async def test_force_bypasses_cache():
    queued = {
        "usage_report/messages": [
            FakeResponse(200, {"data": [USAGE_BUCKET], "has_more": False, "next_page": None}),
            FakeResponse(200, {"data": [USAGE_BUCKET], "has_more": False, "next_page": None}),
        ],
        "cost_report": [
            FakeResponse(200, {"data": [COST_BUCKET], "has_more": False, "next_page": None}),
            FakeResponse(200, {"data": [COST_BUCKET], "has_more": False, "next_page": None}),
        ],
    }
    client = make_client(queued)
    await client.fetch(days=7)
    await client.fetch(days=7, force=True)

    assert len(client._client.calls) == 4  # two fetches, two calls each
