"""Real, billed usage and cost -- straight from Anthropic's Usage & Cost Admin API.

Everything else in this project (``tokens.py``, the dashboard's estimated-cost
cards) is a *model* of what caching should be saving, built from our own
request log. It can be wrong -- and was, during early development, which is
what an Anthropic "your cache hit rate is low" email catches that a local
estimate can miss. This module fetches the actual numbers Anthropic bills
against, so the dashboard can show ground truth next to the estimate instead
of trusting the estimate alone.

Requires a separate **Admin API key** (``sk-ant-admin01-...``), not the
regular API key used elsewhere in this project. Without one, ``fetch()``
returns ``{"available": False, ...}`` -- this is an optional monitoring
feature and must never affect the proxy's own fail-open guarantee.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

_API_VERSION = "2023-06-01"
_MAX_PAGES = 10  # runaway guard; a week at 1d buckets is one page in practice.


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cache_creation_tokens(item: dict[str, Any]) -> int:
    creation = item.get("cache_creation") or {}
    return int(creation.get("ephemeral_5m_input_tokens") or 0) + int(
        creation.get("ephemeral_1h_input_tokens") or 0
    )


class RealUsageClient:
    """Thin async client for the Usage & Cost Admin API, with a short in-memory cache."""

    def __init__(
        self,
        admin_api_key: str | None,
        base: str = "https://api.anthropic.com",
        timeout_s: float = 15.0,
        cache_ttl_s: float = 60.0,
    ) -> None:
        self._admin_api_key = admin_api_key
        self._base = base.rstrip("/")
        self._cache_ttl_s = cache_ttl_s
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=10.0))
        self._cached: dict[str, Any] | None = None
        self._cached_at: float = 0.0
        self._cached_days: int | None = None
        self._last_error: str = ""

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch(self, days: int = 7, force: bool = False) -> dict[str, Any]:
        if not self._admin_api_key:
            return {"available": False, "reason": "no_admin_key"}

        fresh = (time.monotonic() - self._cached_at) < self._cache_ttl_s
        if not force and fresh and self._cached is not None and self._cached_days == days:
            return self._cached

        try:
            result = await self._fetch_live(days)
        except Exception as exc:  # noqa: BLE001 - a monitoring feature must never raise
            result = {"available": False, "reason": str(exc)}

        self._cached = result
        self._cached_at = time.monotonic()
        self._cached_days = days
        return result

    async def _fetch_live(self, days: int) -> dict[str, Any]:
        ending_at = datetime.now(timezone.utc)
        starting_at = ending_at - timedelta(days=days)
        headers = {
            "x-api-key": self._admin_api_key,
            "anthropic-version": _API_VERSION,
        }

        usage_buckets = await self._get_paginated(
            "/v1/organizations/usage_report/messages",
            headers,
            {
                "starting_at": _iso(starting_at),
                "ending_at": _iso(ending_at),
                "bucket_width": "1d",
                "group_by[]": "model",
            },
        )
        if usage_buckets is None:
            return {"available": False, "reason": self._last_error}

        cost_buckets = await self._get_paginated(
            "/v1/organizations/cost_report",
            headers,
            {
                "starting_at": _iso(starting_at),
                "ending_at": _iso(ending_at),
                "bucket_width": "1d",
                "group_by[]": "description",
            },
        )
        if cost_buckets is None:
            return {"available": False, "reason": self._last_error}

        return _summarize(usage_buckets, cost_buckets, starting_at, ending_at, days)

    async def _get_paginated(
        self, path: str, headers: dict[str, str], params: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        """GETs every page of a bucketed report. Returns None (with
        ``self._last_error`` set) on any non-2xx response or transport error."""
        buckets: list[dict[str, Any]] = []
        page: str | None = None
        for _ in range(_MAX_PAGES):
            request_params = dict(params)
            if page:
                request_params["page"] = page
            resp = await self._client.get(f"{self._base}{path}", headers=headers, params=request_params)
            if resp.status_code != 200:
                self._last_error = f"{path} returned HTTP {resp.status_code}"
                return None
            body = resp.json()
            buckets.extend(body.get("data") or [])
            if not body.get("has_more"):
                break
            page = body.get("next_page")
            if not page:
                break
        return buckets


def _summarize(
    usage_buckets: list[dict[str, Any]],
    cost_buckets: list[dict[str, Any]],
    starting_at: datetime,
    ending_at: datetime,
    days: int,
) -> dict[str, Any]:
    daily: list[dict[str, Any]] = []
    by_model: dict[str, dict[str, int]] = {}
    totals = {
        "uncached_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_tokens": 0,
        "output_tokens": 0,
    }

    for bucket in usage_buckets:
        day_totals = {
            "uncached_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_tokens": 0,
            "output_tokens": 0,
        }
        for item in bucket.get("results") or []:
            uncached = int(item.get("uncached_input_tokens") or 0)
            cache_read = int(item.get("cache_read_input_tokens") or 0)
            cache_creation = _cache_creation_tokens(item)
            output = int(item.get("output_tokens") or 0)

            day_totals["uncached_input_tokens"] += uncached
            day_totals["cache_read_input_tokens"] += cache_read
            day_totals["cache_creation_tokens"] += cache_creation
            day_totals["output_tokens"] += output

            totals["uncached_input_tokens"] += uncached
            totals["cache_read_input_tokens"] += cache_read
            totals["cache_creation_tokens"] += cache_creation
            totals["output_tokens"] += output

            model = item.get("model") or "unknown"
            slot = by_model.setdefault(
                model,
                {"uncached_input_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation_tokens": 0, "output_tokens": 0},
            )
            slot["uncached_input_tokens"] += uncached
            slot["cache_read_input_tokens"] += cache_read
            slot["cache_creation_tokens"] += cache_creation
            slot["output_tokens"] += output

        date = str(bucket.get("starting_at") or "")[:10]
        daily.append({"date": date, **day_totals})

    denom = (
        totals["cache_read_input_tokens"]
        + totals["cache_creation_tokens"]
        + totals["uncached_input_tokens"]
    )
    cache_hit_ratio = totals["cache_read_input_tokens"] / denom if denom else 0.0

    cost_by_model: dict[str, float] = {}
    real_cost_usd = 0.0
    for bucket in cost_buckets:
        for item in bucket.get("results") or []:
            try:
                usd = float(item.get("amount") or 0) / 100.0
            except (TypeError, ValueError):
                continue
            real_cost_usd += usd
            model = item.get("model") or "unattributed"
            cost_by_model[model] = cost_by_model.get(model, 0.0) + usd

    return {
        "available": True,
        "window": {"starting_at": _iso(starting_at), "ending_at": _iso(ending_at), "days": days},
        "daily": daily,
        "by_model": [{"model": model, **stats} for model, stats in by_model.items()],
        "totals": totals,
        "cache_hit_ratio": cache_hit_ratio,
        "real_cost_usd": round(real_cost_usd, 4),
        "cost_by_model": [
            {"model": model, "usd": round(usd, 4)} for model, usd in cost_by_model.items()
        ],
    }
