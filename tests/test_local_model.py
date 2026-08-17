"""Tests for the local-model preflight check.

The pruner fails open by design, so an unusable local model degrades silently to
passthrough. Preflight is what turns that silence into a loud error before a
run that costs money -- which means a false negative here is as damaging as a
false positive.
"""

from __future__ import annotations

import pytest

from smartcontext.local_model import LocalModel

INSTALLED = ["gemma3:12b", "nemotron-3.5-lightning:latest", "qwen3-coder:latest"]


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


class FakeClient:
    """Stands in for httpx.AsyncClient; serves one canned /api/tags response."""

    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self._status = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url: str, timeout: float | None = None):
        return FakeResponse(self._payload, self._status)


def install(monkeypatch, names: list[str], status_code: int = 200) -> None:
    payload = {"models": [{"name": n} for n in names]}
    monkeypatch.setattr(
        "smartcontext.local_model.httpx.AsyncClient",
        lambda *a, **k: FakeClient(payload, status_code),
    )


async def test_preflight_accepts_an_exactly_named_model(monkeypatch):
    install(monkeypatch, INSTALLED)
    assert await LocalModel("gemma3:12b").preflight() is None


async def test_preflight_accepts_a_bare_name_as_latest(monkeypatch):
    """Ollama resolves a bare name to the :latest tag, so the check must too --
    otherwise a working configuration is reported as broken."""
    install(monkeypatch, INSTALLED)
    assert await LocalModel("nemotron-3.5-lightning").preflight() is None


async def test_preflight_reports_a_missing_model_and_lists_alternatives(monkeypatch):
    install(monkeypatch, INSTALLED)
    problem = await LocalModel("does-not-exist:70b").preflight()
    assert problem is not None
    assert "not pulled" in problem
    assert "gemma3:12b" in problem  # tells the user what they can use instead


async def test_preflight_catches_a_reachable_endpoint_with_no_models(monkeypatch):
    """The dashboard proxy answers even when the Ollama behind it is gone."""
    install(monkeypatch, [])
    problem = await LocalModel("gemma3:12b").preflight()
    assert problem is not None
    assert "no models" in problem


async def test_available_tracks_preflight(monkeypatch):
    install(monkeypatch, INSTALLED)
    assert await LocalModel("gemma3:12b").available() is True

    install(monkeypatch, [])
    assert await LocalModel("gemma3:12b").available() is False


@pytest.mark.parametrize("configured", ["gemma3:12b", "nemotron-3.5-lightning"])
async def test_unreachable_endpoint_names_the_fix(monkeypatch, configured):
    install(monkeypatch, INSTALLED, status_code=502)
    problem = await LocalModel(configured).preflight()
    assert problem is not None
    assert "ollama serve" in problem
