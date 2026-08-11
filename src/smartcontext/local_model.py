"""Local model client (Ollama).

Design rule: the local model emits **decisions, not prose**. It returns the
indices of chunks worth keeping -- never rewritten text. That keeps its output
tiny (sub-second on a 5090), and makes hallucinated file contents structurally
impossible: every byte forwarded upstream is verbatim from the original request.

Every failure path returns ``None`` so the caller falls back to passthrough.
An unreachable local model must degrade to "no filtering", never to an error.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import httpx

from .config import OLLAMA_PORTS

log = logging.getLogger("smartcontext.local")

PROMPT = """You are a context filter. Decide which numbered excerpts are needed \
to answer the request. Keep anything the answer would cite, reference, or depend on. \
Drop boilerplate, repetition, and unrelated material.

REQUEST:
{task}

EXCERPTS:
{chunks}

Reply with ONLY a JSON array of the integer indices to keep, most relevant first. \
Example: [0, 3, 4]. If unsure, keep the excerpt. No prose.

For tool catalogs, return [] if no tools are needed. No prose."""


@dataclass
class LocalDecision:
    keep: list[int]
    model: str


class LocalModel:
    def __init__(
        self,
        model: str = "gemma3:12b",
        base: str | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self._bases = [base] if base else list(OLLAMA_PORTS)
        self._live_base: str | None = None

    async def _resolve_base(self, client: httpx.AsyncClient) -> str | None:
        """Find a reachable endpoint, preferring the dashboard so calls get logged."""
        if self._live_base:
            return self._live_base
        for base in self._bases:
            try:
                resp = await client.get(f"{base}/api/tags", timeout=3.0)
                if resp.status_code == 200:
                    self._live_base = base
                    return base
            except Exception:
                continue
        return None

    async def available(self) -> bool:
        async with httpx.AsyncClient() as client:
            return await self._resolve_base(client) is not None

    async def select_chunks(self, task: str, chunks: list[str], keep_at_least: int = 1) -> LocalDecision | None:
        """Return the indices worth keeping, or ``None`` if the local model is unusable."""
        if not chunks:
            return LocalDecision(keep=[], model=self.model)

        numbered = "\n\n".join(f"[{i}] {c[:1200]}" for i, c in enumerate(chunks))
        prompt = PROMPT.format(task=task[:3000], chunks=numbered)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                base = await self._resolve_base(client)
                if base is None:
                    log.info("local model unreachable; passing context through unfiltered")
                    return None
                resp = await client.post(
                    f"{base}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        # Reasoning models (e.g. qwen3) default to thinking mode in
                        # Ollama and will burn the whole num_predict budget on hidden
                        # reasoning tokens, leaving "response" empty. This is a
                        # decision task, not a reasoning task -- we want the answer,
                        # not the thinking.
                        "think": False,
                        "options": {"temperature": 0, "num_predict": 128},
                    },
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")
        except Exception as exc:  # noqa: BLE001 - any failure means passthrough
            self._live_base = None
            log.warning("local model call failed (%s); passing context through", exc)
            return None

        keep = _parse_indices(raw, upper=len(chunks))
        if keep is None:
            log.warning("could not parse local model output; passing context through")
            return None

        # Never let the filter return nothing at all -- an empty keep set on a
        # parse quirk would silently blank a tool result.
        if not keep and keep_at_least:
            keep = list(range(min(keep_at_least, len(chunks))))
        return LocalDecision(keep=keep, model=self.model)


def _parse_indices(raw: str, upper: int) -> list[int] | None:
    """Pull a JSON array of indices out of whatever the model produced."""
    if not raw:
        return None

    candidates: list[object] = []
    stripped = raw.strip()
    with_context = [stripped]
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        with_context.append(fence.group(1).strip())
    for text in with_context:
        try:
            candidates.append(json.loads(text))
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:
            continue
        candidates.append(parsed)

    if re.fullmatch(r"\s*\d+(?:\s*,\s*\d+)*\s*", raw):
        candidates.append([int(part.strip()) for part in raw.split(",")])

    for candidate in candidates:
        parsed = _coerce_index_list(candidate)
        if parsed is not None:
            out: list[int] = []
            for item in parsed:
                if isinstance(item, bool):
                    continue
                if isinstance(item, int) and 0 <= item < upper and item not in out:
                    out.append(item)
            return out

    return None


def _coerce_index_list(parsed: object) -> list[object] | None:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, int) and not isinstance(parsed, bool):
        return [parsed]
    if isinstance(parsed, dict):
        for key in ("keep", "indices", "selected", "chunks"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, int) and not isinstance(value, bool):
                return [value]
    return None
