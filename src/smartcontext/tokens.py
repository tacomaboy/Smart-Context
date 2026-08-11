"""Token accounting.

Two separate concerns, deliberately kept apart:

* **Exact counts** come from the upstream response's ``usage`` object. They are
  free (already in the response) and authoritative. All reporting uses these.
* **Estimates** are used only to decide whether a block is big enough to bother
  filtering. A char-based ratio is fine for a threshold and costs nothing.

We deliberately do *not* use ``tiktoken`` -- that is OpenAI's tokenizer and
undercounts Claude by a wide margin, worse on code than prose. If you ever need
an exact pre-flight count, call Anthropic's ``/v1/messages/count_tokens``.
"""

from __future__ import annotations

from typing import Any

# Rough chars-per-token for mixed English + code. Only ever used for thresholds.
CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str) -> int:
    """Cheap, deliberately approximate. Never report this as a real token count."""
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def block_text(block: Any) -> str:
    """Extract the text payload of a content block, whatever shape it arrived in."""
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""

    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block_text(item) for item in content)

    if isinstance(block.get("text"), str):
        return block["text"]
    return ""


def usage_summary(usage: dict[str, Any] | None) -> dict[str, int]:
    """Normalise an Anthropic ``usage`` object into flat ints.

    ``input_tokens`` is only the *uncached remainder* -- total prompt size is
    ``input_tokens + cache_creation_input_tokens + cache_read_input_tokens``.
    Reading ``input_tokens`` alone badly understates a cached conversation.
    """
    usage = usage or {}
    read = int(usage.get("cache_read_input_tokens") or 0)
    write = int(usage.get("cache_creation_input_tokens") or 0)
    fresh = int(usage.get("input_tokens") or 0)
    return {
        "input_tokens": fresh,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": write,
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_prompt_tokens": fresh + read + write,
    }


# Billing multipliers relative to base input price, per Anthropic's pricing.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_5M_MULTIPLIER = 1.25
CACHE_WRITE_1H_MULTIPLIER = 2.0


def relative_input_cost(summary: dict[str, int], ttl: str = "5m") -> float:
    """Input cost in units of 'one uncached input token'.

    This is what makes cache-safety measurable: a request that prunes half the
    context but busts the cache can score *worse* than one that prunes nothing
    and reads it all back at 0.1x.
    """
    write_mult = CACHE_WRITE_1H_MULTIPLIER if ttl == "1h" else CACHE_WRITE_5M_MULTIPLIER
    return (
        summary["input_tokens"]
        + summary["cache_read_input_tokens"] * CACHE_READ_MULTIPLIER
        + summary["cache_creation_input_tokens"] * write_mult
    )


# List price, USD per input token, base (uncached) rate -- for a rough dashboard
# cost-savings estimate only. Matched by substring against the logged model name,
# so date-suffixed IDs (claude-opus-4-6-20261001) still hit. Not a billing source
# of truth: real savings depend on whether the avoided tokens would have been
# fresh, cache-write, or cache-read.
_MODEL_INPUT_PRICE_PER_TOKEN = {
    "opus": 5.00 / 1_000_000,
    "sonnet": 3.00 / 1_000_000,
    "haiku": 1.00 / 1_000_000,
    "fable": 10.00 / 1_000_000,
}
_DEFAULT_INPUT_PRICE_PER_TOKEN = 3.00 / 1_000_000  # sonnet-tier fallback for unrecognised models


def price_basis_for_model(model: str | None) -> tuple[str, float]:
    """Return the price family used for rough dashboard cost estimates.

    The returned price is the base input price in USD per input token. Multiply
    by 1_000_000 to read it as dollars per million input tokens.
    """
    name = (model or "").lower()
    for key, price in _MODEL_INPUT_PRICE_PER_TOKEN.items():
        if key in name:
            return key, price
    return "sonnet", _DEFAULT_INPUT_PRICE_PER_TOKEN


def price_per_token(model: str | None) -> float:
    return price_basis_for_model(model)[1]
