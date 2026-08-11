"""Configuration, all env-driven so the proxy can be reconfigured without edits."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_UPSTREAM = "https://api.anthropic.com"

# Ollama's real port; 4700 is the local usage dashboard that proxies to it.
OLLAMA_PORTS = ("http://localhost:4700", "http://localhost:11434")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime settings.

    ``mode`` controls runtime behavior:
      prune   - filter oversized tool results in the final user turn.
      shadow  - never modify a request; only measure.
    """

    mode: str = "prune"
    upstream: str = DEFAULT_UPSTREAM
    host: str = "127.0.0.1"
    port: int = 4711

    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("SMARTCONTEXT_DATA_DIR", Path.home() / ".smartcontext")))

    # Blocks smaller than this are never touched -- the risk isn't worth the tokens.
    min_block_chars: int = 4000
    # Target size for the kept portion of a filtered block.
    keep_budget_chars: int = 1500
    chunk_chars: int = 900

    local_model: str = "gemma3:12b"
    local_timeout_s: float = 20.0
    ollama_base: str | None = None

    upstream_timeout_s: float = 600.0

    # Off by default: writes every raw request payload to disk, unredacted.
    # Only for building fixtures to feed `smart-context sweep` offline.
    capture: bool = False

    # On by default: trim oversized `tools` catalogs before forwarding.
    trim_tools: bool = True
    max_tools: int = 64
    trim_tools_retry_missing: bool = True

    @property
    def db_path(self) -> Path:
        return self.data_dir / "context.db"

    @property
    def captures_dir(self) -> Path:
        return self.data_dir / "captures"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            mode=os.environ.get("SMARTCONTEXT_MODE", "prune").strip().lower(),
            upstream=os.environ.get("SMARTCONTEXT_UPSTREAM", DEFAULT_UPSTREAM).rstrip("/"),
            host=os.environ.get("SMARTCONTEXT_HOST", "127.0.0.1"),
            port=_env_int("SMARTCONTEXT_PORT", 4711),
            min_block_chars=_env_int("SMARTCONTEXT_MIN_BLOCK_CHARS", 4000),
            keep_budget_chars=_env_int("SMARTCONTEXT_KEEP_BUDGET_CHARS", 1500),
            chunk_chars=_env_int("SMARTCONTEXT_CHUNK_CHARS", 900),
            local_model=os.environ.get("SMARTCONTEXT_LOCAL_MODEL", "gemma3:12b"),
            ollama_base=os.environ.get("SMARTCONTEXT_OLLAMA_BASE") or None,
            capture=_env_bool("SMARTCONTEXT_CAPTURE", False),
            trim_tools=_env_bool("SMARTCONTEXT_TRIM_TOOLS", True),
            max_tools=_env_int("SMARTCONTEXT_MAX_TOOLS", 64),
            trim_tools_retry_missing=_env_bool("SMARTCONTEXT_TRIM_TOOLS_RETRY_MISSING", True),
        )

    def validate(self) -> None:
        if self.mode not in {"shadow", "prune"}:
            raise ValueError(f"SMARTCONTEXT_MODE must be 'shadow' or 'prune', got {self.mode!r}")
        if self.upstream.rstrip("/").startswith(f"http://{self.host}:{self.port}"):
            raise ValueError("upstream points back at this proxy -- that would loop forever")
        if self.max_tools < 1:
            raise ValueError("SMARTCONTEXT_MAX_TOOLS must be >= 1")
