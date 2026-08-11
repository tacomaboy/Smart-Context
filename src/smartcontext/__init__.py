"""smart-context -- a local context manager for Claude API traffic.

A localhost reverse proxy that speaks the Anthropic Messages API. Oversized tool
results in the newest turn are filtered by a local model before they enter the
conversation; whatever is removed is kept in a local store that Claude can query
back through MCP.

The load-bearing constraint is prompt caching: cached input bills at ~0.1x, so
editing already-cached history costs more than it saves. Everything here works
at the tail of the request, where appending is free.
"""

from .config import Settings
from .store import Store

__version__ = "0.1.0"
__all__ = ["Settings", "Store", "__version__"]
