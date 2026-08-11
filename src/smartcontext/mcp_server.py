"""MCP server exposing the local store back to Claude.

This is the other half of the bargain. The proxy is allowed to elide material
from a request *because* Claude can ask for it back through these tools. Without
recall, pruning is silent data loss; with it, pruning is a cache.

Register with Claude Code:

    claude mcp add smart-context -- uv run smart-context mcp
"""

from __future__ import annotations

from .config import Settings
from .store import Store

# The SDK renamed this class in 2.0 (mcp.server.fastmcp.FastMCP -> mcp.server.
# MCPServer). The surface we use -- constructor, @tool decorator, run() -- is
# identical across both, so accept either rather than pinning to one.
try:
    from mcp.server import MCPServer as _Server  # mcp >= 2.0
except ImportError:  # pragma: no cover - depends on installed SDK
    try:
        from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x
    except ImportError as exc:
        raise SystemExit(
            "The MCP server needs the 'mcp' package. Install it with:\n"
            "    uv pip install 'smart-context[mcp]'"
        ) from exc


def build_server(settings: Settings | None = None) -> "_Server":
    settings = settings or Settings.from_env()
    store = Store(settings.db_path)
    server = _Server("smart-context")

    @server.tool()
    def context_recall(query: str, limit: int = 5) -> str:
        """Search context that was elided from earlier messages to save space.

        Call this when a tool result looks truncated, when you see a
        'smart-context' elision marker, or when you need detail from a file or
        command output that was summarised away.

        Args:
            query: Words expected to appear in the elided text.
            limit: Maximum excerpts to return (default 5).
        """
        results = store.search(query, limit=max(1, min(limit, 20)))
        if not results:
            return f"No stored context matched {query!r}."

        parts = [f"{len(results)} stored excerpt(s) matching {query!r}:\n"]
        for chunk in results:
            parts.append(
                f"--- handle: {chunk.handle} (~{chunk.token_est} tokens) ---\n{chunk.content}"
            )
        return "\n\n".join(parts)

    @server.tool()
    def context_get(handle: str) -> str:
        """Fetch one stored excerpt verbatim by its handle.

        Handles look like 'sc_1a2b3c4d5e6f' and appear in elision markers left
        in tool results.

        Args:
            handle: The handle to retrieve.
        """
        chunk = store.get_chunk(handle.strip())
        if chunk is None:
            return f"No stored context with handle {handle!r}."
        return chunk.content

    @server.tool()
    def context_recent(limit: int = 10) -> str:
        """List the most recently elided excerpts, newest first.

        Useful when you know something was just trimmed but not what to search
        for.

        Args:
            limit: How many to list (default 10).
        """
        results = store.recent(limit=max(1, min(limit, 50)))
        if not results:
            return "Nothing has been elided yet."
        return "\n".join(
            f"{c.handle}  (~{c.token_est} tokens)  {c.preview(160)}" for c in results
        )

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
