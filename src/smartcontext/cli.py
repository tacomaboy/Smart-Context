"""Command line entry points."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .config import Settings
from .local_model import LocalModel
from .store import Store


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .proxy import create_app

    settings = Settings.from_env()
    if args.mode:
        settings.mode = args.mode
    if args.port:
        settings.port = args.port
    if args.min_chars is not None:
        settings.min_block_chars = args.min_chars
    if args.capture:
        settings.capture = True
    settings.validate()

    app = create_app(settings)
    print(f"smart-context listening on http://{settings.host}:{settings.port}")
    print(f"  mode:     {settings.mode}")
    print(f"  upstream: {settings.upstream}")
    print(f"  store:    {settings.db_path}")
    if settings.capture:
        print(f"  capture:  ON -- raw, unredacted payloads written to {settings.captures_dir}")
    print()
    print("Point clients at it with:")
    print(f'  $env:ANTHROPIC_BASE_URL = "http://{settings.host}:{settings.port}"')
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
    return 0


def cmd_mcp(_: argparse.Namespace) -> int:
    from .mcp_server import main as mcp_main

    mcp_main()
    return 0


def cmd_stats(_: argparse.Namespace) -> int:
    settings = Settings.from_env()
    store = Store(settings.db_path)
    stats = store.stats()
    if not stats.get("requests"):
        print("No requests recorded yet. Start the proxy and route some traffic through it.")
        return 0

    print(json.dumps(stats, indent=2))
    print()
    hit = stats["cache_hit_ratio"]
    print(f"Cache hit ratio: {hit:.1%} of prompt tokens served from cache.")
    print(
        "Estimated prompt cost: "
        f"{stats['est_prompt_cost_after_usd']:.4f} USD after pruning vs "
        f"{stats['est_prompt_cost_before_usd']:.4f} USD before "
        f"({stats['est_cost_savings_pct']:.1%} saved, estimated)."
    )
    if hit < 0.3 and stats["total_prompt_tokens"] > 50_000:
        print(
            "  Low. Something is invalidating the prefix every turn -- a timestamp in\n"
            "  the system prompt, a varying tool list, or non-deterministic pruning."
        )
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    store = Store(settings.db_path)
    if args.handle:
        chunk = store.get_chunk(args.handle)
        print(chunk.content if chunk else f"No chunk with handle {args.handle!r}.")
        return 0 if chunk else 1

    results = store.search(args.query, limit=args.limit) if args.query else store.recent(args.limit)
    if not results:
        print("Nothing stored yet.")
        return 0
    for chunk in results:
        print(f"{chunk.handle}  (~{chunk.token_est} tokens)")
        print(f"  {chunk.preview()}")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    settings = Settings.from_env()
    print(f"mode:      {settings.mode}")
    print(f"upstream:  {settings.upstream}")
    print(f"store:     {settings.db_path}")

    available = asyncio.run(
        LocalModel(settings.local_model, settings.ollama_base, settings.local_timeout_s).available()
    )
    if available:
        print(f"local LLM: reachable ({settings.local_model})")
    else:
        print(f"local LLM: UNREACHABLE ({settings.local_model})")
        print("           The proxy still works -- it forwards requests unfiltered.")
        print("           Start it with: ollama serve")

    try:
        store = Store(settings.db_path)
        stats = store.stats()
        print(f"requests logged: {stats.get('requests', 0)}")
        print(f"chunks stored:   {stats.get('chunks_stored', 0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"store: ERROR {exc}")
        return 1
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from .sweep import format_table, load_captures, run_sweep

    settings = Settings.from_env()
    payloads = load_captures(settings.captures_dir)
    if not payloads:
        print(f"No captures found in {settings.captures_dir}.")
        print("Run the proxy with --capture (or SMARTCONTEXT_CAPTURE=1) and send some real traffic first.")
        return 1

    print(f"Loaded {len(payloads)} captured payload(s) from {settings.captures_dir}")
    print(f"Local model: {settings.local_model}")
    print()

    points = asyncio.run(run_sweep(payloads, settings, args.min_chars, args.keep_budget))
    print(format_table(points))
    print()
    print("reduction% is an offline token-count proxy, not billed dollars -- the upstream")
    print("prompt cache can't be replayed locally. Confirm the winner with a live run and")
    print("`smart-context stats` before committing to it.")
    return 0


def cmd_bakeoff(args: argparse.Namespace) -> int:
    from .bakeoff import JUDGE_MODEL, estimate_plan, format_table, run_bakeoff
    from .sweep import load_captures

    settings = Settings.from_env()
    payloads = load_captures(settings.captures_dir)[: args.limit]
    if not payloads:
        print(f"No captures found in {settings.captures_dir}.")
        print("Run the proxy with --capture (or SMARTCONTEXT_CAPTURE=1) and send some real traffic first.")
        return 1

    plan = estimate_plan(payloads, args.models)
    print(f"Captures: {plan['captures']}   Candidate models: {plan['models']}")
    print(f"  baseline live calls:        {plan['baseline_calls']}")
    print(f"  candidate answer calls:     {plan['candidate_calls']}")
    print(f"  judge calls ({JUDGE_MODEL}): {plan['judge_calls']}")
    print(f"  total live API calls:       {plan['total_live_calls']}")
    print(f"  rough cost estimate:        ${plan['est_cost_usd']:.2f}  (worst case, ignores caching)")

    if not args.yes:
        print()
        print("This spends real money on live Claude API calls. Re-run with --yes to proceed.")
        return 0

    try:
        import anthropic
    except ImportError:
        print("The 'anthropic' package is required for bakeoff. Install with: pip install 'smart-context[bakeoff]'")
        return 1

    client = anthropic.AsyncAnthropic()
    summaries = asyncio.run(run_bakeoff(payloads, args.models, settings, client, on_event=print))
    print()
    print(format_table(summaries))
    print()
    print("Scores come from a single fixed judge model -- a proxy for")
    print("answer quality, not a guarantee. Judge and target-model calls are live and billed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-context",
        description="Local context manager for Claude API traffic.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the reverse proxy")
    serve.add_argument("--mode", choices=["shadow", "prune"], help="override SMARTCONTEXT_MODE")
    serve.add_argument("--port", type=int, help="override SMARTCONTEXT_PORT")
    serve.add_argument(
        "--min-chars", type=int, default=None,
        help="override SMARTCONTEXT_MIN_BLOCK_CHARS (blocks smaller than this are never touched)",
    )
    serve.add_argument(
        "--capture", action="store_true",
        help="write every raw, unredacted request payload to <data_dir>/captures/ for `sweep` fixtures",
    )
    serve.set_defaults(func=cmd_serve)

    mcp = sub.add_parser("mcp", help="run the recall MCP server on stdio")
    mcp.set_defaults(func=cmd_mcp)

    stats = sub.add_parser("stats", help="show cache and token statistics")
    stats.set_defaults(func=cmd_stats)

    recall = sub.add_parser("recall", help="search or fetch stored context")
    recall.add_argument("query", nargs="?", default="")
    recall.add_argument("--handle", help="fetch one chunk verbatim")
    recall.add_argument("--limit", type=int, default=5)
    recall.set_defaults(func=cmd_recall)

    doctor = sub.add_parser("doctor", help="check configuration and dependencies")
    doctor.set_defaults(func=cmd_doctor)

    sweep = sub.add_parser(
        "sweep", help="offline: replay captured payloads through the pruner across thresholds"
    )
    sweep.add_argument(
        "--min-chars", type=int, nargs="+", default=[500, 1000, 2000, 4000, 8000],
        help="SMARTCONTEXT_MIN_BLOCK_CHARS values to try (default: 500 1000 2000 4000 8000)",
    )
    sweep.add_argument(
        "--keep-budget", type=int, nargs="+", default=None,
        help="SMARTCONTEXT_KEEP_BUDGET_CHARS values to try (default: current setting only)",
    )
    sweep.set_defaults(func=cmd_sweep)

    bakeoff = sub.add_parser(
        "bakeoff", help="live: compare local models on captured payloads, scored by a fixed judge"
    )
    bakeoff.add_argument(
        "--models", nargs="+", required=True,
        help="Ollama model tags to compare, e.g. --models qwen3-coder:latest llama3.2:3b",
    )
    bakeoff.add_argument(
        "--limit", type=int, default=5,
        help="max number of captures to use (default: 5) -- caps live API spend",
    )
    bakeoff.add_argument(
        "--yes", action="store_true",
        help="actually fire the live API calls; without this, only print the cost estimate",
    )
    bakeoff.set_defaults(func=cmd_bakeoff)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
