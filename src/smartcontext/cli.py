"""Command line entry points."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

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
    print(f"scope:     {settings.scan_scope}")
    print(f"upstream:  {settings.upstream}")
    print(f"store:     {settings.db_path}")
    if settings.admin_api_key:
        print("real usage tracking: configured (dashboard will show actual billed cost/cache-hit rate)")
    else:
        print("real usage tracking: not configured (set SMARTCONTEXT_ADMIN_API_KEY to enable)")

    problem = asyncio.run(
        LocalModel(settings.local_model, settings.ollama_base, settings.local_timeout_s).preflight()
    )
    if problem is None:
        print(f"local LLM: ready ({settings.local_model})")
    else:
        print(f"local LLM: UNUSABLE ({settings.local_model})")
        print(f"           {problem}")
        print("           The proxy still works -- it forwards requests unfiltered,")
        print("           so nothing breaks, but nothing gets pruned either.")

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
    from .bakeoff import (
        JUDGE_MODEL,
        default_results_path,
        estimate_plan,
        format_rationales,
        format_table,
        run_bakeoff,
        save_results,
    )
    from .sweep import load_captures

    settings = Settings.from_env()

    if args.session:
        from .sweep import load_capture_sessions_keyed

        keyed = load_capture_sessions_keyed(settings.captures_dir)
        matches = [item for item in keyed if item[0].startswith(args.session)]
        if len(matches) > 1:
            print(f"{args.session!r} matches {len(matches)} sessions; be more specific.")
            return 1
        if not matches:
            print(f"No session matching {args.session!r}.")
            print("List them with: smart-context replay --models x --list-sessions")
            return 1
        session_key, payloads = matches[0]
        payloads = payloads[: args.limit]
        print(f"Session: {session_key}")
    else:
        payloads = load_captures(settings.captures_dir)[: args.limit]

    if not payloads:
        print(f"No captures found in {settings.captures_dir}.")
        print("Run the proxy with --capture (or SMARTCONTEXT_CAPTURE=1) and send some real traffic first.")
        return 1

    # Candidates are local models -- without Ollama every prune is a passthrough
    # and the bake-off scores nothing. Stop before spending anything.
    for model in args.models:
        problem = asyncio.run(
            LocalModel(model, settings.ollama_base, settings.local_timeout_s).preflight()
        )
        if problem:
            print(f"Local model unusable: {problem}")
            return 1

    min_chars = args.min_chars or [settings.min_block_chars]
    plan = estimate_plan(payloads, args.models, args.scopes, min_chars)
    print(
        f"Captures: {plan['captures']}   Candidate models: {plan['models']}   "
        f"Scan scopes: {', '.join(args.scopes)}"
    )
    print(
        f"  min_block_chars: {', '.join(str(m) for m in min_chars)}   "
        f"Candidates: {plan['candidates']}"
    )
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
    runs: list = []
    summaries = asyncio.run(
        run_bakeoff(payloads, args.models, settings, client, scopes=args.scopes,
                    min_chars_values=min_chars, on_event=print, runs_out=runs)
    )

    # Save before printing: a paid run must survive a closed terminal.
    out = Path(args.out) if args.out else default_results_path(settings.data_dir, "bakeoff")
    try:
        saved = save_results(
            out, command="bakeoff", summaries=summaries, runs=runs,
            invocation={
                "models": args.models, "scopes": args.scopes,
                "min_chars": min_chars, "session": args.session, "limit": args.limit,
            },
        )
        print(f"\nResults saved to {saved}")
    except OSError as exc:
        print(f"\nWARNING: could not save results ({exc}) -- copy the output below now")

    print()
    print(format_table(summaries))
    rationales = format_rationales(summaries)
    if rationales:
        print(rationales)
    print()
    print("Scores come from a single fixed judge model -- a proxy for")
    print("answer quality, not a guarantee. Judge and target-model calls are live and billed.")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from .bakeoff import (
        default_results_path,
        estimate_replay_plan,
        format_replay_table,
        run_replay,
        save_results,
    )
    from .sweep import load_capture_sessions_keyed

    settings = Settings.from_env()
    keyed = load_capture_sessions_keyed(settings.captures_dir)
    if not keyed:
        print(f"No multi-turn capture sessions found in {settings.captures_dir}.")
        print("Run the proxy with --capture (or SMARTCONTEXT_CAPTURE=1) and hold a real")
        print("conversation first -- a warm prefix needs at least two turns of one session.")
        return 1

    if args.list_sessions:
        print(f"{'idx':<5}{'session':<20}{'turns':>7}{'max msgs':>10}")
        print("-" * 42)
        for i, (key, payloads) in enumerate(keyed):
            widest = max((len(p.get("messages", [])) for p in payloads), default=0)
            print(f"{i:<5}{key:<20}{len(payloads):>7}{widest:>10}")
        return 0

    # --session takes an index or a session key (prefix is enough). Indexes shift
    # as new captures land and tie on turn count, so the key is the stable handle.
    selected = None
    if args.session.isdigit():
        index = int(args.session)
        if index < len(keyed):
            selected = keyed[index]
    else:
        matches = [item for item in keyed if item[0].startswith(args.session)]
        if len(matches) > 1:
            print(f"{args.session!r} matches {len(matches)} sessions; be more specific.")
            return 1
        if matches:
            selected = matches[0]

    if selected is None:
        print(f"No session matching {args.session!r}. Run with --list-sessions to see them.")
        return 1

    session_key, payloads = selected
    session = payloads[: args.limit]
    print(f"Replaying session {session_key}")
    if len(session) < 2:
        print("Need at least 2 turns to demonstrate a warm prefix; raise --limit.")
        return 1

    # The `off` arm never touches Ollama; every other arm silently degrades to
    # passthrough without it, which would make all arms send identical payloads
    # and the whole comparison meaningless. Stop before spending anything.
    if any(scope != "off" for scope in args.scopes):
        for model in args.models:
            problem = asyncio.run(
                LocalModel(model, settings.ollama_base, settings.local_timeout_s).preflight()
            )
            if problem:
                print(f"Local model unusable: {problem}")
                print()
                print("Pruning would fall back to passthrough on every block, so all arms")
                print("would send identical payloads and the results would be noise.")
                return 1

    plan = estimate_replay_plan(session, args.models, args.scopes)
    print(f"  turns:  {plan['turns']} of {len(payloads)} captured")
    print(f"  models: {', '.join(args.models)}")
    print(f"  scopes: {', '.join(args.scopes)}   arms: {plan['arms']}")
    print(f"  total live API calls:  {plan['total_live_calls']}")
    print(f"  rough cost estimate:   ${plan['est_cost_usd']:.2f}  (worst case, assumes zero cache hits)")

    gaps = max(0, plan["arms"] - 1)
    if args.cooldown > 0 and gaps:
        mins = args.cooldown * gaps / 60
        print(f"  cooldown between arms: {args.cooldown:.0f}s x {gaps} = ~{mins:.0f} min of waiting")
        print("    (arms share the upstream cache; waiting past its TTL makes each start cold)")
    elif gaps:
        print("  cooldown: DISABLED -- arms will share cache entries and the comparison")
        print("    will favour whichever arm runs last. Only use --cooldown 0 for a single arm.")

    if not args.yes:
        print()
        print("This spends real money on live Claude API calls. Re-run with --yes to proceed.")
        return 0

    try:
        import anthropic
    except ImportError:
        print("The 'anthropic' package is required for replay. Install with: pip install 'smart-context[bakeoff]'")
        return 1

    client = anthropic.AsyncAnthropic()
    summaries = asyncio.run(
        run_replay(session, args.models, settings, client, scopes=args.scopes,
                   ttl=args.ttl, cooldown_s=args.cooldown, on_event=print)
    )

    out = Path(args.out) if args.out else default_results_path(settings.data_dir, "replay")
    try:
        saved = save_results(
            out, command="replay", summaries=summaries,
            invocation={
                "models": args.models, "scopes": args.scopes, "session": session_key,
                "limit": args.limit, "ttl": args.ttl, "cooldown": args.cooldown,
            },
        )
        print(f"\nResults saved to {saved}")
    except OSError as exc:
        print(f"\nWARNING: could not save results ({exc}) -- copy the output below now")

    print()
    print(format_replay_table(summaries))
    print()
    print("'rel cost' is input cost in units of one uncached input token:")
    print("  fresh x1.0 + cache read x0.1 + cache write x1.25 (5m) or x2.0 (1h).")
    print("Lower wins. Compare each scope against the 'off' arm -- if a scope trims more")
    print("but scores a higher rel cost, it bought reduction with cache invalidation.")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    from .synth import FACTS, PROBES, write_session

    settings = Settings.from_env()
    written = write_session(
        settings.captures_dir,
        session_key=args.session_key,
        filler_chars=args.filler_chars,
    )
    print(f"Wrote {len(written)} synthetic capture(s) to {settings.captures_dir}")
    print(f"  session key: {args.session_key}")
    print(f"  facts planted: {len(FACTS)}   probes asked back: {len(PROBES)}")
    print(f"  filler per turn: ~{args.filler_chars:,} chars of worthless build log")
    print()
    print("Every probe has one correct answer that appears earlier in the conversation,")
    print("separated from the question by filler. An ideal pruner drops all the filler")
    print("and keeps every fact, so a wrong answer means pruning destroyed something.")
    print()
    print("Replay it with:")
    print(f"  smart-context replay --models <model> --session {args.session_key} --limit 8 --yes")
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
        help="override SMARTCONTEXT_MIN_BLOCK_CHARS (minimum block size to consider; set 0 for no floor)",
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
        "--min-chars", type=int, nargs="+", default=[0, 500, 1000, 2000, 4000, 8000],
        help="SMARTCONTEXT_MIN_BLOCK_CHARS values to try (default: 0 500 1000 2000 4000 8000)",
    )
    sweep.add_argument(
        "--keep-budget", type=int, nargs="+", default=None,
        help="SMARTCONTEXT_KEEP_BUDGET_CHARS values to try (default: current setting only)",
    )
    sweep.set_defaults(func=cmd_sweep)

    synth = sub.add_parser(
        "synth",
        help="offline: generate a capture session with known answers, for measuring pruning",
    )
    synth.add_argument(
        "--session-key", default="synthetic0000000",
        help="session key to write under (default: synthetic0000000)",
    )
    synth.add_argument(
        "--filler-chars", type=int, default=12000,
        help="size of the worthless tool output planted each turn (default: 12000)",
    )
    synth.set_defaults(func=cmd_synth)

    bakeoff = sub.add_parser(
        "bakeoff", help="live: compare local models on captured payloads, scored by a fixed judge"
    )
    bakeoff.add_argument(
        "--models", nargs="+", required=True,
        help="Ollama model tags to compare, e.g. --models qwen3-coder:latest llama3.2:3b",
    )
    bakeoff.add_argument(
        "--session", default=None,
        help=(
            "restrict to one captured session, by key or prefix (e.g. synthetic0000000). "
            "Without it, captures are taken oldest-first across every session."
        ),
    )
    bakeoff.add_argument(
        "--min-chars", type=int, nargs="+", default=None, dest="min_chars",
        help=(
            "SMARTCONTEXT_MIN_BLOCK_CHARS values to score, e.g. --min-chars 2000 8000. "
            "This is the judged version of `sweep`: sweep shows what each threshold "
            "saves, this shows what it costs in answer quality. Each value is a "
            "separate candidate, so listing three triples the live API spend."
        ),
    )
    bakeoff.add_argument(
        "--scopes", nargs="+", choices=["tail", "full"], default=["tail"],
        help=(
            "scan scopes to compare, e.g. --scopes tail full. Each (model, scope) pair "
            "is scored separately, so listing both doubles the live API spend."
        ),
    )
    bakeoff.add_argument(
        "--limit", type=int, default=5,
        help="max number of captures to use (default: 5) -- caps live API spend",
    )
    bakeoff.add_argument(
        "--yes", action="store_true",
        help="actually fire the live API calls; without this, only print the cost estimate",
    )
    bakeoff.add_argument(
        "--out", default=None,
        help="where to save results (default: <data-dir>/results/<timestamp>-bakeoff.json)",
    )
    bakeoff.set_defaults(func=cmd_bakeoff)

    replay = sub.add_parser(
        "replay",
        help="live: replay one captured conversation turn by turn to measure real cache cost",
    )
    replay.add_argument(
        "--models", nargs="+", required=True,
        help="Ollama model tags to replay with, e.g. --models gemma3:12b",
    )
    replay.add_argument(
        "--scopes", nargs="+", choices=["off", "tail", "full"], default=["off", "tail", "full"],
        help="arms to compare; 'off' sends every turn unmodified as the baseline",
    )
    replay.add_argument(
        "--session", default="0",
        help=(
            "which captured session to replay: an index (0 = longest) or a session "
            "key/prefix. Prefer the key -- indexes shift as new captures land."
        ),
    )
    replay.add_argument(
        "--list-sessions", action="store_true",
        help="list the captured sessions with their keys and turn counts, then exit",
    )
    replay.add_argument(
        "--limit", type=int, default=8,
        help="max turns to replay from that session (default: 8) -- caps live API spend",
    )
    replay.add_argument(
        "--ttl", choices=["5m", "1h"], default="5m",
        help="cache TTL to price writes at (default: 5m)",
    )
    replay.add_argument(
        "--cooldown", type=float, default=330.0,
        help=(
            "seconds to wait between arms so each starts against an expired cache "
            "(default: 330, just over the 5m TTL). Without it, arms share cache "
            "entries and later arms look artificially cheap. 0 disables."
        ),
    )
    replay.add_argument(
        "--yes", action="store_true",
        help="actually fire the live API calls; without this, only print the cost estimate",
    )
    replay.add_argument(
        "--out", default=None,
        help="where to save results (default: <data-dir>/results/<timestamp>-replay.json)",
    )
    replay.set_defaults(func=cmd_replay)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
