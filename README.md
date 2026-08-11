# smart-context

A localhost reverse proxy that speaks the Anthropic Messages API. A local model
(Ollama / default `gemma3:12b`) filters oversized tool results out of Claude
API requests before they're sent; everything removed is kept in a local SQLite
store that Claude can query back through MCP.

Point `ANTHROPIC_BASE_URL` at it and every Claude API client routes through it —
Claude Code, the SDKs, the `ant` CLI.

In practice, this means terminal/CLI clients that let you set an Anthropic base
URL are supported. Claude Desktop app sessions are not.

## Who this is for

- People already using Anthropic API clients who want to trim tool-heavy
  requests.
- People with local compute to spare, including MacBooks with enough unified
  memory and machines with a solid GPU.
- People who want to keep pruned context recoverable instead of throwing it
  away.

It is not for GitHub Copilot. Copilot does not expose the Anthropic base-URL
hook this proxy depends on.

It also is not for Claude Desktop. Claude Desktop talks to Claude's own
backend rather than letting you point it at a custom Anthropic base URL, so
there is nothing for this proxy to intercept there.

If you use Claude Code, run it in a terminal environment where you control
`ANTHROPIC_BASE_URL`.

## Open source and contributions

- This project is open source under the MIT License.
- Copyright is held by the smart-context contributors.
- Contributions are welcome. See CONTRIBUTING.md for workflow and expectations.

---

## The constraint that shapes everything

The obvious version of this idea — "have a local model prune the conversation
before each Claude call" — **costs more than doing nothing.**

Anthropic's prompt caching keys on an exact prefix match. Cached input bills at
**0.1×** the base rate; writing cache costs **1.25×** (5-minute TTL) or **2×**
(1-hour TTL). A long conversation is re-sent every turn, and almost all of it
reads back from cache. Rewriting an earlier message changes the prefix and
forces the whole remainder to be re-processed:

| Strategy on a 100k-token conversation | Relative input cost |
| --- | --- |
| Send it all, read from cache | **10,000 units** |
| Prune to 50k, but bust the cache | **62,500 units** |

Pruning half the context costs **six times more**. That comparison is pinned by
a test (`test_pruning_that_busts_the_cache_costs_more_than_not_pruning`).

So this tool never rewrites history. It works only at the **tail** of the
request, where content is new and appending is free.

## How it works

1. **Ingress filtering.** Only the newest turn is touched, and only
   `tool_result` blocks over a size threshold — file dumps, ripgrep output,
   build logs. These are new content that has never been cached, so filtering
   them invalidates nothing.
2. **Decisions, not prose.** The local model returns *indices of chunks to
   keep*. Every byte forwarded upstream is verbatim from the original request,
   so it cannot hallucinate file contents. Its output is a handful of integers,
   which keeps it fast.
3. **Nothing is lost.** Elided text goes to SQLite and is replaced in-band by a
   handle. Claude retrieves it with the `context_recall` MCP tool. Pruning is a
   cache, not deletion.
4. **Decisions are memoised by content hash.** Your client keeps its own full
   copy of every tool result and re-sends it each turn. If the same bytes were
   judged differently from one turn to the next, the prefix would change and
   every request would miss the cache. Deciding once — and identically forever
   after — is what makes filtering cache-safe.

### Invariants

Enforced by the test suite, in priority order:

- **Tail only.** Earlier messages and the system prompt are never modified.
- **Structure preserved.** A `tool_result` is shrunk, never removed — dropping
  one whose `tool_use` still exists is a hard 400. `tool_use_id` and `is_error`
  always survive.
- **Cache breakpoints respected.** Blocks carrying `cache_control` are skipped.
- **Fail open.** Ollama down, a malformed body, or a bug in the pruner all fall
  back to forwarding the original bytes. A context optimiser that can take your
  Claude access down with it is a bad trade at any savings.
- **Requests only.** Responses are never touched, so streaming relays
  byte-for-byte.

## Setup

### Prerequisites

- Python 3.11+
- `uv` installed: https://docs.astral.sh/uv/
- Ollama installed and runnable
- Disk: at least 10 GB free (the default `gemma3:12b` pull is about 8.1 GB)
- Enough RAM/VRAM for your chosen model

### 1) Install and run Ollama (headless first)

You need Ollama running locally before the proxy can prune anything.

Windows:

```powershell
winget install Ollama.Ollama
ollama serve
```

macOS:

```bash
brew install --cask ollama
ollama serve
```

GUI launch also works, but `ollama serve` is the deterministic path for scripts
and coding agents.

Verification:

```bash
ollama --version
```

### 2) Pull the default model

The default model is `gemma3:12b`.

```bash
ollama pull gemma3:12b
```

Verification:

```bash
ollama list
```

Expected: `gemma3:12b` appears in the list.

### 3) Set up the Python app

```bash
uv venv
uv pip install -e ".[dev,mcp]"
```

Verification:

```bash
uv run smart-context doctor
```

Expected: `local LLM: reachable (gemma3:12b)`.

### 4) Authentication note (important)

The proxy does not require credentials of its own. It forwards whatever auth
headers your client already sends (`x-api-key`, `authorization`, and friends).

Only set `ANTHROPIC_API_KEY` if you were already using API-key auth in your
client/session.

### 5) Start the proxy (prune mode by default)

`smart-context serve` starts in `prune` mode by default.

```bash
uv run smart-context serve
```

Verification:

```bash
curl http://127.0.0.1:4711/_smartcontext/health
```

Expected JSON includes `"ok": true` and `"mode": "prune"`.

If you want measurement-only behavior, start in shadow mode explicitly:

```bash
uv run smart-context serve --mode shadow
```

Tripwire (recommended): after a day of normal usage, run:

```bash
uv run smart-context stats
```

If pruning looks active but cache hit ratio falls materially, you can pay more,
not less. Switch to shadow mode and investigate before continuing.

Tools-catalog trimming is enabled by default to reduce large MCP/tools payloads:

```bash
SMARTCONTEXT_TRIM_TOOLS=true
SMARTCONTEXT_MAX_TOOLS=64
```

This keeps a capped subset of tools in each request (favoring recently used and
turn-relevant tool names via local-model selection, with heuristic fallback)
before forwarding upstream. It reduces prompt size, but can hide tools that are
not in the kept subset.

By default, if a trimmed request returns a missing-tool style error, the proxy
automatically retries once with the full original tool list.

If needed, you can disable tools-catalog trimming explicitly:

```bash
SMARTCONTEXT_TRIM_TOOLS=false
```

### 6) Point your Claude client at the proxy

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:4711"
```

For Claude Code, prefer the `env` block in `~/.claude/settings.json` because
the app sets `ANTHROPIC_BASE_URL` at launch:

```json
{
  "model": "sonnet",
  "env": { "ANTHROPIC_BASE_URL": "http://127.0.0.1:4711" }
}
```

After changing env vars or `settings.json`, restart already-running Claude
sessions/processes so they pick up the new values.

This setup applies to terminal-based Claude Code usage. It does not make Claude
Desktop app sessions route through the proxy.

### 7) Add MCP recall

```bash
claude mcp add smart-context -- uv run --directory /path/to/Smart\ Context smart-context mcp
```

Use your real checkout path for `/path/to/Smart Context`.

### 8) Optional autostart recipes

macOS (launchd), `~/Library/LaunchAgents/com.smartcontext.proxy.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.smartcontext.proxy</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string>
    <string>-lc</string>
    <string>ollama serve &amp; cd /path/to/Smart Context &amp;&amp; uv run smart-context serve</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict>
</plist>
```

Linux (systemd user), `~/.config/systemd/user/smart-context.service`:

```ini
[Unit]
Description=smart-context proxy
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/path/to/Smart Context
ExecStart=/usr/bin/env uv run smart-context serve
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
```

Windows Task Scheduler (PowerShell, user context):

```powershell
schtasks /Create /TN "smart-context" /SC ONLOGON /RL LIMITED /TR "powershell -NoProfile -ExecutionPolicy Bypass -Command \"Set-Location 'C:\\path\\to\\Smart Context'; ollama serve; uv run smart-context serve\""
```

### 9) Rollback / uninstall

- Stop the proxy process.
- Remove or unset `ANTHROPIC_BASE_URL` in your shell/profile.
- Remove the `env` override in `~/.claude/settings.json` if you added it.
- Restart Claude sessions so they stop routing through the proxy.
- Optional: remove local data at `SMARTCONTEXT_DATA_DIR` (default
  `~/.smartcontext`) if you no longer need recall history.

## Use

If you already have Ollama and `uv` installed, you can jump straight to step 2.

## Hardware notes

This is fastest when the local model is small enough to stay comfortably in
memory. A MacBook with Apple Silicon can work well for smaller models; so can a
desktop with a real GPU and plenty of VRAM. Bigger models often save more
context, but they can also be slower and less reliable.

The default model is `gemma3:12b` because it gave the best balance in testing.
If you want to override it, set `SMARTCONTEXT_LOCAL_MODEL`.

## Benchmarks

These are live bakeoff results from recent capture sets. Higher judge score is
better; higher reduction is better; lower trim time is better.

### Broad sweep

This was the wider model sweep used to pick the current default.

| Model | Runs | Errs | Avg score | Pass% | Reduction | Trim ms |
| --- | --- | --- | --- | --- | --- | --- |
| `gemma3:12b` | 5 | 1 | 4.50 | 100% | 35.5% | 15845 |
| `llama3.2:3b` | 5 | 0 | 3.40 | 60% | 42.3% | 5140 |
| `qwen2.5:14b` | 5 | 0 | 3.00 | 60% | 35.9% | 8504 |
| `qwen3.6:27b` | 5 | 0 | 2.00 | 20% | 37.4% | 18333 |
| `qwen3.6:latest` | 5 | 0 | 3.60 | 80% | 35.3% | 10323 |
| `qwen3:32b` | 5 | 0 | 2.80 | 60% | 33.7% | 17222 |
| `qwen3:4b` | 5 | 0 | 2.40 | 40% | 1.4% | 9657 |
| `qwen3-coder:latest` | 5 | 0 | 2.20 | 40% | 33.8% | 7322 |

### Latest head-to-head

This was the direct comparison on the newer capture set.

| Model | Runs | Errs | Avg score | Pass% | Reduction | Trim ms |
| --- | --- | --- | --- | --- | --- | --- |
| `gemma3:12b` | 5 | 0 | 2.20 | 40% | 33.0% | 25308 |
| `muse-glimmer` | 5 | 0 | 3.20 | 60% | 1.6% | 35701 |

In short: `gemma3:12b` stays the practical default because it trims far more
context and finishes faster, while `muse-glimmer` scored higher on the judge
but barely reduced the request.

### Dashboard

    http://127.0.0.1:4711/dashboard

Local context held versus context actually sent, refreshing every 5s:

- **Hero figure** — effective context multiplier: how much more the store holds
  than was sent upstream.
- **Stat tiles** — sent (exact), held locally (estimated), cache hit ratio,
  request count.
- **Prompt token composition** — stacked columns per request of cache read /
  cache write / fresh input. Cache read dominating is the healthy shape.
- **Held locally → sent upstream** — a dumbbell per filtered request showing
  what was trimmed.

Watch the **cache hit ratio** as closely as the savings. If pruning is working
but that number falls, you are paying more, not less — that is the failure mode
this whole design is built to avoid. Every chart has a table view, and the page
follows the OS theme or an explicit `data-theme` stamp.

### Other commands

```bash
uv run smart-context doctor          # check config, Ollama, store
uv run smart-context recall "query"  # search elided context
uv run smart-context stats           # cache hit ratio and token totals
```

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `SMARTCONTEXT_MODE` | `prune` | `prune` (default) or `shadow` (measure only) |
| `SMARTCONTEXT_PORT` | `4711` | Listen port |
| `SMARTCONTEXT_UPSTREAM` | `https://api.anthropic.com` | Real API endpoint |
| `SMARTCONTEXT_MIN_BLOCK_CHARS` | `4000` | Below this, blocks are never touched |
| `SMARTCONTEXT_KEEP_BUDGET_CHARS` | `1500` | Target size of the kept portion |
| `SMARTCONTEXT_LOCAL_MODEL` | `gemma3:12b` | Ollama model |
| `SMARTCONTEXT_OLLAMA_BASE` | auto | Tries `:4700` (dashboard) then `:11434` |
| `SMARTCONTEXT_DATA_DIR` | `~/.smartcontext` | SQLite location |
| `SMARTCONTEXT_TRIM_TOOLS` | `true` | If true, trim oversized `tools` catalogs before forwarding |
| `SMARTCONTEXT_MAX_TOOLS` | `64` | Max tools kept when tool-catalog trimming is enabled |
| `SMARTCONTEXT_TRIM_TOOLS_RETRY_MISSING` | `true` | Retry once with full tools if a trimmed call fails with missing-tool error |

## Endpoints

| Path | Purpose |
| --- | --- |
| `/v1/messages` | Filtered and forwarded |
| `/*` | Passthrough |
| `/dashboard` | Effectiveness dashboard |
| `/_smartcontext/health` | Config + whether the local model is reachable |
| `/_smartcontext/stats` | Cache hit ratio, token totals, relative cost |
| `/_smartcontext/dashboard-data` | Everything the dashboard renders |
| `/_smartcontext/requests` | Recent request log |
| `/_smartcontext/recall` | Search the store over HTTP |

## What this does and doesn't buy you

The Anthropic API already ships server-side **context editing**
(`clear_tool_uses_20250919`) and **compaction** (`compact_20260112`) as betas.
If all you want is old tool results cleared, turn those on — no proxy needed,
and no local model to keep running.

What this adds over them:

- **Recall.** The API's context editing clears content permanently. Here it
  stays retrievable, across sessions.
- **Local relevance judgement.** A model that sees your whole history decides
  what matters, rather than a fixed age-based rule.
- **Measurement.** Per-request cache hit ratios and relative input cost, which
  is the only way to know whether any of this is helping.

## Privacy

- The proxy relays auth headers but does not inspect or persist credentials.
- Pruned content is stored locally in SQLite for recall.
- By default this is an unencrypted at-rest file at `~/.smartcontext/context.db`.
- Treat the data directory as sensitive if your prompts contain sensitive
  material.
- If you need a clean slate, use the dashboard Reset button. Shift-click reset
  to wipe stored/recallable chunks as well.


## Cost estimates
- They use Anthropic's input list price for the model family logged on each request.
- All dashboard cost numbers are estimates, not invoice totals.
- Prompt cache effects can make real billed savings higher or lower than the
  list-price estimate.

## Notes

- Token counts in `stats` are exact — taken from each response's `usage` object,
  which is free. The internal size threshold uses a labelled char-based
  estimate. `tiktoken` is deliberately not used anywhere; it is OpenAI's
  tokenizer and undercounts Claude badly, worse on code than prose.
- `input_tokens` alone is only the *uncached remainder*. Total prompt size is
  `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`.
- The proxy never inspects or stores credentials — auth headers are relayed
  untouched.
- The consumer Claude desktop app **cannot** use this. It talks to claude.ai's
  backend rather than the Anthropic API, and its history lives server-side, so
  there is no local context to manage.

## Tests

```bash
uv run pytest -q
```

## Contributing

See CONTRIBUTING.md for setup, testing expectations, and PR checklist.

## License

MIT. See LICENSE.
