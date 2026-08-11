# AGENTS

Machine-readable install/run notes for coding agents working in this repository.

## Scope

- This file is for agent-driven setup and operations.
- README is user-facing; this file is execution-facing.

## Non-interactive setup path

1. Ensure prerequisites:
- Python 3.11+
- uv installed
- Ollama installed
- >= 10 GB free disk for default model pull

2. Start Ollama headless:

```bash
ollama serve
```

3. Pull the default model:

```bash
ollama pull gemma3:12b
```

4. Install project:

```bash
uv venv
uv pip install -e ".[dev,mcp]"
```

5. Validate dependencies:

```bash
uv run smart-context doctor
```

6. Start proxy (default mode is prune):

```bash
uv run smart-context serve
```

7. Health check:

```bash
curl http://127.0.0.1:4711/_smartcontext/health
```

## Expected gates

- `uv run smart-context doctor` should include `local LLM: reachable (gemma3:12b)`.
- `/_smartcontext/health` should return JSON with `"ok": true`.
- If mode is default, health should report `"mode": "prune"`.

## Stop conditions

- If `ollama pull gemma3:12b` fails, stop and report the error. Do not switch to another model without user approval.
- If `doctor` reports local model unreachable, stop and ask whether to continue fail-open.
- If health check fails, stop and report endpoint/output.

## Authentication rules

- The proxy has no credentials of its own.
- It forwards auth headers from the client as-is.
- Do not instruct users to set `ANTHROPIC_API_KEY` unless they already use API-key auth.

## Consent boundaries

Ask for user approval before:

- Editing `~/.claude/settings.json`.
- Installing autostart services (launchd/systemd/Task Scheduler).
- Deleting `SMARTCONTEXT_DATA_DIR` or any SQLite data.

## Security and privacy note

- `~/.smartcontext/context.db` stores pruned prompt/tool content locally.
- Treat this as sensitive at-rest data in enterprise environments.
