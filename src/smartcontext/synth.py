"""Procedurally generated capture sessions with known ground truth.

Real Claude Code captures make a poor measuring stick for pruning: they end in
tool calls rather than prose, so a replayed answer is often empty and the judge
scores "the answer is empty" instead of anything about pruning.

A synthetic session fixes that by construction:

* **Prose in, prose out.** Plain questions produce plain answers, so there is
  always something to score.
* **Known answers.** "What is my favorite color?" has exactly one right answer,
  so correctness is checkable directly -- no judge needed, no comparison against
  a reference that might itself be empty.
* **Facts planted early, asked late.** The fact and the question are separated
  by filler, so answering correctly *requires* that the fact survived trimming.
  That is precisely the property pruning puts at risk.
* **Filler sized to be trimmable.** Each turn carries a large tool_result so the
  pruner has real material to work on, and the filler is worthless by
  construction -- an ideal pruner would drop all of it and keep every fact.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "claude-opus-5"

SYSTEM = (
    "You are a helpful assistant. Answer using facts the user has told you "
    "earlier in this conversation. Answer in one short sentence."
)

# The filler arrives as a tool result, so the tool has to be declared or the
# request is rejected. Roles must also strictly alternate user/assistant --
# two assistant messages in a row is a 400, which is why "Noted." and the tool
# call share one message rather than being sent separately.
TOOLS = [
    {
        "name": "run_build",
        "description": "Run the project build and return the full log.",
        "input_schema": {
            "type": "object",
            "properties": {"target": {"type": "string", "description": "Build target."}},
            "required": ["target"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the repository and return its contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to read."}},
            "required": ["path"],
        },
    },
]

# A value that exists ONLY inside tool output -- the hardest case for a pruner.
# Build logs are safe to discard wholesale; a config dump is not, and the two
# look alike. Worse, the trim happens before anyone asks about it, so the local
# model has to keep it without knowing it will matter.
LISTEN_PORT = "58231"

# Turns that are neither a stated fact nor a probe: currently just the
# read_file call that plants LISTEN_PORT. Named so the session's shape stays
# derivable instead of hard-coded wherever it's checked.
INTERSTITIAL_TURNS = 1


@dataclass
class Probe:
    """A question with a known answer, and the fact it depends on."""

    question: str
    expected: str
    depends_on: str


# Facts are stated, then filler accumulates, then the facts are asked back.
# A CLAUDE.md-style instruction block, injected the way Claude Code injects it:
# a <system-reminder> text block inside the first user turn. Padded past the
# trimming thresholds on purpose -- the open question is whether losing this
# costs anything, and it can only be answered if it is big enough to be cut.
CLAUDE_MD = """<system-reminder>
As you answer the user's questions, you can use the following context:
# claudeMd
Codebase and user instructions are shown below. Be sure to adhere to these
instructions. IMPORTANT: These instructions OVERRIDE any default behavior.

## Response style
Answer in short sentences. If the user asks whether you should give long or
short answers, reply with exactly the word: short

## Project layout
The service entrypoint is src/app/main.py. Configuration is read from
config/settings.toml at startup and validated against config/schema.json.
Database migrations live in migrations/ and are applied with `make migrate`.
Tests live under tests/ and are run with `make test`; the suite must pass
before any commit. Linting is `make lint` and uses the repository's ruff
configuration, which is stricter than the ruff defaults.

## Conventions
Prefer small, focused modules. Keep public functions documented with a single
summary line. Avoid adding dependencies without discussion. When editing
generated files, change the generator instead. Log at INFO for lifecycle
events and DEBUG for per-request detail; never log credentials or request
bodies. Error messages should name the failing input, not just the failure.

## Deployment
Staging deploys automatically from the main branch. Production deploys are
manual and require a tagged release plus a green staging run. Rollbacks are
performed with `make rollback VERSION=<tag>` and are always preferred over
hotfixes applied directly to production.

## Review expectations
Every change needs a description of what it does and why, written for someone
who has not been following the work. Link the issue it closes. If the change
touches migrations, say explicitly whether it is backward compatible and what
the rollback path is. Reviewers should check behaviour before style; a comment
about naming should never block a correctness fix.

## Error handling
Validate at the system boundary and trust internal calls. Do not add defensive
checks for states that cannot occur -- they hide real failures and make the
control flow harder to follow. When an external call fails, surface the failing
input in the message. Retries are appropriate for timeouts and 5xx responses
only; never retry a validation failure, and never retry without backoff.

## Performance
Measure before optimising. The service is I/O bound in practice, so caching a
computed value is rarely the win it appears to be. Batch database access rather
than issuing queries in a loop. If a request path grows past roughly two
hundred milliseconds at the median, treat that as a bug rather than a tuning
opportunity, and find what changed.

## Dependencies
Prefer the standard library. A dependency has to earn its place: it should save
meaningful work, be actively maintained, and have a licence compatible with the
project. Pin exact versions in the lockfile and let the tooling update them on
a schedule rather than opportunistically during unrelated work.
</system-reminder>"""

# Answers must not occur anywhere in the filler, or the model can score
# correctly by reading material the pruner was free to drop. Short numbers are
# the trap here: "47" appears inside filler timings and module ids, so the
# lucky number is deliberately long. A test enforces this.
FACTS = [
    "My favorite day of the week is Friday.",
    "My favorite color is Blue.",
    "My lucky number is 8675309.",
    "I keep my project notes in a file called voyager.md.",
]

PROBES = [
    Probe("What is my favorite day of the week?", "Friday", FACTS[0]),
    Probe("What is my favorite color?", "Blue", FACTS[1]),
    Probe("What is my lucky number?", "8675309", FACTS[2]),
    Probe("What file do I keep my project notes in?", "voyager.md", FACTS[3]),
    # Not a fact the user stated -- an instruction from CLAUDE.md. Tests
    # whether trimming that block costs the model its operating rules.
    Probe("Should you give me long or short answers?", "short", "CLAUDE.md response style"),
    Probe("Where does this service read its configuration from?",
          "settings.toml", "CLAUDE.md project layout"),
    # Lives only in tool output, buried in a dump that looks like discardable
    # noise. Tests whether the pruner can tell useful tool results from filler.
    Probe("What port is the service configured to listen on?",
          LISTEN_PORT, "read_file tool output"),
]


def _filler(turn: int, chars: int) -> str:
    """Bulky, plausible, and information-free.

    Modelled on a build log: the shape the pruner is meant to target, and
    content no correct answer could ever depend on.
    """
    lines = []
    i = 0
    while sum(len(x) for x in lines) < chars:
        lines.append(
            f"[turn {turn}] module_{i:04d}.py  compiled ok  "
            f"({(i * 37) % 900 + 100} ms, {(i * 13) % 400 + 20} symbols, cache hit)"
        )
        i += 1
    return "\n".join(lines)


def _config_dump(chars: int) -> str:
    """A config file: mostly noise, one line that matters.

    The important line sits deep in the file rather than at the top, so keeping
    the first N characters isn't enough -- the pruner has to actually choose it.
    """
    lines = ["# generated configuration -- do not edit by hand"]
    planted = False
    i = 0
    while sum(len(x) for x in lines) < chars:
        if not planted and sum(len(x) for x in lines) > chars * 0.6:
            lines.append(f"listen_port = {LISTEN_PORT}")
            planted = True
        lines.append(f"feature_flag_{i:04d} = {'true' if i % 3 else 'false'}")
        i += 1
    if not planted:
        lines.append(f"listen_port = {LISTEN_PORT}")
    return "\n".join(lines)


def build_session(
    filler_chars: int = 12000,
    model: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """One payload per turn, each containing the whole conversation so far.

    Mirrors how a real client re-sends full history every turn -- the property
    that makes pruning decisions compound across a conversation.
    """
    payloads: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    turn = 0

    def snapshot(question: str, extra_blocks: list[dict[str, Any]] | None = None) -> None:
        nonlocal turn
        turn += 1
        blocks = list(extra_blocks or []) + [{"type": "text", "text": question}]
        messages.append({"role": "user", "content": blocks})
        payloads.append(
            {
                "model": model,
                "max_tokens": 512,
                "system": SYSTEM,
                "tools": [json.loads(json.dumps(t)) for t in TOOLS],
                "messages": [json.loads(json.dumps(m)) for m in messages],
            }
        )

    for index, fact in enumerate(FACTS):
        # The CLAUDE.md block rides in the first user turn, as the real client
        # sends it: once, never repeated, carried forward in history.
        extra = [{"type": "text", "text": CLAUDE_MD}] if index == 0 else None
        snapshot(fact, extra)

        # One assistant message carrying both the acknowledgement and the tool
        # call. Splitting them into two assistant messages breaks the required
        # user/assistant alternation and the API rejects the request.
        tool_id = f"toolu_synth_{index}"
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Noted."},
                {"type": "tool_use", "id": tool_id, "name": "run_build", "input": {"target": "all"}},
            ],
        })
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": _filler(turn, filler_chars)}
            ],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "The build succeeded."}],
        })

    # A read_file call whose output carries a real fact, sandwiched between the
    # facts and the questions so it has to survive the same trimming.
    snapshot("Read config/settings.toml so we have it on hand.")
    messages.append({
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Reading it now."},
            {"type": "tool_use", "id": "toolu_synth_cfg", "name": "read_file",
             "input": {"path": "config/settings.toml"}},
        ],
    })
    messages.append({
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_synth_cfg",
             "content": _config_dump(filler_chars)}
        ],
    })
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": "Loaded the configuration."}],
    })

    for probe in PROBES:
        snapshot(probe.question)
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": probe.expected}],
        })

    return payloads


def write_session(
    captures_dir: Path,
    session_key: str = "synthetic0000000",
    filler_chars: int = 12000,
    model: str = DEFAULT_MODEL,
) -> list[Path]:
    """Write the session in capture format so replay/bakeoff can load it."""
    captures_dir.mkdir(parents=True, exist_ok=True)
    payloads = build_session(filler_chars=filler_chars, model=model)
    base = time.time_ns()
    written = []
    for i, payload in enumerate(payloads):
        path = captures_dir / f"{base + i}_{session_key}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        written.append(path)
    return written


def score_answer(answer: str, probe: Probe) -> bool:
    """Ground truth beats a judge: the expected answer either appears or it doesn't."""
    return probe.expected.lower() in (answer or "").lower()


def probe_for_payload(payload: dict[str, Any]) -> Probe | None:
    """The probe this payload is asking, or None if it isn't asking one.

    Lets a caller check a real answer against known ground truth without
    knowing anything about how the session was built.
    """
    messages = payload.get("messages") or []
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            return None
        for probe in PROBES:
            if probe.question in text:
                return probe
        return None  # newest user turn isn't a probe; older ones don't count
    return None
