# Contributing

Thanks for contributing.

## Ground Rules

- Keep changes focused and small when possible.
- Preserve cache-safe behavior and request-shape invariants.
- Add or update tests for behavior changes.
- Keep docs accurate when commands, defaults, or behavior change.

## Local Setup

1. Install dependencies:

   ```bash
   uv venv
   uv pip install -e ".[dev,mcp]"
   ```

2. Run tests:

   ```bash
   uv run pytest -q
   ```

3. Optional quick checks during development:

   ```bash
   uv run smart-context doctor
   uv run smart-context stats
   ```

## Pull Request Checklist

- Tests pass locally.
- New behavior is covered by tests.
- README/docs reflect user-visible changes.
- No secrets or credentials are committed.

## Commit and Review Expectations

- Use clear commit messages that explain intent.
- Prefer one logical change per PR.
- Be explicit about cache-safety implications for pruning logic changes.

## Licensing

By submitting a contribution, you agree that your contribution is licensed
under the repository license (MIT).
