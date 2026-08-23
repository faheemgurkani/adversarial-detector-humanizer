# Contributing

## Environment

Use Python 3.11 and the project virtualenv:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install `[local]` only when you need Raschka model inference or MiniLM.

## Workflow

1. Open an issue for anything larger than a typo.
2. Keep the engine library-first. The CLI should stay a thin wrapper around `humanize()` / `score()`.
3. Do not add regex phrase-swappers or silent fallbacks when a rewriter key is missing.
4. Do not call Pangram or GPTZero inside the inner loop.
5. Add tests for edge cases (empty input, lock restoration, meaning drift, max rounds).
6. Run `python -m pytest` before you open a pull request.

## Commit style

This repository uses numbered commits of the form `#[n] Commit` on `main`. Feature branches may use clearer messages; the maintainer will squash or renumber if needed.

Use `scripts/git-sync.sh <n>` only if you intend to push with the local git identity.

## Code style

Ruff is configured in `pyproject.toml`. Prefer small, typed functions and Pydantic models for public contracts.

## What not to contribute

- Claims or marketing copy that a detector is “bypassed 100%”
- Training scripts that require a paper-style GRPO run (that is Phase 6, later)
- Telemetry that uploads user text
