# Contributing

Backend-first open-core project. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for layer boundaries before adding features.

## Environment

Full install, extras, and `.env` notes: [docs/SETUP.md](docs/SETUP.md).

Use Python 3.11 and the project virtualenv:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

Install `[local]` only when you need Raschka model inference or MiniLM. Do not treat Pangram or GPTZero extras as part of the current contributor setup.

## Workflow

1. Open an issue for anything larger than a typo.
2. Keep the engine library-first. CLI and HTTP must call `service.run_humanize()` / `run_score()` — not duplicate adapter wiring.
3. Do not add regex phrase-swappers or silent fallbacks when a rewriter key is missing.
4. Do not call Pangram or GPTZero inside the inner loop.
5. Add tests for edge cases (empty input, lock restoration, meaning drift, max rounds).
6. HTTP routes live in `src/adh/api.py` and must stay covered by `tests/test_api.py` and `tests/test_jobs.py`. Contract: `docs/BACKEND_PRD.md`.
7. Run `python -m pytest` before you open a pull request.

## API review gate

Any pull request that touches **public field names** must keep these surfaces aligned:

| Surface | Location |
|---------|----------|
| HTTP request/response | `src/adh/schemas.py`, `docs/BACKEND_PRD.md` |
| CLI flags | `src/adh/cli.py` |
| Config file | `examples/adh.yaml`, `src/adh/templates/adh.yaml`, `src/adh/config.py` (`YAML_TO_ADH`) |

Checklist before merge:

1. Field added/renamed in one surface → update all three (or document why N/A).
2. `stop_reason` enum change → update `src/adh/report.py`, BACKEND_PRD, and OpenAPI enum test.
3. New error code → add to `src/adh/errors.py` (`ERROR_CODES`) and BACKEND_PRD error table.
4. Run `tests/test_config.py::test_config_field_names_match_humanize_request` and API/idempotency tests.

## Commit style

This repository uses numbered commits of the form `#[n] Commit` on `main`. Feature branches may use clearer messages; the maintainer will squash or renumber if needed.

Use `scripts/git-sync.sh <n>` only if you intend to push with the local git identity.

## Code style

Ruff is configured in `pyproject.toml`. Prefer small, typed functions and Pydantic models for public contracts.

## What not to contribute

- Claims or marketing copy that a detector is “bypassed 100%”
- Training scripts that require a paper-style GRPO run (that is Phase 6, later)
- Telemetry that uploads user text
