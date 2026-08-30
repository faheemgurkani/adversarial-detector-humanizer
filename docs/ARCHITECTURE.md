# Architecture

Engine layout and backend product shell. **Transformation plan:** [ROADMAP.md](ROADMAP.md). **HTTP contract:** [BACKEND_PRD.md](BACKEND_PRD.md).

---

## System layers

```
Doors          cli.py · api.py (sync + jobs)
                  ↓
Product shell  service.py · config.py · doctor.py · jobs/ · idempotency.py · errors.py · ids.py
                  ↓
Engine         engine.py · report.py · gates/ · preserve.py · …
                  ↓
Plugins        registry.py ← entry_points (adh.detectors · adh.rewriters · adh.gates)
```

**Hard rule:** Hosted SaaS, MCP, n8n, and LangChain integrations call `service.run_humanize()` (or `engine.humanize()` in tests). Never reimplement the loop in another language or transport.

---

## Humanize loop (engine)

```
input
  → optional scrub (Unicode)
  → detector.score (document)
  → optional structural prepass (flagged paragraphs only)
  → split sentences (pysbd, offsets kept)
  → detector.score_spans (per sentence)
  → flag sentences over threshold (or top-k, max_rewrite_ratio)
  → preserve-lock facts → rewriter (best-of-N, optional history)
  → logprob + detector blend → meaning gate stack → restore locks
  → optional hard-mode token guidance (stubborn sentences)
  → reassemble → document-level gate → rescore
  → repeat until target, max rounds, or no valid candidate
  → optional verify + detector breakdown
  → RunReport (+ report_id assigned in service layer)
```

---

## Protocols and plugins

| Protocol | Methods | Implementations |
|----------|---------|-----------------|
| **Detector** | `score()`, `score_spans()` | Raschka local, fake, statistical, ensemble, Pangram, GPTZero |
| **Rewriter** | `rewrite()`, `rewrite_candidates()` | OpenAI-compatible, identity (`fast` profile) |
| **SemanticGate** | similarity check | MiniLM, lexical |
| **Translator** | `translate()` | LLM, Google (optional), identity (tests) |

`registry.load_plugin(group, name, **kwargs)` loads constructors from `pyproject.toml` entry points. `factory.py` adds policy (e.g. `assert_inner_loop_detector` blocks Pangram/GPTZero in the inner loop).

---

## Configuration

Single config shape across CLI, HTTP, and yaml:

| Source | Module | Notes |
|--------|--------|-------|
| `adh.yaml` / `ADH_CONFIG` | `config.load_config()` | Nested yaml → flat `AdhConfig` |
| Profiles | `profiles.py` | `fast`, `standard`, `quality`, `verify-only` |
| Overrides | `resolve_adh_config()` | precedence: defaults → profile → file → explicit CLI/HTTP fields |

`adh init` writes `examples/adh.yaml` template. `adh doctor` validates the resolved config against the environment (keys, torch, model artifacts).

---

## Service layer

`service.run_humanize(text, config=…)` is the **only** production path for humanize:

1. Resolve `AdhConfig` from `HumanizeRequest`, `AdhConfig`, or `EngineConfig`
2. Load detector / rewriter / gate via factory (or use injected adapters in tests / `adh serve`)
3. Call `engine.humanize()`
4. Attach `report_id` (`report_` + 16 hex)

`service.run_score()` mirrors this for scoring-only routes.

CLI and HTTP are thin: parse input → call service → format output (Typer table, JSON, compact response).

---

## HTTP surfaces

### Sync (`POST /v1/humanize`)

- Default `compact: true` — agent-friendly payload with `agent_hint`, `output` alias, `metadata`
- `Idempotency-Key` + body hash → cached response (24h TTL), no duplicate engine runs
- Structured errors via global `AdhError` handler
- `X-Request-Id: req_…` on every response (middleware)

### Async jobs

```
POST /v1/jobs/humanize  → 202 + job_id + Location
GET  /v1/jobs/{job_id}  → 200 always (pending | processing | done | failed)
```

- `jobs/store.py` — in-memory job registry + create idempotency
- `jobs/worker.py` — background thread; handler calls `service.run_humanize()`
- Failed jobs use the same `error` envelope as sync routes
- Completed jobs include `report_id` and nested compact `report`

---

## Meaning gates

`MeaningGateStack` (`gates/stack.py`) combines semantic similarity with mechanical vetoes: numerals, hedges, deletion, optional NLI entailment, optional role preservation.

Modes: `auto`, `minilm`, `lexical`, `full` via `--meaning-gate` / `meaning_gate_mode`.

---

## Detectors

- **Inner loop:** local Raschka exports, fake, statistical, `ensemble-local` (qwen3 + statistical, max)
- **Verify / score only:** Pangram, GPTZero — blocked as inner-loop drivers (latency + cost)

Default quality model: `qwen3-variable`. CI / test mode: `fake` via `--profile fast`.

---

## Stop reasons

`passed`, `max_rounds`, `no_flagged_sentences`, `all_candidates_rejected`, `max_rewrite_ratio`, `already_below_target`

Frozen in `/v1` contract. Each maps to an `agent_hint` string for HTTP compact responses (`hints.py`).

---

## Module map

| Module | Role |
|--------|------|
| `engine.py` | Closed loop |
| `service.py` | Shared use-case layer |
| `config.py` | yaml + profiles + merge |
| `registry.py` / `factory.py` | Plugin loading |
| `api.py` | FastAPI app |
| `cli.py` | Typer CLI |
| `doctor.py` | Pre-flight checks |
| `jobs/` | Async humanize queue |
| `idempotency.py` | Sync retry deduplication |
| `ids.py` | Prefixed resource IDs |
| `errors.py` | Error code catalog + envelope |
| `schemas.py` | HTTP Pydantic models |
| `report.py` | `RunReport` and related types |

---

## Environment

CLI loads `.env` via python-dotenv. Template: [`.env.example`](../.env.example). Setup: [SETUP.md](SETUP.md).

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ADH_REWRITER_MODEL` | Rewriter |
| `ADH_CONFIG` | Path to `adh.yaml` |
| `ADH_MODELS_DIR` | Raschka artifact cache |
| `PANGRAM_API_KEY`, `GPTZERO_API_KEY` | Verification scoring |

---

## Planned (ROADMAP Steps 7–9)

Docker image + compose, MCP server (`adh mcp serve`), Python SDK — all reuse the service layer and `/v1` contract documented in [BACKEND_PRD.md](BACKEND_PRD.md).
