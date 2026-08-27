# ADH transformation roadmap

Single source of truth for turning **adversarial-detector-humanizer** from a research engine into a **local-first, backend utility** that humans, agents, CI, and workflows can install and embed — without a consumer UI.

**Positioning:** detector-verified text refinement. Verified score reduction — we show before/after, no bypass guarantees.

**Contract docs (current):** [BACKEND_PRD.md](BACKEND_PRD.md) · **Setup:** [SETUP.md](SETUP.md) · **Engine layout:** [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 1. One-sentence shift

**Stop building one program; start building one library with many doors.**

Every surface (CLI, HTTP, MCP, jobs, Python import) calls the same pure `humanize()` function. No logic duplicated across transports.

---

## 2. What exists today (engine v0.1)

The closed loop in `src/adh/engine.py` is implemented and tested:

| Area | Status | Module(s) |
|------|--------|-----------|
| Score → flag → rewrite → rescore loop | Done | `engine.py` |
| Meaning gate stack (numerals, hedges, deletion, NLI optional, roles optional) | Done | `gates/` |
| Preserve-lock + sentinel multiset | Done | `preserve.py` |
| Unicode scrub pre-loop | Done | `scrub.py` |
| AI-tells tie-break | Done | `tells.py` |
| Max ensemble + dual threshold (`verdict_score`) | Done | `detectors/remote.py`, `engine.py` |
| Post-loop verify bundle (Pangram/GPTZero) | Done | `verify.py` |
| Detector breakdown / transfer reporting | Done | `audit.py` |
| Logprob + detector blend ranking | Done | `ranking.py`, `rewriter.py` |
| Token-guided hard mode (optional GPU) | Done | `hard/` |
| Structural translation prepass (flagged paragraphs) | Done | `prepass/` |
| Statistical detector + `ensemble-local` | Done | `detectors/statistical.py`, `factory.py` |
| Rewriter history carry-over (multi-round) | Done | `rewriter.py`, `engine.py` |
| CLI (`score`, `humanize`, `serve`, `models`) | Done | `cli.py` |
| HTTP API (`/v1/score`, `/v1/humanize`, …) | Done | `api.py` |
| Benchmark harness | Done | `scripts/benchmark.py`, `docs/BENCHMARK.md` |

**Product shell gaps (this roadmap):** monolithic package, no config file, hardcoded factory registry, sync-only HTTP, no Docker, no MCP, no SDK, no `adh doctor`.

Reference clones for research live under `docs/resources/` (gitignored). See [resources/README.md](resources/README.md).

---

## 3. Target architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Doors (thin adapters — no business logic)                  │
│  adh-cli · adh-server · adh-mcp · adh-sdk · Python import  │
└───────────────────────────┬─────────────────────────────────┘
                            │ AdhConfig + text in → RunReport out
┌───────────────────────────▼─────────────────────────────────┐
│  adh-core (pure engine)                                     │
│  humanize() · score() · gates · preserve · report types     │
│  No HTTP, no env reads, no CLI imports                      │
└───────────────────────────┬─────────────────────────────────┘
                            │ protocols
┌───────────────────────────▼─────────────────────────────────┐
│  Plugins (entry_points)                                     │
│  adh.detectors · adh.rewriters · adh.gates · adh.prepass      │
└─────────────────────────────────────────────────────────────┘
```

### Package split (Phase 1)

| Package | Responsibility |
|---------|----------------|
| `adh-core` | `humanize()`, `EngineConfig`, `RunReport`, protocols, gate stack |
| `adh-cli` | Typer commands, `adh init`, `adh doctor`, reads `adh.yaml` |
| `adh-server` | FastAPI, sync + async jobs, OpenAPI |
| `adh-mcp` | MCP stdio tools over server/client |
| `adh-sdk` | Thin HTTP client (Python first; JS optional later) |

**Migration:** Start by extracting `adh-core` as a namespace inside `src/adh/` with strict import boundaries; split to separate PyPI packages only when boundaries are stable.

**Hard rule:** Hosted SaaS, MCP, n8n, LangChain — all call `adh_core.humanize()`. Never reimplement the loop elsewhere.

---

## 4. Design principles

1. **Engine dumb and pure** — config passed in as objects; no `os.environ` inside core.
2. **One config shape everywhere** — same fields in `adh.yaml`, CLI, HTTP body, MCP tool schema (no interface accepts a wider knob set than another).
3. **Profiles over flags** — agents and humans use `profile: standard`, not twenty tuning params.
4. **Honest tool descriptions** — “reduces local proxy scores; does not guarantee commercial detector pass.”
5. **Local loop, remote verify** — Pangram/GPTZero post-loop only; never inner sentence loop.
6. **Versioned contract** — `/v1/*` field names and `stop_reason` values frozen; breaking changes → `/v2`.
7. **Consistent naming** — `job_id` everywhere (not `task_id` in one place and `job_id` in another).
8. **Plugins use the same registry** — built-in detectors register via `entry_points` like third-party ones.
9. **Fast CLI startup** — use `importlib.metadata.entry_points()`, never `pkg_resources`.
10. **Meaning-safe first** — no attack-style full-doc rewrite without gates and locks.

---

## 4a. Interface discipline (Stripe + Ollama)

Two proven products, different domains, same pattern: **the engine matters less than a stable interface and a painless day one.**

### From Stripe — API-as-product

Treat API design, errors, and docs as the product — not an add-on.

| Habit | ADH application |
|-------|-----------------|
| **Prefixed resource IDs** | `job_a1b2…`, `report_x9y8…`, `req_…` on every job, run report, and request — add **before** external users exist |
| **Idempotency on mutating calls** | `Idempotency-Key` header (or body field) on `POST /v1/humanize` and `POST /v1/jobs/humanize` — retries must not run the loop twice |
| **Actionable errors** | Structured body: `code`, `message`, `retryable`, `doc_url`, `request_id` — never a bare string |
| **`metadata` on every object** | Optional `metadata: dict[str, string]` on jobs and compact reports — future-proofs v1 without schema breaks |
| **Consistency is the product** | Mandatory **API review gate** before any change to field names, `stop_reason` values, or config keys across CLI + HTTP + MCP |
| **Safe sandbox** | Profile `fast` + `fake` detector = **test mode**: zero OpenAI key, zero cost, full loop shape — try the product in ~30 seconds |

### From Ollama — frictionless install, familiar shapes

**Make the first five minutes frictionless** even if the engine is complex.

| Habit | ADH application |
|-------|-----------------|
| **One command install, one command run** | `pip install …` → `adh humanize --profile fast --detector fake --text "…"` (no keys) |
| **Don't invent vocabulary if one exists** | Job polling mirrors familiar async patterns (202 + poll 200 with status body, like Stripe/Pangram); compact response uses **`input` / `output` aliases** alongside existing fields where non-breaking |
| **Drop-in mental model** | Point existing HTTP clients at `adh serve` with minimal field mapping; optional OpenAI-compatible **rewriter** base URL is separate from ADH's own API shape |
| **Local-first default** | Works offline for score/humanize with `fake` + lexical gate — cloud keys are an upgrade path, not a gate |

### API review gate (process)

Before shipping any change to public contract:

1. Same field name in CLI JSON, HTTP body, MCP tool schema, and `adh.yaml`
2. Additive-only in `/v1` unless explicitly versioning to `/v2`
3. Update [BACKEND_PRD.md](BACKEND_PRD.md) + error catalog in same PR
4. Changelog entry for integrators

**Common thread:** integrators trust that names won't shift under them, and day one doesn't require hunting keys.

---

## 5. Unified config (`adh.yaml`)

Soup-style: `adh init` writes a default file; CLI and server read it.

```yaml
# adh.yaml — canonical config (v1)
profile: standard          # fast | standard | quality | verify-only

detector: ensemble-local   # inner-loop detector name
device: auto
models_dir: null           # default ~/.cache/adversarial-detector-humanizer/models

rewriter:
  backend: openai          # entry_point name
  model: gpt-4o-mini
  base_url: null           # env OPENAI_BASE_URL if null

humanize:
  target_score: 30
  verdict_score: 45
  max_rounds: 5
  prepass: none            # none | structural
  hard_mode: false

verify:
  detectors: []            # e.g. [pangram]
  threshold: 45
  on_input: true

deploy_detectors: []         # post-run breakdown
```

### Profiles (preset bundles)

| Profile | Detector | Rounds | Prepass | Verify | Keys needed | Use case |
|---------|----------|--------|---------|--------|-------------|----------|
| `fast` | `fake` | 1 | none | no | **none** | **Test mode / sandbox** — try full flow in ~30s |
| `standard` | `qwen3-variable` | 3–5 | none | optional | rewriter + optional local models | Default local use |
| `quality` | `ensemble-local` | 5 | structural optional | optional | rewriter + local models | Best local proxy |
| `verify-only` | `qwen3-variable` | 0 | none | yes | verify API keys | Score + Pangram, no rewrite |

CLI/API/MCP accept **`text` + `profile` + optional overrides** (`target_score`, `detector`) — not the full `EngineConfig` surface unless `advanced: true`.

---

## 6. Plugin registry

Replace `factory.py` if-chains with setuptools entry points. **Lock group names now** — they become a public contract.

```toml
# pyproject.toml (built-ins register the same way as third-party)
[project.entry-points."adh.detectors"]
fake = "adh.detectors.fake:FakeDetector"
qwen3-variable = "adh.detectors.local_raschka:LocalRaschkaDetector"
statistical = "adh.detectors.statistical:StatisticalDetector"
ensemble-local = "adh.detectors.ensemble:build_ensemble_local"
pangram = "adh.detectors.remote:PangramDetector"
gptzero = "adh.detectors.remote:GPTZeroDetector"

[project.entry-points."adh.rewriters"]
openai = "adh.rewriters.openai:OpenAICompatibleRewriter"
identity = "adh.rewriters.testing:IdentityRewriter"

[project.entry-points."adh.gates"]
auto = "adh.gates:build_meaning_gate_stack"
```

Loader (`adh.registry`):

```python
from importlib.metadata import entry_points

def load_plugin(group: str, name: str):
    eps = entry_points(group=group)
    ...
```

---

## 7. HTTP API — current and planned

**Current contract:** [BACKEND_PRD.md](BACKEND_PRD.md)

### 7.1 Sync humanize (freeze as stable)

- `POST /v1/humanize` — blocks until done
- **`compact: true` becomes default for agent clients** (full `RunReport` on `compact: false`)
- Add **`agent_hint`** string derived from `stop_reason` (additive field, non-breaking)
- Add **`report_id`** prefix (`report_…`) on every response
- Add optional **`metadata`** map (string keys/values, max 50 pairs) on request and echoed on response
- **`Idempotency-Key`** header: same key + same body within 24h returns cached `report_id` / result (no double humanize)
- **Familiar aliases** (non-breaking): compact body may include `input` (= source text hash or truncated input) and `output` (= `output_text`) for OpenAI-style client mental models

### 7.1b Error envelope (Stripe-style)

All HTTP errors use one shape:

```json
{
  "error": {
    "code": "rewriter_unavailable",
    "message": "OPENAI_API_KEY is not set",
    "retryable": false,
    "doc_url": "https://github.com/.../docs/SETUP.md#rewriter",
    "request_id": "req_abc123"
  }
}
```

Every response includes `X-Request-Id: req_…` (or `request_id` in JSON body). CLI `--json` errors use the same `code` values.

**Error code catalog** lives in BACKEND_PRD (maintained with API review gate).

### 7.2 Async jobs (Phase 6)

Long documents and agent timeouts require job-based execution:

```
POST /v1/jobs/humanize
  → 202 Accepted
  → Location: /v1/jobs/{job_id}
  → Body: { "job_id": "job_…", "status": "pending", "metadata": {} }

GET /v1/jobs/{job_id}
  → 200 OK always (while polling — not 202 on poll)
  → Body: { "job_id", "status", "report_id"?, "report"?, "error"?, "metadata"? }
```

**Job IDs:** always prefixed `job_`. Completed runs link to `report_id` (`report_…`).

**Idempotency:** same `Idempotency-Key` on job create returns existing `job_id` if still valid.

**Job status enum:** `pending` → `processing` → `done` | `failed` (mirror common async API vocabulary; document mapping in BACKEND_PRD)

**Structured error on failed:**

```json
{
  "error": {
    "code": "rewriter_unavailable",
    "message": "OPENAI_API_KEY is not set",
    "retryable": false,
    "doc_url": "https://github.com/.../docs/SETUP.md#rewriter",
    "request_id": "req_abc123"
  }
}
```

Optional: `POST /v1/jobs/humanize` accepts `webhook_url` for completion callback.

**Implementation sketch:** in-process queue first (SQLite or memory + worker thread); Redis/Celery only if multi-worker deployment needed.

---

## 8. Agent integration

Expose **small, honest tools** — not chat prose.

| Tool | Mutating? | Purpose |
|------|-----------|---------|
| `adh_score` | no | Check AI-likeness |
| `adh_humanize` | yes | Refine flagged sentences |
| `adh_verify` | no | Pangram/GPTZero after local pass |
| `adh_doctor` | no | Pre-flight setup check |

**Agent response shape (compact default):**

```json
{
  "report_id": "report_abc123",
  "output_text": "...",
  "output": "...",
  "ai_score_before": 78,
  "ai_score_after": 32,
  "stop_reason": "passed",
  "passed_verdict": true,
  "agent_hint": "Local score reached target. Run adh_verify if user asked about Pangram.",
  "metadata": {}
}
```

**Flows:**

- *Agent drafts then humanizes:* draft → `adh_score` → if high → `adh_humanize` → optional `adh_verify`
- *User drafts, agent humanizes at end:* `adh_humanize` once → report scores

**MCP:** `adh mcp serve` — stdio, tools call same `/v1` or in-process engine (Phase 8).

---

## 9. Build order

Front-load naming contracts and package boundaries; they are painful to change after third parties depend on them.

| Step | Deliverable | Why now |
|------|-------------|---------|
| **0** | **Test mode path** — document + default `fast`/`fake` zero-key flow; `adh try` one-liner | Ollama-style day one (~30s) |
| **0b** | **Prefixed IDs + error envelope** — `report_id`, `req_…`, structured `error` object spec in BACKEND_PRD | Stripe discipline; painful to add later |
| **1** | Package boundary: pure engine vs CLI/API imports | Prevents drift before more surface area |
| **2** | `entry_points` registry (`importlib.metadata`) | Lock `adh.detectors` / `adh.rewriters` group names |
| **3** | `adh.yaml` + `adh init` + profiles (incl. test mode) | Same config for CLI, server, agents |
| **4** | `adh doctor` | Trust + pre-flight before integrations |
| **5** | Freeze sync `/v1/humanize`; `compact` default; `agent_hint`; `metadata`; idempotency | Stable contract + API review gate |
| **6** | Async jobs (`job_…` IDs, idempotency, poll 200 + status, structured errors) | Long docs, real deployments |
| **7** | Docker image + compose | One-command team deploy |
| **8** | MCP server (`adh mcp serve`) | Cursor / Claude Code workflows |
| **9** | Python SDK + n8n/LangChain examples | Easiest once 1–6 stable |

### Per-step acceptance criteria

**Step 0 — Test mode (Ollama-style day one)**
- [x] `adh humanize --profile fast --detector fake --text "…" --json` works with no `.env` keys
- [x] SETUP.md "Try in 30 seconds" section at top
- [x] `adh try` (or documented one-liner) prints scores + stop_reason

**Step 0b — IDs and errors (Stripe-style)**
- [x] `report_id` with `report_` prefix on every humanize response
- [x] `X-Request-Id` / `req_` on all HTTP responses
- [x] Error envelope: `code`, `message`, `retryable`, `doc_url`, `request_id`
- [x] Error code catalog started in BACKEND_PRD

**Step 1 — Package boundary**
- [x] `adh.engine`, `adh.report`, `adh.gates`, `adh.preserve` import no `fastapi`, `typer`, `httpx` (rewriter adapters excepted behind protocols)
- [x] Single integration test: CLI and API produce identical `RunReport` for same config

**Step 2 — Registry**
- [x] All built-in detectors/rewriters registered via entry points
- [x] `load_detector("unknown")` lists available names from registry
- [x] CLI cold start < 200ms on typical machine (no `pkg_resources`)

**Step 3 — Config**
- [x] `adh init` writes `adh.yaml`
- [x] `adh humanize` reads config; CLI flags override file
- [x] `adh serve` loads same file from cwd or `ADH_CONFIG` path
- [x] Four profiles documented and tested

**Step 4 — Doctor**
- [x] Checks: Python version, optional `[local]` torch, rewriter key/URL, model artifacts, Pangram key if verify configured
- [x] Exit 0 = ready; exit 1 = actionable fix list

**Step 5 — Sync API polish**
- [x] `agent_hint` on compact response
- [x] Optional `metadata` on request/response
- [x] `Idempotency-Key` deduplicates humanize within TTL
- [x] OpenAPI documents all `stop_reason` values + error codes
- [x] API review checklist in CONTRIBUTING or ROADMAP §4a
- [x] BACKEND_PRD updated; semver policy written

**Step 6 — Async jobs**
- [ ] 202 on create; 200 + status body on poll (not 202 on GET)
- [ ] `job_id` prefixed `job_`; links to `report_id`
- [ ] Idempotency on job create
- [ ] Failed jobs use same structured `error` envelope as sync routes

**Step 7 — Docker**
- [ ] `Dockerfile` with `[api,local]` optional GPU variant
- [ ] Volume for model cache + `.env`
- [ ] `docker compose up` serves `/health`

**Step 8 — MCP**
- [ ] Tools: score, humanize, doctor, verify
- [ ] Honest descriptions in tool schema
- [ ] Uses profile + compact response

**Step 9 — SDK / integrations**
- [ ] `pip install adversarial-detector-humanizer[sdk]` or separate `adh-sdk` package
- [ ] Example: LangChain tool, n8n HTTP template, GitHub Action snippet

---

## 10. Remaining engine work (after product shell)

These are **engine enhancements**, not packaging. Lower priority than Steps 1–6.

| ID | Topic | Source | Module target |
|----|-------|--------|---------------|
| E1 | StealthRL HF LoRA rewriter backend | StealthRL | `rewriters/stealthrl.py` + entry point |
| E2 | GRPO-trained rewriter spec (KL + gates) | ai-detector-from-scratch | training script + optional backend |
| E3 | Pangram humanizer-head fields in verify report | Pangram API | `verify.py`, `report.py` |

---

## 11. Distribution model

| Tier | Who runs it | How users access |
|------|-------------|------------------|
| **Open-core (primary)** | User / team | `pip install`, Docker, `adh serve` on own infra |
| **Optional hosted API (later)** | You | API key + same `/v1` contract |
| **Agent / workflow** | User's agent | MCP or HTTP to local/ hosted server |

No credit card required for open-core. Hosted tier is optional follow-on (metering, keys) — not required for the utility to be useful.

---

## 12. Non-goals (this roadmap)

- Consumer paste UI / web SaaS (others can build on API)
- Chrome extension (depends on hosted API + iframe APIs)
- Auth / Stripe in open-core process (hosted layer only)
- Pangram/GPTZero in inner rewrite loop
- GRPO training before inference shell is stable
- Telemetry or storing user text by default

---

## 13. Documentation map (after consolidation)

| Doc | Role |
|-----|------|
| [ROADMAP.md](ROADMAP.md) | **This file** — transformation plan + build order |
| [BACKEND_PRD.md](BACKEND_PRD.md) | Frozen HTTP contract (current + additive changes) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Engine loop, protocols, modules |
| [SETUP.md](SETUP.md) | Install, env, first commands |
| [BENCHMARK.md](BENCHMARK.md) | Smoke benchmarks, metrics honesty |
| [resources/README.md](resources/README.md) | Local reference clones (gitignored) |

Removed: per-feature `future_work_plans/` shards and separate `PRODUCT.md` (merged here).

---

## 14. Success criteria

The utility is “product-ready” (open-core) when a new user can:

1. `pip install adversarial-detector-humanizer[dev]` (minimal extras)
2. **`adh humanize --profile fast --detector fake --text "Hello world." --json`** — no keys, ~30 seconds (test mode)
3. `adh init && adh doctor` → all green before production profile
4. `adh humanize --profile standard --file draft.txt --json` (with rewriter key)
5. `docker compose up` → `curl POST /v1/humanize` with `Idempotency-Key` and `compact: true`
6. Plug an MCP client into `adh mcp serve` and humanize a paragraph
7. Read [BACKEND_PRD.md](BACKEND_PRD.md) for contract, error codes, and `stop_reason` values

Without cloning the repo, reading twenty flags, or guessing env vars.

---

## 15. Implementation playbook (file-level)

Concrete **how / when / where** for each build step, mapped to the **current** tree under `src/adh/`. Each step lists new files, edits, dependencies, and **test cases** (file + function names to add).

**Current duplication to eliminate:** `cli.py` lines 178–207 and `api.py` lines 148–177 both hand-build `EngineConfig` and call `humanize()` — that wiring moves to a shared service layer in Step 1.

```
src/adh/
├── engine.py          ← keep pure; no new imports from cli/api
├── factory.py         ← Step 2: thin wrapper over registry
├── cli.py             ← doors: score, humanize, serve, init, doctor, try, mcp
├── api.py             ← doors: /v1/*, middleware, job routes
├── schemas.py         ← HTTP DTOs + error envelope
├── report.py          ← RunReport + report_id generation
├── exceptions.py      ← add .code on each AdhError subclass
├── service.py         ← NEW Step 1: run_humanize(), run_score()
├── registry.py        ← NEW Step 2: load_plugin(), list_plugins()
├── config.py          ← NEW Step 3: AdhConfig, load_yaml(), merge_profile()
├── profiles.py        ← NEW Step 3: PROFILE_PRESETS dict
├── ids.py             ← NEW Step 0b: new_report_id(), new_job_id(), new_request_id()
├── errors.py          ← NEW Step 0b: error_response(), ERROR_CODES
├── idempotency.py     ← NEW Step 5: IdempotencyStore (memory/SQLite)
├── jobs/              ← NEW Step 6: store.py, worker.py, routes in api.py
└── mcp_server.py      ← NEW Step 8
```

---

### Step 0 — Test mode (zero-key day one)

**Goal:** A new developer runs the full humanize loop in ~30s with no `.env` keys.

**When:** First — no dependencies on other steps except optionally registering `identity` rewriter (can ship in Step 0 or 2).

**Why blocked today:** `load_rewriter()` in `factory.py:66–67` always returns `OpenAICompatibleRewriter`, which reads `OPENAI_API_KEY` (`rewriter.py`). `tests/test_cli.py::test_humanize_without_key_fails` documents this failure.

| Where | How |
|-------|-----|
| **`src/adh/profiles.py`** (new) | Define `PROFILE_PRESETS["fast"]`: `detector="fake"`, `rewriter="identity"`, `max_rounds=1`, `allow_lexical_gate=True`, `semantic="lexical"`, `meaning_gate_mode="lexical"`. |
| **`src/adh/factory.py`** | Add branch or (later) registry entry: `load_rewriter(name="identity")` → `IdentityRewriter` from `rewriter.py:69`. |
| **`src/adh/cli.py`** | Add `--profile` option to `humanize_cmd` (default `None`; `--profile fast` applies preset before flags). Add `try` command: one-liner that humanizes sample text with `fast` profile and prints compact JSON. |
| **`docs/SETUP.md`** | Top section “Try in 30 seconds” with `pip install -e ".[dev]"` + `adh try`. |
| **`README.md`** | Link to SETUP try section. |

**Test cases** — add `tests/test_test_mode.py`:

| Test | Assert |
|------|--------|
| `test_fast_profile_no_openai_key` | `CliRunner` + `monkeypatch.delenv("OPENAI_API_KEY")`; `adh humanize --profile fast --text "Furthermore, note this." --json` → exit 0, valid JSON with `stop_reason`. |
| `test_try_command_exits_zero` | `adh try` → exit 0, stdout contains `stop_reason` and `output_text`. |
| `test_fast_profile_uses_fake_detector` | Parse JSON; `detector == "fake"`. |
| `test_profile_overridden_by_explicit_flag` | `--profile fast --detector statistical` → report/detector name is `statistical` (when local extras installed) or skip with `@pytest.mark.skipif`. |
| `test_fast_api_humanize_no_key` | `TestClient` with injected `IdentityRewriter`; `POST /v1/humanize` body `{"text":"…","detector":"fake","compact":true}` + profile field once Step 3 adds it → 200. |

**Regression:** Keep `test_humanize_without_key_fails` for **non-fast** profile (default still requires key until profile or `--rewriter identity` is explicit).

---

### Step 0b — Prefixed IDs + structured errors

**Goal:** Logs, support, and clients can correlate requests; errors are actionable (Stripe-style).

**When:** Immediately after Step 0 (or in parallel); before external integrators depend on response shape.

**Current state:** `api.py:131,179` raises `HTTPException(detail=str(error))` — bare string. `CompactHumanizeResponse` in `schemas.py:61–69` has no `report_id`. `ErrorBody` in `schemas.py:92–96` exists but is unused.

| Where | How |
|-------|-----|
| **`src/adh/ids.py`** (new) | `new_report_id() -> str` returns `report_` + 16 hex; same for `job_`, `req_`. Use `secrets.token_hex(8)`. |
| **`src/adh/errors.py`** (new) | Map each `AdhError` subclass → `{code, retryable, doc_url}`. `error_response(exc, request_id) -> dict`. Catalog constant `ERROR_CODES`. |
| **`src/adh/exceptions.py`** | Add optional `code: str` attribute on base `AdhError`; set on subclasses (`InputError.code = "invalid_input"`, etc.). |
| **`src/adh/report.py`** | Add optional field `report_id: str | None = None` on `RunReport`; set in `engine.humanize()` return path or in `service.run_humanize()`. |
| **`src/adh/schemas.py`** | Extend `CompactHumanizeResponse`: `report_id`, optional `output` alias (= `output_text`), optional `metadata`. Add `StructuredErrorResponse` model. |
| **`src/adh/api.py`** | Middleware: assign `request.state.request_id = new_request_id()`, set header `X-Request-Id`. Exception handler: return `{"error": {...}}` instead of `{"detail": "..."}`. |
| **`src/adh/cli.py`** | On `_fail`, if `--json`, print same error envelope as HTTP. |
| **`docs/BACKEND_PRD.md`** | New § “Error codes” table; document `X-Request-Id`, `report_id` on responses. |

**Test cases** — extend `tests/test_api.py` + new `tests/test_errors.py`:

| Test | Assert |
|------|--------|
| `test_humanize_response_includes_report_id` | 200 body has `report_id` matching `^report_[0-9a-f]{16}$`. |
| `test_request_id_header_on_all_routes` | `GET /health`, `POST /v1/score` → `X-Request-Id` header matches `^req_`. |
| `test_unknown_detector_structured_error` | `POST /v1/score` with `detector: "nope"` → 422, body `error.code == "unknown_detector"`, `error.request_id` present, `error.doc_url` is URL string. |
| `test_rewriter_missing_key_error` | No key, non-identity rewriter → 502, `error.code == "rewriter_unavailable"`, `retryable is False`. |
| `test_pangram_inner_loop_error` | Existing `test_humanize_pangram_stub_is_not_found_as_loop_detector` updated: assert `error.code == "remote_detector_unsupported"`. |
| `test_cli_json_error_matches_http_shape` | CLI `--json` on invalid input → stderr/stdout JSON with same `code` as HTTP. |
| `test_error_codes_documented_in_backend_prd` | `tests/test_setup_docs.py`: read BACKEND_PRD, assert each `ERROR_CODES` key appears in doc. |

---

### Step 1 — Package boundary (core vs doors)

**Goal:** Engine + report types import no transport; CLI and API share one service function so reports cannot drift.

**When:** Before Step 6 (jobs) or Step 8 (MCP) — any new surface should call the service, not copy wiring.

**Current violations to fix:**

| Module | Imports to remove from core path |
|--------|----------------------------------|
| `engine.py` | Already clean (no FastAPI/Typer). Keep it that way. |
| New **`service.py`** | `run_humanize(text, *, config: AdhConfig \| EngineConfig, ...) -> RunReport` — loads detector/rewriter/gate via factory, calls `humanize()`. |
| **`cli.py`** | Replace inline `EngineConfig(...)` block with `service.run_humanize(...)`. |
| **`api.py`** | Same; endpoints only parse `HumanizeRequest` → service call → serialize. |

| Where | How |
|-------|-----|
| **`src/adh/service.py`** (new) | `build_engine_config(from_request: HumanizeRequest \| AdhConfig) -> EngineConfig`. `run_humanize(...)`, `run_score(...)`. |
| **`src/adh/config.py`** (stub) | Minimal `AdhConfig` dataclass mirroring future yaml fields; Step 3 expands it. |
| **`tests/test_service.py`** (new) | Parity tests (below). |
| **`docs/ARCHITECTURE.md`** | Update “Surfaces” table: CLI/API → `service.py` → `engine.py`. |

**Import boundary test** — add `tests/test_import_boundary.py`:

| Test | Assert |
|------|--------|
| `test_core_modules_no_fastapi` | `importlib.util.find_spec` / AST scan: `engine`, `report`, `gates`, `preserve`, `detectors`, `ranking`, `audit` do not import `fastapi`, `typer`, `uvicorn`. |
| `test_service_produces_same_report_as_cli` | Same text + fake detector + identity rewriter: `service.run_humanize(...)` vs CLI subprocess or direct call → equal `stop_reason`, `score_before`, `score_after`, `output_text`. |
| `test_service_produces_same_report_as_api` | `TestClient` POST vs `service.run_humanize` with equivalent config → same compact fields. |
| `test_engine_config_fields_match_humanize_request` | Field parity checklist: every `HumanizeRequest` field maps to `EngineConfig` in `build_engine_config` (prevents drift). |

---

### Step 2 — Plugin registry (replace factory if-chain)

**Goal:** Built-in and third-party detectors/rewriters register by name; no edit to `factory.py` for new plugins.

**When:** After Step 1 (service loads via registry). Lock group names before publishing.

**Current state:** `factory.py:25–63` — 15+ `if/elif` branches. `pyproject.toml` has no `[project.entry-points]`.

| Where | How |
|-------|-----|
| **`pyproject.toml`** | Add `[project.entry-points."adh.detectors"]`, `"adh.rewriters"`, `"adh.gates"` per §6 of this doc. Map each existing class. |
| **`src/adh/registry.py`** (new) | `load_plugin(group, name, **kwargs)`, `list_plugins(group) -> list[str]`. Use `importlib.metadata.entry_points(group=group)` — never `pkg_resources`. |
| **`src/adh/factory.py`** | Replace body of `load_detector` with `registry.load_plugin("adh.detectors", name, ...)`. Keep `assert_inner_loop_detector` here. |
| **`src/adh/rewriter.py`** | Move `IdentityRewriter` registration target; optional `rewriters/testing.py` if splitting. |
| **`tests/fixtures_plugin/`** (new, dev only) | Tiny package with one fake entry point for integration test. |

**Test cases** — `tests/test_registry.py`:

| Test | Assert |
|------|--------|
| `test_load_builtin_fake_detector` | `load_detector("fake")` → `FakeDetector`, `.name == "fake"`. |
| `test_load_all_entry_point_detectors` | Every name in `list_plugins("adh.detectors")` loads without exception. |
| `test_unknown_detector_lists_available` | `load_detector("nope")` raises `InputError`; message contains at least `"fake"`. |
| `test_third_party_entry_point` | Install fixture plugin in test via `importlib.metadata` mock or `pytest` monkeypatch of `entry_points()` → load by custom name. |
| `test_registry_cold_start_under_200ms` | `time.perf_counter()` around first `load_detector("fake")` < 0.2s (mark `@pytest.mark.slow` optional). |
| `test_service_uses_registry` | After refactor, `service.run_score(..., detector="statistical")` works. |

---

### Step 3 — `adh.yaml` + profiles + `adh init`

**Goal:** One config file read by CLI, server, and (later) MCP; profiles bundle knobs.

**When:** After Steps 0–2 so profiles can reference registry names.

**Current state:** All config via CLI flags (`cli.py:131–163`) and HTTP body (`schemas.py:29–58`). Server `create_app()` binds detector at startup (`cli.py:115–122`) — does not read a file.

| Where | How |
|-------|-----|
| **`src/adh/config.py`** | Pydantic model `AdhConfig` matching §5 yaml. `load_config(path: Path \| None)` — cwd `adh.yaml`, else `ADH_CONFIG` env. `apply_profile(name) -> AdhConfig`. `merge_cli_overrides(config, **flags)`. |
| **`src/adh/profiles.py`** | `PROFILE_PRESETS: dict[str, partial[AdhConfig]]` for `fast`, `standard`, `quality`, `verify-only`. |
| **`examples/adh.yaml`** (new) | Committed template. |
| **`src/adh/cli.py`** | Commands: `init` (write `examples/adh.yaml` to cwd), `humanize`/`serve` call `load_config()`. Flags override file values. |
| **`src/adh/api.py`** | Optional `profile: str` on `HumanizeRequest`; server startup `create_app(config_path=...)`. |
| **`src/adh/schemas.py`** | Add `profile: str \| None`, `metadata: dict[str, str] \| None` to `HumanizeRequest`. |

**Test cases** — `tests/test_config.py`:

| Test | Assert |
|------|--------|
| `test_init_writes_adh_yaml` | `adh init` in tmp_path → `adh.yaml` exists, parses as valid YAML. |
| `test_load_config_from_cwd` | Write yaml with `profile: fast`; `load_config()` → detector fake, rewriter identity. |
| `test_cli_flag_overrides_yaml` | File says `target_score: 30`; CLI `--target 20` → `EngineConfig.target_score == 20`. |
| `test_serve_loads_same_config` | Set `ADH_CONFIG=tmp/adh.yaml`; create_app reads default detector from file. |
| `test_profile_standard_fields` | `apply_profile("standard")` → `max_rounds in (3,5)`, detector default `qwen3-variable`. |
| `test_unknown_profile_raises` | `apply_profile("nope")` → `InputError` with code `unknown_profile`. |
| `test_api_accepts_profile_field` | POST with `"profile": "fast"` only + text → 200 without OpenAI key (identity rewriter). |
| `test_config_field_names_match_humanize_request` | Automated diff: yaml keys ↔ `HumanizeRequest` fields (API review gate helper). |

---

### Step 4 — `adh doctor`

**Goal:** Pre-flight checklist before production profile or CI integration.

**When:** After Step 3 (reads same config).

| Where | How |
|-------|-----|
| **`src/adh/doctor.py`** (new) | Checks: Python ≥3.11, optional `[local]` torch import, `OPENAI_API_KEY` if profile needs rewriter, Raschka artifact for configured detector, Pangram key if `verify` non-empty, writable models dir. Returns `list[CheckResult]`. |
| **`src/adh/cli.py`** | `@app.command() def doctor(...)` — print table, exit 1 if any failed. |
| **`docs/SETUP.md`** | Document doctor output and fixes. |

**Test cases** — `tests/test_doctor.py`:

| Test | Assert |
|------|--------|
| `test_doctor_fast_profile_all_green_no_keys` | Config profile fast → all checks pass without env keys. |
| `test_doctor_standard_fails_without_rewriter_key` | Profile standard, no `OPENAI_API_KEY` → failed check with actionable message. |
| `test_doctor_reports_missing_local_model` | Profile standard + detector qwen3-variable + no artifact → `DetectorNotReady`-style warning. |
| `test_doctor_exit_code_1_on_failure` | CLI `adh doctor` with bad config → exit code 1. |
| `test_doctor_json_output` | `--json` → list of `{name, ok, message, fix}`. |

---

### Step 5 — Sync API polish (contract freeze)

**Goal:** Stable agent-facing sync API: compact default, idempotency, metadata, agent_hint, OpenAPI catalog.

**When:** After Steps 0b, 1, 3 — builds on IDs, service layer, profiles.

| Where | How |
|-------|-----|
| **`src/adh/idempotency.py`** (new) | In-memory dict keyed by `(Idempotency-Key, hash(body))` → cached `report_id` + response; TTL 24h. Optional SQLite for multi-worker later. |
| **`src/adh/api.py`** | Read header `Idempotency-Key` on `POST /v1/humanize`; return cached 200 on replay. Middleware already from 0b. |
| **`src/adh/schemas.py`** | `compact: bool = True` (breaking default change — document in BACKEND_PRD changelog). Add `agent_hint: str` on compact response; derive from `stop_reason` map in `service.py`. |
| **`src/adh/service.py`** | `agent_hint_for(report) -> str` e.g. passed → “Local score reached target…”. |
| **`docs/BACKEND_PRD.md`** | Full `stop_reason` enum, error catalog, idempotency semantics, metadata limits (50 keys). |
| **`CONTRIBUTING.md`** or ROADMAP §4a | API review checklist for PRs touching public fields. |

**Test cases** — extend `tests/test_api.py`, `tests/test_schemas.py`, new `tests/test_idempotency.py`:

| Test | Assert |
|------|--------|
| `test_compact_true_by_default` | POST `/v1/humanize` minimal body → response has `ai_score_before`, not full `sentences` array. |
| `test_compact_false_returns_full_report` | `"compact": false` → includes `sentences`, `locks`. |
| `test_agent_hint_present_on_compact` | Every `stop_reason` in enum → non-empty `agent_hint`. |
| `test_metadata_round_trip` | Request `metadata: {"ticket":"123"}` echoed on response. |
| `test_metadata_max_keys_rejected` | 51 keys → 422 `invalid_input`. |
| `test_idempotency_same_key_same_body` | Two POSTs same key → same `report_id`, humanize called once (mock/spy on engine). |
| `test_idempotency_same_key_different_body` | 409 or 422 `idempotency_key_reused`. |
| `test_output_alias_equals_output_text` | `output == output_text`. |
| `test_openapi_lists_stop_reasons` | `/openapi.json` description or enum includes all `StopReason` values. |

---

### Step 6 — Async jobs

**Goal:** Long humanize runs without gateway timeout; same engine, job polling semantics.

**When:** After Step 5 (reuse idempotency, error envelope, report_id).

| Where | How |
|-------|-----|
| **`src/adh/jobs/store.py`** (new) | `JobRecord`: `job_id`, `status`, `report_id?`, `error?`, `metadata`, `created_at`. SQLite or in-memory + background thread. |
| **`src/adh/jobs/worker.py`** (new) | Pull pending jobs → `service.run_humanize()` → update store. |
| **`src/adh/api.py`** | `POST /v1/jobs/humanize` → 202 + `Location`; `GET /v1/jobs/{job_id}` → 200 always while polling. |
| **`src/adh/schemas.py`** | `JobCreateRequest`, `JobResponse`. |
| **`src/adh/cli.py`** | Optional `adh humanize --async` → POST job, poll until done. |

**Test cases** — `tests/test_jobs.py`:

| Test | Assert |
|------|--------|
| `test_create_job_returns_202` | POST → 202, body `job_id` matches `^job_`, `status == "pending"`. |
| `test_poll_until_done` | GET after worker runs → `status == "done"`, `report_id` present, nested `report` or link. |
| `test_get_job_always_200_while_polling` | Pending/processing/done all return 200 (not 202 on GET). |
| `test_failed_job_structured_error` | Force rewriter error → `status == "failed"`, `error.code` set. |
| `test_job_idempotency` | Same `Idempotency-Key` on create → same `job_id`. |
| `test_job_metadata_persisted` | Metadata on create echoed on GET. |
| `test_cli_async_humanize` | `--async` prints job_id then final output (mock fast worker). |

---

### Step 7 — Docker

**Goal:** One-command team deploy with same contract as local.

**When:** After Steps 5–6 stable.

| Where | How |
|-------|-----|
| **`Dockerfile`** | Multi-stage: base + optional `[local]` CUDA variant. `CMD ["adh", "serve", "--host", "0.0.0.0"]`. |
| **`docker-compose.yml`** | Service `adh`, volume `./models:/root/.cache/...`, env file, port 8000. |
| **`.dockerignore`** | Exclude `.venv`, `docs/resources/`. |

**Test cases** — `tests/test_docker.py` (optional, `@pytest.mark.slow`):

| Test | Assert |
|------|--------|
| `test_dockerfile_builds` | `docker build -t adh:test .` exit 0. |
| `test_compose_health` | `docker compose up -d` → `curl /health` 200 within 60s. |
| `test_compose_humanize_fast_profile` | POST humanize with profile fast → 200. |

---

### Step 8 — MCP server

**Goal:** Cursor / Claude Code invoke score, humanize, doctor without custom HTTP glue.

**When:** After Steps 3–5 (profiles + compact + agent_hint).

| Where | How |
|-------|-----|
| **`src/adh/mcp_server.py`** (new) | stdio MCP; tools call `service.py` in-process (not reimplemented loop). |
| **`src/adh/cli.py`** | `adh mcp serve` subcommand. |
| **`pyproject.toml`** | Optional dep `mcp` extra. |

**Test cases** — `tests/test_mcp.py`:

| Test | Assert |
|------|--------|
| `test_mcp_lists_tools` | score, humanize, doctor, verify registered. |
| `test_mcp_humanize_fast_no_key` | Tool call with profile fast → compact JSON with agent_hint. |
| `test_mcp_schema_matches_http_fields` | Tool input schema keys ⊆ `HumanizeRequest` fields. |

---

### Step 9 — SDK + integrations

**Goal:** `pip install adh-sdk` (or extra) wraps `/v1` with typed client; examples for LangChain/n8n.

**When:** Last — contract frozen.

| Where | How |
|-------|-----|
| **`src/adh_sdk/`** or **`packages/adh-sdk/`** | `Client(base_url, api_key=None)`, `humanize(text, profile=...)`, `poll_job(job_id)`. |
| **`examples/langchain_tool.py`**, **`examples/n8n/`** | Copy-paste templates. |

**Test cases** — `tests/test_sdk.py`:

| Test | Assert |
|------|--------|
| `test_sdk_humanize_against_testclient` | SDK pointed at `TestClient` app → same response as raw HTTP. |
| `test_sdk_idempotency_header` | Client sends `Idempotency-Key` automatically on retry helper. |
| `test_sdk_poll_job` | Create job via SDK → poll until done. |

---

### Cross-cutting: API review gate (ongoing)

**When:** Every PR that touches `schemas.py`, `cli.py` options, `config.py`, `mcp_server.py` tool schemas, or `examples/adh.yaml`.

**Automated test** — `tests/test_contract_parity.py`:

| Test | Assert |
|------|--------|
| `test_humanize_fields_cli_api_yaml` | Set diff empty between public field names across the three surfaces (allowlist for transport-only fields). |
| `test_stop_reason_enum_frozen` | Snapshot test: changing `StopReason` in `report.py` fails until BACKEND_PRD + OpenAPI updated. |
| `test_error_codes_synced` | Every `ERROR_CODES` key in BACKEND_PRD; every `AdhError` subclass has `.code`. |

---

### Test file map (summary)

| File | Steps covered |
|------|----------------|
| `tests/test_test_mode.py` | 0 |
| `tests/test_errors.py` | 0b |
| `tests/test_service.py` | 1 |
| `tests/test_import_boundary.py` | 1 |
| `tests/test_registry.py` | 2 |
| `tests/test_config.py` | 3 |
| `tests/test_doctor.py` | 4 |
| `tests/test_idempotency.py` | 5 |
| `tests/test_api.py` (extend) | 0b, 5 |
| `tests/test_schemas.py` (extend) | 5 |
| `tests/test_jobs.py` | 6 |
| `tests/test_docker.py` | 7 (slow, optional CI job) |
| `tests/test_mcp.py` | 8 |
| `tests/test_sdk.py` | 9 |
| `tests/test_contract_parity.py` | ongoing |
| `tests/test_setup_docs.py` (extend) | docs stay synced |

**CI recommendation:** Run all except `test_docker*` and `@pytest.mark.slow` on every push; nightly job for Docker + model downloads.

---

## 16. References

- [Soup CLI](https://trysoup.dev/) — config file, profiles, MCP, backend-first product shape
- [Stripe API design](https://stripe.com/docs/api) — idempotency, prefixed IDs, actionable errors, metadata, API review discipline
- [Ollama](https://ollama.com/) — one-command install/run, familiar API shapes, local-first sandbox
- [Pangram API](https://docs.pangram.com/api-reference/introduction) — async task polling pattern
- Research basis in root [README.md](../README.md)
