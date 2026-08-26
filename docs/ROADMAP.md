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
- [ ] `adh humanize --profile fast --detector fake --text "…" --json` works with no `.env` keys
- [ ] SETUP.md "Try in 30 seconds" section at top
- [ ] `adh try` (or documented one-liner) prints scores + stop_reason

**Step 0b — IDs and errors (Stripe-style)**
- [ ] `report_id` with `report_` prefix on every humanize response
- [ ] `X-Request-Id` / `req_` on all HTTP responses
- [ ] Error envelope: `code`, `message`, `retryable`, `doc_url`, `request_id`
- [ ] Error code catalog started in BACKEND_PRD

**Step 1 — Package boundary**
- [ ] `adh.engine`, `adh.report`, `adh.gates`, `adh.preserve` import no `fastapi`, `typer`, `httpx` (rewriter adapters excepted behind protocols)
- [ ] Single integration test: CLI and API produce identical `RunReport` for same config

**Step 2 — Registry**
- [ ] All built-in detectors/rewriters registered via entry points
- [ ] `load_detector("unknown")` lists available names from registry
- [ ] CLI cold start < 200ms on typical machine (no `pkg_resources`)

**Step 3 — Config**
- [ ] `adh init` writes `adh.yaml`
- [ ] `adh humanize` reads config; CLI flags override file
- [ ] `adh serve` loads same file from cwd or `ADH_CONFIG` path
- [ ] Four profiles documented and tested

**Step 4 — Doctor**
- [ ] Checks: Python version, optional `[local]` torch, rewriter key/URL, model artifacts, Pangram key if verify configured
- [ ] Exit 0 = ready; exit 1 = actionable fix list

**Step 5 — Sync API polish**
- [ ] `agent_hint` on compact response
- [ ] Optional `metadata` on request/response
- [ ] `Idempotency-Key` deduplicates humanize within TTL
- [ ] OpenAPI documents all `stop_reason` values + error codes
- [ ] API review checklist in CONTRIBUTING or ROADMAP §4a
- [ ] BACKEND_PRD updated; semver policy written

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

## 15. References

- [Soup CLI](https://trysoup.dev/) — config file, profiles, MCP, backend-first product shape
- [Stripe API design](https://stripe.com/docs/api) — idempotency, prefixed IDs, actionable errors, metadata, API review discipline
- [Ollama](https://ollama.com/) — one-command install/run, familiar API shapes, local-first sandbox
- [Pangram API](https://docs.pangram.com/api-reference/introduction) — async task polling pattern
- Research basis in root [README.md](../README.md)
