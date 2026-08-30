# adversarial-detector-humanizer

**Detector-verified, sentence-targeted, meaning-preserving text humanizer** — built as an open-core **backend engine** with a production-minded product shell.

This is not a paraphrasing API bolted onto a script. It is a **layered backend system**: a pure rewrite loop at the center, a shared service layer, plugin registries, unified configuration, and thin transport adapters (CLI, HTTP, jobs). Humans, agents, CI, and future MCP clients all hit the same execution path.

> **Verified score reduction** — we show before/after scores. No bypass guarantees.

| Doc | Purpose |
|-----|---------|
| [docs/ROADMAP.md](docs/ROADMAP.md) | Transformation plan and build order |
| [docs/BACKEND_PRD.md](docs/BACKEND_PRD.md) | Frozen `/v1` HTTP contract |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Engine loop, modules, boundaries |
| [docs/SETUP.md](docs/SETUP.md) | Install, env, profiles, doctor |
| [CONTRIBUTING.md](CONTRIBUTING.md) | API review gate and workflow |

**Try in 30 seconds** (no API key, no model download):

```bash
python -m pip install -e ".[dev]"
adh try
```

---

## Backend system design

The product follows a single rule: **one library, many doors**. Business logic lives once; transports parse input and format output.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Doors (thin adapters)                                               │
│  Typer CLI · FastAPI sync · FastAPI async jobs · Python import         │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  AdhConfig / HumanizeRequest in
                                │  RunReport / compact JSON out
┌───────────────────────────────▼──────────────────────────────────────┐
│  Product shell                                                       │
│  config.py   adh.yaml + profiles + override precedence                 │
│  service.py  run_humanize() / run_score() — single use-case layer    │
│  doctor.py   config-aware pre-flight checks                          │
│  jobs/       async queue + worker → same service                     │
│  idempotency.py · ids.py · errors.py  Stripe-style HTTP discipline   │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  Engine (pure)                                                       │
│  engine.humanize() · gates · preserve-lock · report types            │
│  No FastAPI, Typer, or env reads in the core loop                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  protocols
┌───────────────────────────────▼──────────────────────────────────────┐
│  Plugins (importlib.metadata entry_points)                           │
│  adh.detectors · adh.rewriters · adh.gates                             │
└──────────────────────────────────────────────────────────────────────┘
```

### Design decisions (why it is built this way)

| Concern | Approach |
|---------|----------|
| **Config drift** | One shape everywhere: `adh.yaml`, CLI flags, HTTP JSON, profiles — merged via `resolve_adh_config()` with explicit override precedence |
| **Surface drift** | CLI and HTTP call `service.run_humanize()`, not duplicate wiring |
| **Plugin extensibility** | Detectors/rewriters/gates register via `pyproject.toml` entry points; `registry.load_plugin()` constructs them |
| **Agent integrations** | Compact-by-default HTTP responses, `agent_hint`, `metadata`, prefixed IDs (`report_`, `job_`, `req_`) |
| **Retries** | `Idempotency-Key` on sync humanize and async job create — same body → same result, no double spend |
| **Long runs** | `POST /v1/jobs/humanize` → 202 + poll `GET /v1/jobs/{id}` — worker calls the same service as sync |
| **Operability** | `adh doctor` reads your config and returns actionable fixes before production profiles |
| **Errors** | Structured `{ error: { code, message, retryable, doc_url, request_id } }` — not bare strings |

Local detector scores are **proxies**. They correlate with commercial tools; they are not Pangram, GPTZero, or Turnitin verdicts.

---

## What the engine does

```
Score → flag sentences → preserve-lock facts → register-shift rewrite
      → restore locks → meaning gates → re-score → repeat
```

- **Detector-guided loop** — rewrite flagged sentences only, not the whole document
- **Register/style shift** — rhythm and transitions, not synonym maps
- **Preserve-lock** — numbers, URLs, emails, DOIs, quotes, code, acronyms, names
- **Meaning gate stack** — semantic similarity + mechanical vetoes (numerals, hedges, deletion)
- **Local Raschka detectors** — free/dev inner-loop verifiers (`qwen3-variable`, etc.)
- **Remote adapters** — Pangram / GPTZero for post-loop verify and `adh score` only

---

## Configuration

Project settings live in **`adh.yaml`** (or `ADH_CONFIG`). Profiles bundle sensible defaults:

| Profile | Use case |
|---------|----------|
| `fast` | Zero-key test mode (fake detector, identity rewriter) |
| `standard` | Default local production (OpenAI rewriter + local detector) |
| `quality` | `ensemble-local` detector, higher rounds |
| `verify-only` | Score/verify path, identity rewriter |

```bash
adh init
adh doctor                    # pre-flight: keys, models, torch
adh humanize --file draft.txt --json
```

CLI flags and HTTP fields **override** file values for that request only. See [docs/SETUP.md](docs/SETUP.md).

---

## CLI

```bash
adh --help
adh try                                          # zero-key smoke test
adh init && adh doctor                           # config + pre-flight
adh score --detector fake --text "Sample text."
adh humanize --profile fast --text "..." --json
adh humanize --async --profile fast --text "..." # job queue (in-process)
adh serve --host 127.0.0.1 --port 8000
```

Pipe or file input:

```bash
adh score --file examples/sample.txt --detector fake
pbpaste | adh score --detector fake
```

Production humanize requires an OpenAI-compatible rewriter (`OPENAI_API_KEY` or local `OPENAI_BASE_URL`). There is **no regex humanizer fallback**.

---

## HTTP API

Install `[api]` or `[dev]`, then:

```bash
adh serve --host 127.0.0.1 --port 8000 --detector fake --semantic lexical
```

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/v1/models` | Published detector artifacts |
| POST | `/v1/score` | Document score (0–100) |
| POST | `/v1/humanize` | Sync humanize (compact by default) |
| POST | `/v1/jobs/humanize` | Async humanize → **202** + `job_id` |
| GET | `/v1/jobs/{job_id}` | Poll job status (**always 200**) |
| POST | `/v1/sentences` | Offset-preserving split |

**Sync example** (minimal agent payload):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/humanize \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: idem-abc-123' \
  -d '{"text": "Furthermore, note this.", "profile": "fast"}'
```

**Async example**:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/jobs/humanize \
  -H 'Content-Type: application/json' \
  -d '{"text": "Long document...", "profile": "fast"}'
# → 202, Location: /v1/jobs/job_…

curl -s http://127.0.0.1:8000/v1/jobs/job_…
# → {"status":"done","report_id":"report_…","report":{…}}
```

Every response includes `X-Request-Id: req_…`. Full contract, error codes, idempotency semantics: **[docs/BACKEND_PRD.md](docs/BACKEND_PRD.md)**. Interactive docs: `http://127.0.0.1:8000/docs`.

---

## Library and service layer

Prefer the **service layer** over wiring the engine directly — it matches CLI and HTTP behavior (config merge, adapter loading, `report_id` assignment).

```python
from adh.service import run_humanize
from adh.config import resolve_adh_config

report = run_humanize(
    "Furthermore, note this.",
    config=resolve_adh_config(profile="fast"),
)
print(report.report_id, report.stop_reason, report.output_text)
```

Lower-level engine access remains available for tests and custom integrations:

```python
from adh.engine import EngineConfig, humanize
from adh.factory import load_detector, load_rewriter
from adh.gates import build_meaning_gate_stack

report = humanize(
    "Furthermore, the method is important to note in 2024.",
    detector=load_detector("fake"),
    rewriter=load_rewriter(name="identity"),
    meaning_gate_stack=build_meaning_gate_stack(prefer="lexical", allow_lexical=True),
    config=EngineConfig(min_semantic_similarity=0.2),
)
```

---

## Setup

Python **3.11+**. Step-by-step install, extras, model fetch, troubleshooting: **[docs/SETUP.md](docs/SETUP.md)**.

```bash
git clone https://github.com/faheemgurkani/adversarial-detector-humanizer.git
cd adversarial-detector-humanizer
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
adh init && adh doctor
```

Local neural detectors: `pip install -e ".[local,dev]"` then `adh models fetch --model qwen3-variable`.

---

## Tests

```bash
python -m pytest
```

~200 tests. CI uses `FakeDetector` and lexical gates — no GPU, no paid APIs. Coverage includes config merge, registry, service parity (CLI ≡ API), idempotency, async jobs, and doctor pre-flight.

---

## Project layout

```
src/adh/
  engine.py          closed-loop humanize (pure)
  service.py         run_humanize / run_score (shared use-case layer)
  config.py          adh.yaml, profiles, resolve_adh_config()
  registry.py        entry_points plugin loader
  factory.py         detector / rewriter / gate construction
  api.py             FastAPI: sync + async jobs
  cli.py             Typer commands
  doctor.py          pre-flight checks
  jobs/              store, worker, runner (async humanize)
  idempotency.py     Idempotency-Key cache (24h TTL)
  ids.py             report_ / job_ / req_ ID generation
  errors.py          structured error envelope + catalog
  detectors/         Raschka local, fake, statistical, remote, ensemble
  gates/             meaning gate stack
  schemas.py         HTTP request/response models (contract)
tests/               unit + integration per module
docs/
  ROADMAP.md         transformation plan (Steps 0–9)
  BACKEND_PRD.md     frozen /v1 contract
  ARCHITECTURE.md    engine + boundaries
  SETUP.md           install and operations
examples/adh.yaml    committed config template
```

---

## Research basis

This product applies published techniques. It is not a paper.

1. [Adversarial Paraphrasing (2025)](https://arxiv.org/abs/2506.07001) — detector-guided rewrite
2. [Raschka, *Building an AI Text Detector From Scratch*](https://magazine.sebastianraschka.com/p/ai-detector-from-scratch) — detector-in-the-loop
3. [untell](https://github.com/ssamba1/untell) — closed loop + citation locking
4. [AuthorMist](https://arxiv.org/abs/2503.08716) / [StealthRL](https://arxiv.org/abs/2602.08934) — detector-as-reward (future GRPO stage)
5. [APT-Eval (2025)](https://aclanthology.org/2025.findings-acl.1303.pdf) — polished human writing mis-flagged

---

## Honest limits

- Do not use this to cheat on academic integrity checks.
- Do not treat local scores as commercial detector verdicts.
- Pangram/GPTZero have their own terms — follow them.
- Do not claim 100% detector bypass in marketing or docs.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution to `rasbt/ai-detector-from-scratch`.
