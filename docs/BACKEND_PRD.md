# Backend PRD

Open-core HTTP and library contract for `adversarial-detector-humanizer`.

The ASGI app in `src/adh/api.py` is a thin wrapper around `adh.service.run_humanize()` — the same use-case layer as the CLI. A hosted SaaS must keep calling this service path. It must not reimplement the loop in another language.

**Scope:** sync `/v1/humanize`, async `/v1/jobs/*`, idempotency, prefixed IDs, structured errors, profiles, and metadata. **Planned:** MCP, SDK, Docker — [ROADMAP.md](ROADMAP.md). Additive fields are preferred over breaking changes; breaking changes require `/v2`.

Positioning: **verified score reduction**. Local detector scores are proxies. This API does not guarantee a Pangram, GPTZero, or Turnitin pass.

---

## Runtime

| Item | Value |
|------|--------|
| Process | `adh serve --host 127.0.0.1 --port 8000` |
| Default bind | `127.0.0.1:8000` |
| OpenAPI UI | `GET /docs` (FastAPI) |
| OpenAPI JSON | `GET /openapi.json` |
| Python | 3.11+ virtualenv `.venv` |
| Extra | `pip install -e ".[api]"` or `.[dev]` |

Injected fakes (tests): `create_app(detector=..., rewriter=..., semantic_gate=...)`.

Production loop detector: `qwen3-variable` after `adh models fetch`. Dev HTTP default for `adh serve` is `fake` so the process starts without weights.

---

## Error model

All domain errors are `AdhError` subclasses mapped to HTTP status codes and a structured envelope:

```json
{
  "error": {
    "code": "rewriter_unavailable",
    "message": "OPENAI_API_KEY is not set. ...",
    "retryable": false,
    "doc_url": "https://github.com/.../BACKEND_PRD.md#error-codes",
    "request_id": "req_a1b2c3d4e5f67890"
  }
}
```

Every HTTP response includes `X-Request-Id: req_…` (16 hex chars).

Pydantic validation failures return `422` with `error.code = "invalid_input"`.

### Error codes

| Code | HTTP | Retryable | When |
|------|------|-----------|------|
| `invalid_input` | 422 | no | Empty text, bad detector name, extra JSON fields, metadata > 50 keys |
| `unknown_detector` | 422 | no | Detector name not in registry |
| `preserve_lock_failed` | 422 | no | Lock restore failed |
| `remote_detector_unsupported` | 501 | no | `pangram` / `gptzero` used as inner-loop detector |
| `remote_detector_error` | 502 | yes | Pangram or GPTZero HTTP/API failure |
| `rewriter_unavailable` | 502 | no | Missing API key, upstream LLM HTTP error |
| `detector_not_ready` | 503 | no | Weights missing |
| `semantic_backend_unavailable` | 503 | no | MiniLM extra missing |
| `hard_mode_unavailable` | 503 | no | Hard mode extras missing |
| `idempotency_key_reused` | 409 | no | Same `Idempotency-Key` with a different body |
| `internal_error` | 400 | yes | Unmapped domain error |

CLI `--json` failures use the same `error.code` values.

---

## Changelog (v0.1 contract polish)

| Change | Notes |
|--------|--------|
| `compact` default `true` on `POST /v1/humanize` | Clients needing full `RunReport` must pass `"compact": false`. |
| `report_id` on every humanize response | Prefix `report_` + 16 hex chars. |
| Structured `error` object | Replaces bare FastAPI `{"detail": "..."}`. |
| `Idempotency-Key` header | Same key + same body within 24h returns cached response. |
| `agent_hint`, `output`, `metadata` on compact response | Additive fields for agents. |

### Semver policy

- `/v1/*` field names and `stop_reason` values are frozen until `/v2`.
- Additive response fields and optional request fields are allowed in minor releases.
- Breaking changes require a new major version path (`/v2`) and a documented migration.

---

## Endpoints

### `GET /health`

Liveness. No auth.

**200**

```json
{"status": "ok", "version": "0.1.0", "detector": "fake"}
```

Tested in `tests/test_api.py::test_health`.

---

### `GET /v1/models`

Lists published Raschka exports and whether local artifacts are ready.

**200**

```json
{
  "models": [
    {
      "name": "qwen3-variable",
      "kind": "causal",
      "hub": "rasbt/ai-text-detector-qwen3-0.6b-variable",
      "status": "artifact directory is missing",
      "ready": "no",
      "path": "...",
      "description": "Qwen3-0.6B with a variable-position readout"
    }
  ]
}
```

Tested in `tests/test_api.py::test_list_models`.

---

### `POST /v1/score`

Score one document. Does not rewrite.

**Request** (`ScoreRequest`)

| Field | Type | Default | Notes |
|-------|------|---------|--------|
| `text` | string | required | Non-empty after strip at detector layer |
| `detector` | string | `qwen3-variable` | `fake`, `logreg`, `distilbert`, `qwen3-variable`, `pangram`, … |
| `device` | string | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `models_dir` | string or null | null | Override cache / Raschka `models/` |

**200** (`ScoreResponse`)

```json
{"detector": "fake", "score": 80.0, "label": "ai-leaning", "windows": []}
```

`score` is 0–100. `label` is `human-leaning` (<30), `uncertain` (<70), `ai-leaning`.

**422** empty/whitespace text, unknown detector, extra fields.

Tested: `test_score_ok`, `test_score_empty_rejected`, `test_score_unknown_detector`, `test_extra_fields_rejected`.

---

### `POST /v1/humanize`

Closed loop: score → flag sentences → preserve-lock → register-shift rewrite → restore → semantic gate → rescore.

**Request** (`HumanizeRequest`)

| Field | Type | Default |
|-------|------|---------|
| `text` | string | required |
| `profile` | string or null | null (`fast` = zero-key test mode) |
| `detector` | string | `qwen3-variable` |
| `device` | string | `auto` |
| `models_dir` | string or null | null |
| `target_score` | float 0–100 | 30 |
| `max_rounds` | int 1–20 | 5 |
| `sentence_threshold` | float 0–100 | 50 |
| `min_semantic_similarity` | float 0–1 | 0.88 |
| `max_rewrite_ratio` | float 0–1 | 0.4 |
| `best_of_n` | int 1–8 | 3 |
| `rewriter_model` | string or null | env `ADH_REWRITER_MODEL` |
| `semantic` | `auto` / `minilm` / `lexical` | `auto` |
| `meaning_gate_mode` | string | `auto` |
| `allow_lexical_gate` | bool | false |
| `verify` | list of strings | `[]` |
| `deploy_detectors` | list of strings | `[]` |
| `hard_mode` | bool | false |
| `prepass` | string | `none` |
| `compact` | bool | **true** |
| `metadata` | object (string keys/values, max 50 pairs) | `{}` |

**Headers**

| Header | Required | Notes |
|--------|----------|--------|
| `Idempotency-Key` | no | Same key + same JSON body within 24h returns cached `report_id` and response |

**200 full** (`compact: false`, `RunReport`)

```json
{
  "input_text": "...",
  "output_text": "...",
  "detector": "cue",
  "score_before": 92.0,
  "score_after": 18.0,
  "semantic_similarity": 0.91,
  "rounds": 1,
  "stop_reason": "passed",
  "report_id": "report_a1b2c3d4e5f67890",
  "sentences": [
    {
      "i": 0,
      "original": "...",
      "rewritten": "...",
      "score_before": 92.0,
      "score_after": 18.0,
      "kept": true,
      "start": 0,
      "end": 40
    }
  ],
  "locks": [{"id": "001", "text": "2024", "ok": true}],
  "flagged_count": 1,
  "rewrite_ratio": 1.0
}
```

**200 compact** (default, `compact: true`)

```json
{
  "report_id": "report_a1b2c3d4e5f67890",
  "ai_score_before": 92.0,
  "ai_score_after": 18.0,
  "semantic_score": 0.91,
  "stop_reason": "passed",
  "detector": "cue",
  "output_text": "...",
  "output": "...",
  "agent_hint": "Local score reached target. Run verify if the user asked about Pangram or GPTZero.",
  "metadata": {"ticket": "JIRA-123"},
  "input": "a1b2c3d4e5f67890"
}
```

`output` equals `output_text`. `input` is a SHA-256 fingerprint (first 16 hex chars) of the source text.

**Stop reasons:** `passed`, `max_rounds`, `no_flagged_sentences`, `all_candidates_rejected`, `max_rewrite_ratio`, `already_below_target`.

### Idempotency

Send `Idempotency-Key` on `POST /v1/humanize`. The server hashes the JSON body; identical replays within 24 hours return the cached response without re-running the engine. Reusing the key with a different body returns `409 idempotency_key_reused`.

**501** if `detector` is `pangram` or `gptzero` (inner loop is local-only).

**502** if the rewriter backend is missing or fails.

Never put Pangram or GPTZero in the inner sentence loop. A later paid tier may call those APIs **once** after this endpoint returns.

Tested: `test_humanize_already_below_target`, `test_humanize_compact_and_loop`, `test_humanize_pangram_stub_is_not_found_as_loop_detector`, `test_scripted_humanize_preserves_url`.

---

### `POST /v1/jobs/humanize`

Async humanize using the same request body as sync humanize.

**202**

```json
{
  "job_id": "job_a1b2c3d4e5f67890",
  "status": "pending",
  "metadata": {"ticket": "JIRA-123"}
}
```

Header: `Location: /v1/jobs/{job_id}`. Supports `Idempotency-Key` on create.

### `GET /v1/jobs/{job_id}`

Poll job status. Known jobs always return **200** (`pending`, `processing`, `done`, `failed`).

When `status` is `done`, response includes `report_id` and nested compact `report` (same shape as sync compact humanize). When `failed`, includes structured `error`.

CLI: `adh humanize --async --profile fast --text "..."`.

Tested: `tests/test_jobs.py`.

---

### `POST /v1/sentences`

Offset-preserving sentence split (`pysbd`). Used by the engine and later Chrome highlighting.

**Request:** `{"text": "..."}`  
**200:** `{"sentences": [{"i": 0, "text": "...", "start": 0, "end": 12}, ...]}`  
**422:** empty text.

Tested: `test_sentences`, `test_sentences_empty`.

---

## Modules

| Module | Role | Tests |
|--------|------|--------|
| `adh.engine` | Loop, flagging, prepass, history, stop reasons | `test_engine.py` |
| `adh.preserve` | Fact/citation sentinels | `test_preserve.py` |
| `adh.gates` | Meaning gate stack | `test_meaning_gates.py` |
| `adh.scrub` | Unicode pre-loop scrub | `test_scrub.py` |
| `adh.tells` | AI-tells tie-break | `test_tells.py` |
| `adh.ranking` | Logprob + detector blend | `test_ranking.py` |
| `adh.prepass` | Structural translation pre-pass | `test_prepass.py` |
| `adh.hard` | Token-guided decode | `test_ap_features.py` |
| `adh.verify` | Post-loop commercial verify | `test_verify.py` |
| `adh.audit` | Detector breakdown | `test_audit.py` |
| `adh.sentences` | Split + reassemble with offsets | `test_sentences.py` |
| `adh.semantic` | MiniLM / lexical cosine gate | `test_semantic.py` |
| `adh.rewriter` | OpenAI-compatible register-shift | `test_rewriter.py`, `test_rewrite_history.py` |
| `adh.detectors.*` | Protocol, Raschka, fake, statistical, remote, ensemble | `test_detectors.py`, `test_statistical_detector.py` |
| `adh.models` | Hub registry, fetch, cache dir | `test_models.py` |
| `adh.report` | `RunReport` / compact dict | `test_report.py` |
| `adh.schemas` | HTTP Pydantic contracts | used by `test_api.py` |
| `adh.api` | FastAPI routes | `test_api.py` |
| `adh.cli` | `score`, `humanize`, `models`, `serve` | `test_cli.py` |
| `adh.factory` | Adapter construction | `test_detectors.py` |

---

## Features that must stay true

1. Rewrite **flagged sentences only**, not the whole document.
2. **Preserve-lock** numbers, URLs, emails, DOIs, ISBNs, quotes, code, acronyms, proper nouns; reject dropped sentinels.
3. **Semantic gate** rejects meaning flips.
4. **No regex humanizer fallback** when `OPENAI_API_KEY` is missing.
5. Remote commercial detectors are **stubs** in open-core (`501`).
6. User text is not stored. No telemetry.
7. Marketing must not claim a 100% detector bypass.

---

## Auth, metering, async jobs (not in this process)

Documented in [ROADMAP.md](ROADMAP.md). Not implemented in the open-core server.

- API keys and Stripe (optional hosted tier)
- Word metering
- Async `POST /v1/jobs/humanize` + polling
- One Pangram 4 `predict()` at start and end when verify is enabled
- Optional GPTZero `POST https://api.gptzero.me/v2/predict/text`

---

## CLI equivalents

| CLI | HTTP |
|-----|------|
| `adh score` | `POST /v1/score` |
| `adh humanize` | `POST /v1/humanize` |
| `adh humanize --async` | `POST /v1/jobs/humanize` + poll `GET /v1/jobs/{job_id}` |
| `adh models list` | `GET /v1/models` |
| `adh serve` | process hosting the routes above |

---

## Test command

```bash
source .venv/bin/activate
python -m pytest
```
