# Backend PRD

Open-core HTTP and library contract for `adversarial-detector-humanizer`.

The ASGI app in `src/adh/api.py` is a thin wrapper around `adh.engine.humanize`. A later hosted SaaS must keep calling this engine. It must not reimplement the loop in another language.

**Planned product changes** (async jobs, config file, MCP, prefixed IDs, idempotency, Stripe/Ollama interface discipline): [ROADMAP.md](ROADMAP.md). This document describes the **current** `/v1` contract; additive fields are preferred over breaking changes.

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

All domain errors are `AdhError` subclasses mapped as:

| Exception | HTTP | When |
|-----------|------|------|
| `InputError`, `PreserveLockError` | 422 | Empty text, bad detector name, extra JSON fields (Pydantic), lock restore |
| `RemoteDetectorUnavailableError` | 501 | `pangram` / `gptzero` used as the inner-loop detector |
| `RemoteDetectorError` | 502 | Pangram or GPTZero HTTP/API failure |
| `RewriterError` | 502 | Missing API key, upstream LLM HTTP error, empty candidates |
| `DetectorNotReadyError`, `SemanticBackendError` | 503 | Weights missing, MiniLM extra missing |
| Unmapped `AdhError` | 400 | Fallback |

Response body: FastAPI `{"detail": "<message>"}`. Extra request fields are rejected (`extra="forbid"`).

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
| `compact` | bool | false |

**200 full** (`RunReport`)

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

**200 compact** (`compact: true`)

```json
{
  "ai_score_before": 92.0,
  "ai_score_after": 18.0,
  "semantic_score": 0.91,
  "stop_reason": "passed",
  "detector": "cue",
  "output_text": "..."
}
```

**Stop reasons:** `passed`, `max_rounds`, `no_flagged_sentences`, `all_candidates_rejected`, `max_rewrite_ratio`, `already_below_target`.

**501** if `detector` is `pangram` or `gptzero` (inner loop is local-only).

**502** if the rewriter backend is missing or fails.

Never put Pangram or GPTZero in the inner sentence loop. A later paid tier may call those APIs **once** after this endpoint returns.

Tested: `test_humanize_already_below_target`, `test_humanize_compact_and_loop`, `test_humanize_pangram_stub_is_not_found_as_loop_detector`, `test_scripted_humanize_preserves_url`.

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
| `adh models list` | `GET /v1/models` |
| `adh serve` | process hosting the routes above |

---

## Test command

```bash
source .venv/bin/activate
python -m pytest
```
