# Architecture

```
input
  → split sentences (pysbd, offsets kept)
  → detector.score (whole text) and detector.score_spans (per sentence)
  → flag sentences over threshold (or top-k)
  → preserve-lock facts
  → rewriter.register-shift (OpenAI-compatible, best-of-N)
  → restore locks (reject if sentinels missing)
  → semantic gate (MiniLM or lexical)
  → rescore
  → repeat until target, max rounds, or no valid candidate
  → RunReport
```

## Detector protocol

`Detector.score(text) -> ScoreResult`  
`Detector.score_spans(texts) -> list[ScoreResult]`

Adapters:

- `LocalRaschkaDetector` — published HF exports from rasbt
- `FakeDetector` / `CueDetector` (tests)
- `PangramDetector`, `GPTZeroDetector` — raise `RemoteDetectorUnavailableError` in open-core
- `EnsembleDetector` — weighted blend of ready members

Default quality model: `qwen3-variable`. Faster: `distilbert`. CI: `fake` or `logreg` when present.

## Quality gates

- Preserve-lock sentinels `__LOCK_<token>_<nnn>__`
- Semantic cosine ≥ `min_semantic_similarity` (default 0.88)
- `max_rewrite_ratio` caps how much of the document can be rewritten in a round

## Stop reasons

`passed`, `max_rounds`, `no_flagged_sentences`, `all_candidates_rejected`, `max_rewrite_ratio`, `already_below_target`

The engine always returns a report, including the best intermediate text.

## Environment

The CLI loads `.env` from the working directory via python-dotenv. Template: [`.env.example`](../.env.example). Setup: [SETUP.md](SETUP.md).

Used today:

- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ADH_REWRITER_MODEL` — OpenAI-compatible rewriter
- `ADH_MODELS_DIR` — local Raschka artifact cache

Reserved, unused until remote detectors ship:

- `PANGRAM_API_KEY`, `GPTZERO_API_KEY`

## HTTP

See [BACKEND_PRD.md](BACKEND_PRD.md). Routes: `GET /health`, `GET /v1/models`, `POST /v1/score`, `POST /v1/humanize`, `POST /v1/sentences`.
