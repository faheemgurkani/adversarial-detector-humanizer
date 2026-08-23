# Product PRD (later phases)

The open-core engine in this repository is the source of truth. Hosted surfaces must call `adh.engine.humanize` and return a `RunReport`. Do not reimplement the loop in JavaScript.

## Positioning

Writers polish drafts (their own, or AI-assisted) and want fewer false “AI” flags without turning the piece into a synonym stew. Marketing line:

> Verified score reduction — we show before/after, no bypass guarantees.

## Surfaces

### Web SaaS

- FastAPI around `humanize()` and `score()`
- Paste UI, before/after scores, sentence diff
- Auth plus Stripe
- Free: a few local-detector runs per day
- Pro (~$15–20/mo): 50 documents, optional Pangram verification

### Paid detectors

After the **local** loop converges, call Pangram 4 **once** at the start and once at the end (`pangram-sdk` `Pangram().predict(text, model="pangram-4")`, `$0.05 / 100` words realtime). Never put Pangram in the inner sentence loop.

Pangram 4 windows include `is_humanized` and `humanizer_score`. A later pass may flag windows that are AI **or** humanized.

Optional GPTZero `POST https://api.gptzero.me/v2/predict/text` for per-sentence `generated_prob`.

### REST API

`POST /v1/humanize` → `RunReport`  
`POST /v1/score` → `{score, label}`  
`report.to_public_dict()` is the compact body: `ai_score_before`, `ai_score_after`, `semantic_score`.

Meter by words. Planned API plan ~$49/mo.

### Chrome extension

Highlight sentences from `sentences[]` in Google Docs or Substack and rewrite in place through the hosted API. Planned add-on ~$15/mo. Docs iframe APIs are the hard part.

## Phase 6 (optional)

GRPO-train a small rewriter (Qwen3-0.6B) with local detector reward + semantic similarity + KL. Only after the inference loop is stable. Raschka’s RLVR run reward-hacked without those constraints.

## Non-goals for MVP

Auth, billing, Chrome, Pangram keys in the inner loop, GRPO training, telemetry.
