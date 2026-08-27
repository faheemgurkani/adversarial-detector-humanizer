# Architecture

Engine layout today and target package boundary. **Transformation plan:** [ROADMAP.md](ROADMAP.md).

## Loop (current)

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
  → RunReport
```

## Protocols

| Protocol | Methods | Implementations |
|----------|---------|-----------------|
| **Detector** | `score()`, `score_spans()` | Raschka local, fake, statistical, ensemble (max/mean), Pangram, GPTZero |
| **Rewriter** | `rewrite()`, `rewrite_candidates()` | OpenAI-compatible, identity (`fast` profile), test helpers |
| **SemanticGate** | similarity check | MiniLM, lexical |
| **Translator** | `translate()` | LLM, Google (optional), identity (tests) |

Plugins: `src/adh/registry.py` loads `adh.detectors`, `adh.rewriters`, and `adh.gates` via `importlib.metadata.entry_points`. `src/adh/factory.py` is a thin wrapper (policy such as `assert_inner_loop_detector` stays there).

Project config: `adh.yaml` (or `ADH_CONFIG`) is loaded by `src/adh/config.py` and shared by CLI, HTTP, and profiles.

## Meaning gates

`MeaningGateStack` (`gates/stack.py`) combines semantic similarity with mechanical vetoes: numerals, hedges, deletion, optional NLI entailment, optional role preservation, scaffolding checks.

Modes: `auto`, `minilm`, `lexical`, `full` via CLI `--meaning-gate`.

## Quality gates

- Preserve-lock sentinels `__LOCK_<token>_<nnn>__` — multiset equality enforced
- Semantic cosine ≥ `min_semantic_similarity` (default 0.88)
- `max_rewrite_ratio` caps rewritten words per round
- AI-tells tie-break when detector scores are close

## Detectors

- **Inner loop:** local Raschka exports, fake, statistical, `ensemble-local` (qwen3 + statistical, max)
- **Verify / score only:** Pangram, GPTZero — blocked as inner-loop drivers (latency + cost)

Default quality model: `qwen3-variable`. CI: `fake`. See `adh models list`.

## Stop reasons

`passed`, `max_rounds`, `no_flagged_sentences`, `all_candidates_rejected`, `max_rewrite_ratio`, `already_below_target`

The engine always returns a report with the best intermediate text.

## Surfaces (today)

| Surface | Module | Calls |
|---------|--------|-------|
| Library | `adh.engine.humanize` | Engine directly |
| Service | `adh.service` | factory + `engine.humanize` |
| Config | `adh.config` | `adh.yaml`, profiles, merge overrides |
| CLI | `adh.cli` | `service.run_humanize` / `run_score` |
| HTTP | `adh.api` | `service.run_humanize` / `run_score` |

CLI and HTTP are thin doors: they parse input, call the service, and format output. Test mode is `--profile fast` / `adh try` (fake detector, identity rewriter, lexical gates). Full yaml config comes later — see [ROADMAP.md §3](ROADMAP.md#3-target-architecture).

## Environment

CLI loads `.env` via python-dotenv. Template: [`.env.example`](../.env.example). Setup: [SETUP.md](SETUP.md).

- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ADH_REWRITER_MODEL` — rewriter
- `ADH_MODELS_DIR` — Raschka artifact cache
- `PANGRAM_API_KEY`, `GPTZERO_API_KEY` — verification scoring

## HTTP

Current routes: [BACKEND_PRD.md](BACKEND_PRD.md). Planned: async jobs, MCP — [ROADMAP.md](ROADMAP.md).
