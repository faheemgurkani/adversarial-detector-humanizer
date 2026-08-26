# Benchmark harness

Reproducible smoke benchmarks for **adversarial-detector-humanizer**.

## Run locally

```bash
python scripts/benchmark.py score --detector fake
python scripts/benchmark.py humanize --detector fake
```

With local Raschka weights:

```bash
adh models fetch --model qwen3-variable
python scripts/benchmark.py score --detector qwen3-variable
```

## Corpus

See `benchmarks/samples.jsonl` (~5 hand-written rows). Extend with your own licensed text; do not commit third-party corpora without permission.

## Ceiling honesty

- Inner-loop scores use the **local proxy** detector (default `qwen3-variable`).
- Local score drops do **not** guarantee Pangram/GPTZero pass.
- Publish negative transfer results when the proxy anti-correlates with commercial detectors.
- Use post-loop verify: `adh humanize --verify pangram --verify-threshold 45`.

## Metrics to track

| Metric | Why |
|--------|-----|
| Human text FPR @ `verdict_score` | Calibrate reporting threshold |
| Δ score after humanize | Primary product metric |
| `detector_breakdown.transfer_ok` | Guidance vs deploy transfer |
| `all_candidates_rejected` rate | Gates too strict? |
| `passes_all` in verify block | External ground when keys available |

## Statistical detector (`--detector statistical`)

CPU-only heuristic from humanize-text (TTR, sentence-length CV, hapax ratio). It is a **weak proxy** with possible human FPR if used alone. Prefer the max ensemble preset:

```bash
adh score --detector ensemble-local --text "Your paragraph here."
adh humanize --detector ensemble-local --file examples/sample.txt
```

`ensemble-local` = `qwen3-variable` + `statistical` with **max** aggregation, forcing the loop to optimize rhythm diversity as well as classifier score. Single-sentence span scores from `statistical` are neutral (50.0) because burstiness needs ≥2 sentences.

## Last smoke (fake detector)

Run `python scripts/benchmark.py humanize --detector fake` and paste results here after local runs.
