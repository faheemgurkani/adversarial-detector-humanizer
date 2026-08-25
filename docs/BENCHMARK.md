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
| `detector_breakdown.transfer_ok` | Guidance vs deploy transfer (Plan 09) |
| `all_candidates_rejected` rate | Gates too strict? |
| `passes_all` in verify block | External ground when keys available |

## Last smoke (fake detector)

Run `python scripts/benchmark.py humanize --detector fake` and paste results here after local runs.
