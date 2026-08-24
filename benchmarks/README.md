# Benchmark corpus

Small redistributable samples for local score/humanize smoke runs.

Provenance:
- Hand-written AI-ish and human-ish sentences in this repo.
- Not shipped: full Raschka `human-vs-ai-50k` text (cite dataset IDs only if extending).

Run:

```bash
python scripts/benchmark.py score --detector fake
python scripts/benchmark.py humanize --detector fake
```
