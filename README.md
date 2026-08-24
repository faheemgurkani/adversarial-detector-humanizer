# adversarial-detector-humanizer

Detector-verified, sentence-targeted, meaning-preserving text humanizer.

This is an **open-core engine**, not a one-shot paraphraser. It scores text, rewrites only the sentences a detector flags, locks facts and citations, and rejects candidates that drift in meaning. It reports before/after scores. It does **not** guarantee that any commercial detector will call the result human.

> Verified score reduction — we show before/after, no bypass guarantees.

## What it does

```
Score → flag sentences → preserve-lock facts → register-shift rewrite
      → restore locks → semantic gate → re-score → repeat
```

- **Detector-guided loop** rather than a blind full rewrite
- **Register/style shift** (rhythm, transitions, hedging) rather than synonym maps
- **Preserve-lock** for numbers, URLs, emails, DOIs, quotes, code, acronyms, names
- **Semantic gate** so meaning cannot silently flip
- **Local Raschka detectors** as the free/dev verifier
- **Remote stubs** for Pangram and GPTZero so a later hosted tier can plug in

Local scores are proxies. They correlate with tools such as Pangram; they are not Pangram.

## Research basis

This product applies published techniques. It is not a paper.

1. [Adversarial Paraphrasing (2025)](https://arxiv.org/abs/2506.07001) — detector-guided rewrite, training-free
2. [Raschka, *Building an AI Text Detector From Scratch*](https://magazine.sebastianraschka.com/p/ai-detector-from-scratch) — agent loop with a detector API as feedback
3. [untell](https://github.com/ssamba1/untell) — closed loop plus citation/number locking
4. [AuthorMist](https://arxiv.org/abs/2503.08716) / [StealthRL](https://arxiv.org/abs/2602.08934) — detector-as-reward (optional later GRPO stage)
5. [APT-Eval (2025)](https://aclanthology.org/2025.findings-acl.1303.pdf) — polished human writing is often mis-flagged

## Honest limits

- Do not use this to cheat on academic integrity checks.
- Do not treat a local score as a Pangram, GPTZero, or Turnitin verdict.
- Pangram 4 also has a humanizer-detection head. Phrase-swapping tools are often flagged as *humanized AI*. This engine avoids those cheap tricks; it still cannot promise a pass.
- Detector APIs and model cards have their own terms. Follow them.

## Setup (Python 3.11)

Step-by-step extras, env vars, model fetch, API, and troubleshooting: **[docs/SETUP.md](docs/SETUP.md)**.

```bash
git clone https://github.com/faheemgurkani/adversarial-detector-humanizer.git
cd adversarial-detector-humanizer
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `ADH_REWRITER_MODEL`) before `adh humanize`. The CLI loads `.env` from the working directory. GPTZero and Pangram keys are reserved for later and are ignored today.

Local neural detectors and MiniLM need the `local` extra:

```bash
python -m pip install -e ".[local,dev]"
adh models fetch --model distilbert
```

Published weights live on the Hugging Face Hub under `rasbt/ai-text-detector-*` and are cached at `~/.cache/adversarial-detector-humanizer/models/` (override with `ADH_MODELS_DIR` or `--models-dir`).

## CLI

```bash
adh --help
adh models list
adh score --detector fake --text "Furthermore, it is important to note the result."
adh humanize --detector fake --semantic lexical --allow-lexical-gate \
  --text "Furthermore, it is important to note the result in 2024."
```

`humanize` requires an OpenAI-compatible rewriter (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `ADH_REWRITER_MODEL`). There is no regex humanizer fallback.

Pipe or file input:

```bash
adh score --file examples/sample.txt --detector fake
pbpaste | adh score --detector fake
```

JSON `RunReport`:

```bash
adh humanize --file examples/sample.txt --json --output out.txt
```

The report includes `score_before`, `score_after`, `semantic_similarity`, per-sentence diffs, and lock records. That object is the seed for `POST /v1/humanize`.

## HTTP API

Install `[api]` or `[dev]`, then:

```bash
adh serve --host 127.0.0.1 --port 8000 --detector fake --semantic lexical
```

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/v1/models` | Published detector artifacts |
| POST | `/v1/score` | 0–100 AI score |
| POST | `/v1/humanize` | Closed-loop rewrite + `RunReport` |
| POST | `/v1/sentences` | Offset-preserving split |

Interactive docs: `http://127.0.0.1:8000/docs`. Full contract: [docs/BACKEND_PRD.md](docs/BACKEND_PRD.md).

## Library

```python
from adh.engine import EngineConfig, humanize
from adh.detectors.fake import FakeDetector
from adh.rewriter import ScriptedRewriter
from adh.semantic import LexicalSemanticGate

report = humanize(
    "Furthermore, the method is important to note in 2024.",
    detector=FakeDetector(default_score=90),
    rewriter=ScriptedRewriter({
        "Furthermore, the method is important to note in 2024.": [
            "The method mattered in 2024."
        ]
    }),
    semantic_gate=LexicalSemanticGate(),
    config=EngineConfig(min_semantic_similarity=0.2),
)
print(report.to_public_dict())
```

## Tests

```bash
source .venv/bin/activate
python -m pytest
```

CI uses `FakeDetector` and a lexical gate. No GPU and no paid APIs are required.

## Project layout

```
src/adh/            engine, CLI, preserve-lock, semantic gate, detectors
tests/              unit tests for every module
docs/SETUP.md       clone, venv, extras, .env, first commands
docs/PRODUCT.md     later SaaS / API / extension PRD
docs/ARCHITECTURE.md
.env.example        rewriter and optional detector env template
examples/sample.txt
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution to `rasbt/ai-detector-from-scratch`.
