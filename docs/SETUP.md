# Setup

This is the setup guide for the open-core CLI, library, and local HTTP API.

Python **3.11** is the supported version for this repository. Python 3.12 usually works; 3.10 does not.

GPTZero and Pangram are **verification detectors**. Set their keys in `.env`, then score with `--detector pangram` or `--detector gptzero`. They cannot drive the `humanize` inner loop (too slow and billed per call).

## What you need

| You want to… | Install extra | Also needed |
|---|---|---|
| Run tests / develop | `[dev]` | nothing else |
| Score text with the fake detector | core or `[dev]` | nothing else |
| Humanize text | core or `[dev]` | `.env` rewriter key **or** a local OpenAI-compatible server |
| Score / humanize with a real local detector | `[local]` | `adh models fetch` |
| Use MiniLM as the semantic gate | `[local]` | first MiniLM download happens on use |
| Serve `POST /v1/humanize` | `[api]` or `[dev]` | rewriter key for humanize routes |

`[pangram]` is listed in `pyproject.toml` for a later phase. Do not install it now.

## 1. Clone and virtualenv

```bash
git clone https://github.com/faheemgurkani/adversarial-detector-humanizer.git
cd adversarial-detector-humanizer

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows (Command Prompt):

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Confirm the interpreter:

```bash
python -V
```

You should see `Python 3.11.x`.

## 2. Install the package

Developer / tests / local API (no neural models):

```bash
python -m pip install -e ".[dev]"
```

Local neural detectors and MiniLM (larger download: PyTorch, Transformers):

```bash
python -m pip install -e ".[local,dev]"
```

API only, without pytest/ruff:

```bash
python -m pip install -e ".[api]"
```

After install, `adh` is on your PATH inside the venv:

```bash
adh --help
```

## 3. Environment file

```bash
cp .env.example .env
```

Edit `.env` and set at least the rewriter values you will use. The CLI loads `.env` from the current working directory and parent directories. Existing shell variables win over `.env` (dotenv does not override them).

| Variable | Required for | Default | Notes |
|---|---|---|---|
| `OPENAI_API_KEY` | `adh humanize`, `POST /v1/humanize` | empty | Skip only if `OPENAI_BASE_URL` is localhost / `127.0.0.1` |
| `OPENAI_BASE_URL` | rewriter | `https://api.openai.com/v1` | Groq, OpenRouter, Ollama, vLLM, LM Studio |
| `ADH_REWRITER_MODEL` | rewriter | `gpt-4o-mini` | Must exist on that provider |
| `ADH_MODELS_DIR` | local detectors | `~/.cache/adversarial-detector-humanizer/models/` | Override cache location |
| `HF_TOKEN` | Hub downloads | unset | Optional; public Raschka exports do not need it |
| `PANGRAM_API_KEY` | `adh score --detector pangram` | unset | Post-loop verification; not for `humanize` loop |
| `GPTZERO_API_KEY` | `adh score --detector gptzero` | unset | Post-loop verification; not for `humanize` loop |

Provider snippets are commented in `.env.example`.

`adh score` with `--detector fake` does **not** need a rewriter key.

## 4. Download a local detector (optional)

Published weights live on the Hugging Face Hub under `rasbt/ai-text-detector-*`. They are cached under `ADH_MODELS_DIR` (or the default cache above).

```bash
adh models list
adh models fetch --model distilbert
```

Omit `--model` to fetch every published export. That is slow and large. For a first run, fetch `distilbert`. Quality default in the engine is `qwen3-variable` (heavier).

Then:

```bash
adh score --detector distilbert --file examples/sample.txt
```

If a named detector is not on disk, the CLI raises a not-ready error. Fetch it, or pass `--detector fake` while you are wiring the rewriter.

## 5. First commands

Score without models or API keys:

```bash
adh score --detector fake --text "Furthermore, it is important to note the result."
adh score --file examples/sample.txt --detector fake
```

Humanize (needs `.env` rewriter, and `--allow-lexical-gate` if `[local]` is not installed):

```bash
adh humanize --detector fake --semantic lexical --allow-lexical-gate \
  --text "Furthermore, it is important to note the result in 2024."
```

With a fetched local detector and MiniLM:

```bash
adh humanize --detector distilbert --file examples/sample.txt --json --output out.txt
```

Verify with commercial detectors after humanizing (requires API keys in `.env`):

```bash
adh score --detector pangram --file out.txt
adh score --detector gptzero --file out.txt
```

Statistical + local neural ensemble (CPU statistical signal, no extra download):

```bash
adh score --detector statistical --text "Furthermore, it is important to note the result."
adh humanize --detector ensemble-local --semantic lexical --allow-lexical-gate \
  --text "Furthermore, it is important to note the result in 2024."
```

Optional structural translation pre-pass on flagged paragraphs only (higher latency):

```bash
adh humanize --detector fake --prepass structural --prepass-lang fi \
  --semantic lexical --allow-lexical-gate --text "Your AI draft here."
```

Pipe input:

```bash
pbpaste | adh score --detector fake
```

## 6. Local HTTP API

```bash
adh serve --host 127.0.0.1 --port 8000 --detector fake --semantic lexical
```

Open `http://127.0.0.1:8000/docs`. Route contract: [BACKEND_PRD.md](BACKEND_PRD.md).

Humanize routes still need the rewriter env vars.

## 7. Tests

```bash
source .venv/bin/activate
python -m pytest
```

CI uses `FakeDetector` and a lexical gate. No GPU and no paid APIs.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `OPENAI_API_KEY is not set` | Copy `.env.example` → `.env`, set a key, or point `OPENAI_BASE_URL` at a local server |
| Detector not ready | `adh models fetch --model <name>` after installing `[local]` |
| MiniLM / semantic backend missing | `pip install -e ".[local]"` or pass `--semantic lexical --allow-lexical-gate` |
| `adh: command not found` | Activate `.venv` and reinstall with `pip install -e ".[dev]"` |
| Hugging Face download stalls | Set `HF_TOKEN` in `.env`, or retry `adh models fetch` |
| Torch install is huge | That is expected for `[local]`. Use `--detector fake` until you need real scores |

## Honest limits

Local scores are proxies. They are not Pangram, GPTZero, or Turnitin. See the root [README](../README.md).
