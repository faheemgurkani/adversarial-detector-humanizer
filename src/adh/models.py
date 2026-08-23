"""Registry and download helpers for published Raschka detector exports."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from adh.exceptions import DetectorNotReadyError, InputError

HUB_MODEL_REPOSITORIES: dict[str, str] = {
    "logreg": "rasbt/ai-text-detector-logreg",
    "distilbert": "rasbt/ai-text-detector-distilbert",
    "distilbert-lora": "rasbt/ai-text-detector-distilbert-lora",
    "distilbert-mica": "rasbt/ai-text-detector-distilbert-mica",
    "modernbert": "rasbt/ai-text-detector-modernbert",
    "gpt2-variable": "rasbt/ai-text-detector-gpt2-variable",
    "gpt2-fixed": "rasbt/ai-text-detector-gpt2-fixed",
    "qwen3-variable": "rasbt/ai-text-detector-qwen3-0.6b-variable",
    "qwen3-fixed": "rasbt/ai-text-detector-qwen3-0.6b-fixed",
}

DEFAULT_MODEL = "qwen3-variable"
ENCODER_MODELS = frozenset({"distilbert", "modernbert"})
PEFT_MODELS = frozenset({"distilbert-lora", "distilbert-mica"})
CAUSAL_MODELS = frozenset(
    {"gpt2-variable", "gpt2-fixed", "qwen3-variable", "qwen3-fixed"}
)
SKLEARN_MODELS = frozenset({"logreg"})

HUB_IGNORE_PATTERNS = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "figures/*",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    artifact_path: Path
    description: str
    hub_id: str


def default_cache_dir() -> Path:
    override = os.environ.get("ADH_MODELS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".cache" / "adversarial-detector-humanizer" / "models"


def model_kind(name: str) -> str:
    if name in SKLEARN_MODELS:
        return "sklearn"
    if name in PEFT_MODELS:
        return "peft"
    if name in ENCODER_MODELS:
        return "encoder"
    if name in CAUSAL_MODELS:
        return "causal"
    raise InputError(f"unknown detector model {name!r}")


def _artifact_path(name: str, models_dir: Path) -> Path:
    if name == "logreg":
        return models_dir / "logreg" / "logreg-ai-detector.joblib"
    return models_dir / name


def model_registry(models_dir: Path | None = None) -> dict[str, ModelSpec]:
    root = Path(models_dir).expanduser().resolve() if models_dir else default_cache_dir()
    descriptions = {
        "logreg": "TF-IDF logistic regression with Platt scaling",
        "distilbert": "Fully fine-tuned DistilBERT",
        "distilbert-lora": "DistilBERT with a LoRA adapter",
        "distilbert-mica": "DistilBERT with a MiCA adapter",
        "modernbert": "Fully fine-tuned ModernBERT-base",
        "gpt2-variable": "GPT-2 with a variable-position readout",
        "gpt2-fixed": "GPT-2 with a fixed-position readout",
        "qwen3-variable": "Qwen3-0.6B with a variable-position readout",
        "qwen3-fixed": "Qwen3-0.6B with a fixed-position readout",
    }
    return {
        name: ModelSpec(
            name=name,
            kind=model_kind(name),
            artifact_path=_artifact_path(name, root),
            description=descriptions[name],
            hub_id=repo,
        )
        for name, repo in HUB_MODEL_REPOSITORIES.items()
    }


def _has_model_weights(spec: ModelSpec) -> bool:
    if spec.kind == "sklearn":
        return spec.artifact_path.is_file()
    if not spec.artifact_path.is_dir():
        return False
    patterns = (
        ("adapter_model*.safetensors", "adapter_model*.bin")
        if spec.kind == "peft"
        else (
            "model*.safetensors",
            "pytorch_model*.bin",
            "model*.index.json",
            "pytorch_model*.index.json",
        )
    )
    return any(any(spec.artifact_path.glob(pattern)) for pattern in patterns)


def artifact_status(spec: ModelSpec) -> tuple[bool, str]:
    if spec.kind == "sklearn":
        if spec.artifact_path.is_file():
            return True, "ready"
        return False, "model file is missing"
    if not spec.artifact_path.is_dir():
        return False, "artifact directory is missing"
    if not (spec.artifact_path / "detector-config.json").is_file():
        return False, "detector-config.json is missing"
    if not _has_model_weights(spec):
        return False, "model weights are missing"
    return True, "ready"


def ensure_artifact_ready(spec: ModelSpec) -> None:
    ready, status = artifact_status(spec)
    if ready:
        return
    raise DetectorNotReadyError(
        f"{spec.name} is not ready: {status}. Expected export at "
        f"{spec.artifact_path}. Run `adh models fetch --model {spec.name}`."
    )


def list_models(models_dir: Path | None = None) -> list[dict[str, str]]:
    rows = []
    for spec in model_registry(models_dir).values():
        ready, status = artifact_status(spec)
        rows.append(
            {
                "name": spec.name,
                "kind": spec.kind,
                "hub": spec.hub_id,
                "status": status,
                "ready": "yes" if ready else "no",
                "path": str(spec.artifact_path),
                "description": spec.description,
            }
        )
    return rows


def fetch_models(
    names: Iterable[str] | None = None,
    *,
    models_dir: Path | None = None,
) -> list[ModelSpec]:
    registry = model_registry(models_dir)
    selected = list(names) if names else list(registry)
    unknown = [name for name in selected if name not in registry]
    if unknown:
        raise InputError(f"unknown models: {', '.join(unknown)}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise DetectorNotReadyError(
            "huggingface-hub is required to fetch models. "
            "Install extras: pip install 'adversarial-detector-humanizer[local]'"
        ) from error

    fetched: list[ModelSpec] = []
    for name in selected:
        spec = registry[name]
        spec.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=spec.hub_id,
            local_dir=str(
                spec.artifact_path.parent if spec.kind == "sklearn" else spec.artifact_path
            ),
            ignore_patterns=list(HUB_IGNORE_PATTERNS),
        )
        ensure_artifact_ready(spec)
        fetched.append(spec)
    return fetched
