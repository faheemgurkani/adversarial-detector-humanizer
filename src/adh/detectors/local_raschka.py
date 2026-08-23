"""Load published Raschka detector exports and score text.

Inference is adapted from rasbt/ai-detector-from-scratch (Apache-2.0).
"""

from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

from adh.detectors.base import ScoreResult, probability_to_result, require_text
from adh.exceptions import DetectorNotReadyError, InputError
from adh.models import (
    CAUSAL_MODELS,
    DEFAULT_MODEL,
    ModelSpec,
    artifact_status,
    default_cache_dir,
    ensure_artifact_ready,
    model_registry,
)

_InternalSpec = namedtuple("InternalSpec", "name kind artifact_path description")


def _resolve_device(requested: str):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise DetectorNotReadyError("CUDA was requested but is not available")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise DetectorNotReadyError("MPS was requested but is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise InputError("device must be one of: auto, cpu, cuda, mps")
    return torch.device(requested)


def _load_metadata(artifact_path: Path) -> dict:
    metadata_path = artifact_path / "detector-config.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _validate_text(text: str) -> str:
    return require_text(text)


class _SklearnTextClassifier:
    def __init__(self, spec: _InternalSpec) -> None:
        import numpy as np
        from joblib import load

        self.spec = spec
        self.model = load(spec.artifact_path)
        classes = np.asarray(self.model.classes_)
        matches = np.flatnonzero(classes == 1)
        if matches.size != 1:
            raise DetectorNotReadyError("The saved classifier must contain AI class 1")
        self.ai_column = int(matches[0])

    def score_many(self, texts: list[str], *, batch_size: int = 8) -> list[float]:
        del batch_size
        validated = [_validate_text(text) for text in texts]
        if not validated:
            return []
        probabilities = self.model.predict_proba(validated)
        return [float(probability) for probability in probabilities[:, self.ai_column]]


class _EncoderTextClassifier:
    def __init__(self, spec: _InternalSpec, *, device: str = "auto") -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.spec = spec
        self.metadata = _load_metadata(spec.artifact_path)
        self.device = _resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.artifact_path)
        if spec.kind == "peft":
            from peft import AutoPeftModelForSequenceClassification

            self.model = AutoPeftModelForSequenceClassification.from_pretrained(
                spec.artifact_path
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                spec.artifact_path
            )
        self.model.to(self.device)
        self.model.eval()
        self.max_length = int(self.metadata["max_length"])
        self.temperature = float(self.metadata.get("temperature", 1.0))
        self.ai_index = int(self.metadata.get("label_mapping", {}).get("ai", 1))
        if self.temperature <= 0:
            raise DetectorNotReadyError("Saved temperature must be positive")

    def score_many(self, texts: list[str], *, batch_size: int = 8) -> list[float]:
        import torch

        validated = [_validate_text(text) for text in texts]
        if batch_size < 1:
            raise InputError("batch_size must be at least 1")
        probabilities: list[float] = []
        for start in range(0, len(validated), batch_size):
            batch = validated[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                batch_probabilities = (
                    (logits / self.temperature)
                    .softmax(dim=1)[:, self.ai_index]
                    .float()
                    .cpu()
                    .tolist()
                )
            probabilities.extend(batch_probabilities)
        return probabilities


def _build_causal_batch(
    encoded_input_ids,
    *,
    pad_token_id,
    eos_token_id,
    context_length,
    readout_position,
):
    if readout_position not in {"variable", "fixed"}:
        raise DetectorNotReadyError("readout_position must be 'variable' or 'fixed'")
    if context_length < 1:
        raise DetectorNotReadyError("context length must be positive")
    if any(len(input_ids) + 1 > context_length for input_ids in encoded_input_ids):
        raise DetectorNotReadyError("encoded input exceeds the configured context length")
    if not encoded_input_ids:
        return [], []

    input_rows = []
    mask_rows = []
    if readout_position == "variable":
        batch_length = max(len(input_ids) + 1 for input_ids in encoded_input_ids)
        for input_ids in encoded_input_ids:
            padding_length = batch_length - len(input_ids) - 1
            input_rows.append(list(input_ids) + [eos_token_id] + [pad_token_id] * padding_length)
            mask_rows.append([1] * (len(input_ids) + 1) + [0] * padding_length)
    else:
        for input_ids in encoded_input_ids:
            padding_length = context_length - len(input_ids) - 1
            input_rows.append(
                list(input_ids) + [pad_token_id] * padding_length + [eos_token_id]
            )
            mask_rows.append([1] * len(input_ids) + [0] * padding_length + [1])
    return input_rows, mask_rows


class _CausalTextClassifier:
    def __init__(self, spec: _InternalSpec, *, device: str = "auto") -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.spec = spec
        self.metadata = _load_metadata(spec.artifact_path)
        self.device = _resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.artifact_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(spec.artifact_path)
        self.model.to(self.device)
        self.model.eval()
        self.max_text_length = int(self.metadata["max_text_length"])
        self.context_length = int(self.metadata["context_length"])
        self.readout_position = str(self.metadata["readout_position"])
        self.temperature = float(self.metadata.get("temperature", 1.0))
        self.ai_index = int(self.metadata.get("label_mapping", {}).get("ai", 1))
        if self.tokenizer.pad_token_id is None:
            raise DetectorNotReadyError("Saved tokenizer does not define a padding token")
        if self.tokenizer.eos_token_id is None:
            raise DetectorNotReadyError("Saved tokenizer does not define an EOS token")
        if self.temperature <= 0:
            raise DetectorNotReadyError("Saved temperature must be positive")

    def _prepare_batch(self, texts: list[str]):
        import torch

        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_text_length,
        )
        input_rows, mask_rows = _build_causal_batch(
            encoded["input_ids"],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            context_length=self.context_length,
            readout_position=self.readout_position,
        )
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        }

    def score_many(self, texts: list[str], *, batch_size: int = 1) -> list[float]:
        import torch

        validated = [_validate_text(text) for text in texts]
        if batch_size < 1:
            raise InputError("batch_size must be at least 1")
        probabilities: list[float] = []
        for start in range(0, len(validated), batch_size):
            batch = validated[start : start + batch_size]
            encoded = {
                key: value.to(self.device)
                for key, value in self._prepare_batch(batch).items()
            }
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                batch_probabilities = (
                    (logits / self.temperature)
                    .softmax(dim=1)[:, self.ai_index]
                    .float()
                    .cpu()
                    .tolist()
                )
            probabilities.extend(batch_probabilities)
        return probabilities


def _load_backend(spec: ModelSpec, *, device: str):
    ensure_artifact_ready(spec)
    internal = _InternalSpec(
        name=spec.name,
        kind=spec.kind,
        artifact_path=spec.artifact_path,
        description=spec.description,
    )
    if spec.kind == "sklearn":
        return _SklearnTextClassifier(internal)
    if spec.kind in {"encoder", "peft"}:
        return _EncoderTextClassifier(internal, device=device)
    return _CausalTextClassifier(internal, device=device)


class LocalRaschkaDetector:
    """Published Raschka classifier used as a local loop verifier."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        models_dir: Path | str | None = None,
        device: str = "auto",
    ) -> None:
        if model_name not in model_registry():
            available = ", ".join(sorted(model_registry()))
            raise InputError(f"Unknown model {model_name!r}. Available: {available}")
        root = Path(models_dir).expanduser().resolve() if models_dir else default_cache_dir()
        self.spec = model_registry(root)[model_name]
        self.name = model_name
        self.device = device
        self._backend = None

    def _classifier(self):
        if self._backend is None:
            self._backend = _load_backend(self.spec, device=self.device)
        return self._backend

    def ready(self) -> bool:
        ready, _ = artifact_status(self.spec)
        return ready

    def score(self, text: str) -> ScoreResult:
        require_text(text)
        probability = self._classifier().score_many([text], batch_size=1)[0]
        return probability_to_result(probability)

    def score_spans(self, texts: list[str]) -> list[ScoreResult]:
        if not texts:
            return []
        batch_size = 1 if self.spec.name in CAUSAL_MODELS else 8
        probabilities = self._classifier().score_many(texts, batch_size=batch_size)
        return [probability_to_result(probability) for probability in probabilities]
