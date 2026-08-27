"""Config-aware pre-flight checks before humanize or CI integration."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from adh.config import AdhConfig
from adh.models import (
    DEFAULT_MODEL,
    HUB_MODEL_REPOSITORIES,
    artifact_status,
    default_cache_dir,
    model_registry,
)
from adh.registry import GROUP_DETECTORS, GROUP_REWRITERS, list_plugins

_NO_ARTIFACT_DETECTORS = frozenset(
    {
        "fake",
        "statistical",
        "ensemble",
        "ensemble-max",
        "pangram",
        "gptzero",
    }
)

_VERIFY_ENV_KEYS: dict[str, str] = {
    "pangram": "PANGRAM_API_KEY",
    "gptzero": "GPTZERO_API_KEY",
}

_SETUP_DOC = "docs/SETUP.md"


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
    fix: str | None = None
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("skipped", None)
        return payload


def local_models_for_detector(detector: str) -> list[str]:
    """Return Raschka model names whose artifacts are required for ``detector``."""
    if detector in _NO_ARTIFACT_DETECTORS:
        return []
    if detector == "ensemble-local":
        return [DEFAULT_MODEL]
    if detector in HUB_MODEL_REPOSITORIES:
        return [detector]
    return []


def needs_local_stack(detector: str) -> bool:
    return bool(local_models_for_detector(detector))


def all_passed(results: list[CheckResult]) -> bool:
    return all(item.ok for item in results)


def _check_python_version() -> CheckResult:
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info.micro}"
    if (major, minor) >= (3, 11):
        return CheckResult(
            name="python_version",
            ok=True,
            message=version,
        )
    return CheckResult(
        name="python_version",
        ok=False,
        message=f"{version} (requires >= 3.11)",
        fix="Install Python 3.11+ and recreate your virtualenv (see docs/SETUP.md).",
    )


def _check_plugin_registry(
    *,
    group: str,
    kind: str,
    name: str,
) -> CheckResult:
    available = list_plugins(group)
    if name in available:
        return CheckResult(
            name=f"{kind}_registry",
            ok=True,
            message=f"{name} registered",
        )
    listing = ", ".join(available) or "(none)"
    return CheckResult(
        name=f"{kind}_registry",
        ok=False,
        message=f"unknown {kind} {name!r}",
        fix=f"Pick a registered name. Available: {listing}",
    )


def _is_local_rewriter_base_url(base_url: str) -> bool:
    host = urlparse(base_url).hostname or ""
    return host in {"localhost", "127.0.0.1", "::1"}


def _check_rewriter(config: AdhConfig) -> CheckResult:
    backend = (config.rewriter or "openai").strip().lower()
    if backend == "identity":
        return CheckResult(
            name="rewriter",
            ok=True,
            skipped=True,
            message="identity (no API key required)",
        )

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = (
        os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    ).rstrip("/")
    if api_key:
        return CheckResult(
            name="rewriter_api_key",
            ok=True,
            message="OPENAI_API_KEY is set",
        )
    if _is_local_rewriter_base_url(base_url):
        return CheckResult(
            name="rewriter_api_key",
            ok=True,
            message=f"local rewriter at {base_url} (no cloud key required)",
        )
    return CheckResult(
        name="rewriter_api_key",
        ok=False,
        message="OPENAI_API_KEY is not set",
        fix=(
            "export OPENAI_API_KEY=... or set OPENAI_BASE_URL to a local "
            f"OpenAI-compatible server (see {_SETUP_DOC}#3-environment-file)."
        ),
    )


def _check_local_torch() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult(
            name="local_torch",
            ok=False,
            message="torch is not installed",
            fix="pip install 'adversarial-detector-humanizer[local]'",
        )
    return CheckResult(
        name="local_torch",
        ok=True,
        message=f"torch {torch.__version__}",
    )


def _check_local_model(model_name: str, models_dir: Path | str | None) -> CheckResult:
    registry = model_registry(models_dir)
    spec = registry.get(model_name)
    if spec is None:
        return CheckResult(
            name=f"local_model_{model_name}",
            ok=False,
            message=f"unknown model {model_name!r}",
            fix="Use a published Raschka export name from `adh models list`.",
        )
    ready, status = artifact_status(spec)
    if ready:
        return CheckResult(
            name=f"local_model_{model_name}",
            ok=True,
            message=f"{model_name} ready at {spec.artifact_path}",
        )
    return CheckResult(
        name=f"local_model_{model_name}",
        ok=False,
        message=f"{model_name} artifact missing ({status})",
        fix=f"adh models fetch --model {model_name}",
    )


def _models_root(models_dir: Path | str | None) -> Path:
    if models_dir:
        return Path(models_dir).expanduser().resolve()
    return default_cache_dir()


def _check_models_directory(models_dir: Path | str | None) -> CheckResult:
    root = _models_root(models_dir)
    if not root.exists():
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            return CheckResult(
                name="models_directory",
                ok=False,
                message=f"cannot create {root}: {error}",
                fix="Choose a writable path and set ADH_MODELS_DIR or models_dir in adh.yaml.",
            )
    if not os.access(root, os.W_OK):
        return CheckResult(
            name="models_directory",
            ok=False,
            message=f"{root} is not writable",
            fix="Fix directory permissions or point ADH_MODELS_DIR at a writable folder.",
        )
    return CheckResult(
        name="models_directory",
        ok=True,
        message=str(root),
    )


def _check_verify_keys(config: AdhConfig) -> CheckResult:
    requested = [item.strip().lower() for item in config.verify_detectors if item.strip()]
    if not requested:
        return CheckResult(
            name="verify_keys",
            ok=True,
            skipped=True,
            message="verify disabled",
        )

    missing: list[str] = []
    for name in requested:
        env_key = _VERIFY_ENV_KEYS.get(name)
        if env_key is None:
            missing.append(f"{name} (unknown verify detector)")
            continue
        if not os.environ.get(env_key, "").strip():
            missing.append(env_key)

    if missing:
        return CheckResult(
            name="verify_keys",
            ok=False,
            message=f"missing: {', '.join(missing)}",
            fix=(
                "Add the keys to `.env` for each verify detector "
                f"(see {_SETUP_DOC}#3-environment-file)."
            ),
        )
    return CheckResult(
        name="verify_keys",
        ok=True,
        message=f"keys set for {', '.join(requested)}",
    )


def _skip_local_models(profile: str | None) -> CheckResult:
    label = profile or "current config"
    return CheckResult(
        name="local_models",
        ok=True,
        skipped=True,
        message=f"not required for profile {label}",
    )


def run_checks(config: AdhConfig) -> list[CheckResult]:
    """Validate environment prerequisites for ``config``."""
    results: list[CheckResult] = [
        _check_python_version(),
        _check_plugin_registry(
            group=GROUP_DETECTORS,
            kind="detector",
            name=config.detector,
        ),
        _check_plugin_registry(
            group=GROUP_REWRITERS,
            kind="rewriter",
            name=config.rewriter,
        ),
        _check_rewriter(config),
    ]

    model_names = local_models_for_detector(config.detector)
    if model_names:
        results.append(_check_local_torch())
        results.append(_check_models_directory(config.models_dir))
        seen: set[str] = set()
        for model_name in model_names:
            if model_name in seen:
                continue
            seen.add(model_name)
            results.append(_check_local_model(model_name, config.models_dir))
    else:
        results.append(_skip_local_models(config.profile))

    results.append(_check_verify_keys(config))
    return results
