"""Load detectors, rewriters, and gates from importlib.metadata entry points."""

from __future__ import annotations

import inspect
from importlib.metadata import entry_points
from typing import Any

from adh.exceptions import InputError

GROUP_DETECTORS = "adh.detectors"
GROUP_REWRITERS = "adh.rewriters"
GROUP_GATES = "adh.gates"

_VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD
_ACCEPTED_KINDS = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


def list_plugins(group: str) -> list[str]:
    """Return sorted unique plugin names registered in ``group``."""
    return sorted({ep.name for ep in entry_points(group=group)})


def load_plugin(group: str, name: str, **kwargs: Any) -> Any:
    """Import and construct the plugin registered as ``name`` in ``group``."""
    eps = entry_points(group=group)
    if name not in eps.names:
        available = ", ".join(list_plugins(group)) or "(none)"
        raise InputError(
            f"unknown plugin {name!r} in group {group!r}. Available: {available}"
        )
    target = eps[name].load()
    return _construct(target, plugin_name=name, kwargs=kwargs)


def _construct(target: Any, *, plugin_name: str, kwargs: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        if not kwargs:
            return target()
        try:
            return target(**kwargs)
        except TypeError:
            return target()

    merged = dict(kwargs)
    if "model_name" in signature.parameters and "model_name" not in merged:
        merged["model_name"] = plugin_name

    if any(param.kind == _VAR_KEYWORD for param in signature.parameters.values()):
        return target(**merged)

    accepted = {
        name
        for name, param in signature.parameters.items()
        if param.kind in _ACCEPTED_KINDS
    }
    return target(**{key: value for key, value in merged.items() if key in accepted})
