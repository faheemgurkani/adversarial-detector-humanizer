from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_IMPORTS = frozenset({"fastapi", "typer", "uvicorn"})
CORE_RELATIVE = (
    "engine.py",
    "report.py",
    "preserve.py",
    "ranking.py",
    "audit.py",
)


def _adh_src() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "adh"


def _python_files(*relative: str) -> list[Path]:
    root = _adh_src()
    files: list[Path] = []
    for item in relative:
        path = root / item
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        else:
            files.append(path)
    return files


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_core_modules_no_fastapi() -> None:
    files = _python_files(*CORE_RELATIVE, "gates", "detectors")
    assert files
    violations: list[str] = []
    for path in files:
        imported = _imported_modules(path) & FORBIDDEN_IMPORTS
        if imported:
            violations.append(f"{path.relative_to(_adh_src())}: {sorted(imported)}")
    assert violations == []
