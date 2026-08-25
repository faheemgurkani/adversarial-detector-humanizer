"""Typer CLI for scoring and humanizing text."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.table import Table

from adh.engine import EngineConfig, humanize
from adh.exceptions import AdhError, InputError
from adh.factory import assert_inner_loop_detector, load_detector, load_gate, load_rewriter
from adh.models import DEFAULT_MODEL, fetch_models, list_models
from adh.report import score_to_label

load_dotenv(find_dotenv(usecwd=True))

app = typer.Typer(
    help="Detector-verified, sentence-targeted, meaning-preserving humanizer.",
    no_args_is_help=True,
    epilog="Verified score reduction only. No bypass guarantees.",
)
models_app = typer.Typer(help="List and download published Raschka detector exports.")
app.add_typer(models_app, name="models")
console = Console()
err_console = Console(stderr=True)


def _read_input(
    text: Optional[str],
    file: Optional[Path],
) -> str:
    provided = [value is not None for value in (text, file)]
    if sum(provided) > 1:
        raise InputError("use only one of --text or --file")
    if text is not None:
        return text
    if file is not None:
        if not file.is_file():
            raise InputError(f"file not found: {file}")
        return file.read_text(encoding="utf-8")
    if sys.stdin.isatty():
        raise InputError("pass --text, --file, or pipe text on stdin")
    return sys.stdin.read()


def _fail(error: Exception) -> None:
    err_console.print(f"[red]error:[/red] {error}")
    raise typer.Exit(code=1)


@app.command()
def score(
    text: Optional[str] = typer.Option(None, "--text", help="Text to score."),
    file: Optional[Path] = typer.Option(None, "--file", exists=False, help="UTF-8 file to score."),
    detector: str = typer.Option(DEFAULT_MODEL, "--detector", help="Detector name."),
    device: str = typer.Option("auto", "--device", help="auto, cpu, cuda, or mps."),
    models_dir: Optional[Path] = typer.Option(None, "--models-dir", help="Local model directory."),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON object."),
) -> None:
    """Score text with a local detector. Does not rewrite."""
    try:
        payload = _read_input(text, file)
        loaded = load_detector(detector, models_dir=models_dir, device=device)
        result = loaded.score(payload)
    except AdhError as error:
        _fail(error)
        return
    if as_json:
        payload_json = {
            "detector": loaded.name,
            "score": result.score,
            "label": result.label,
            "windows": [
                {
                    "text": window.text,
                    "score": window.score,
                    "label": window.label,
                    "start": window.start,
                    "end": window.end,
                }
                for window in result.windows
            ],
        }
        typer.echo(json.dumps(payload_json, indent=2))
        return
    console.print(f"detector: {loaded.name}")
    console.print(f"score: {result.score:.4f} ({result.label})")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", min=1, max=65535),
    detector: str = typer.Option("fake", "--detector", help="Bound detector for this process."),
    device: str = typer.Option("auto", "--device"),
    models_dir: Optional[Path] = typer.Option(None, "--models-dir"),
    semantic: str = typer.Option("lexical", "--semantic"),
    allow_lexical: bool = typer.Option(True, "--allow-lexical-gate/--no-allow-lexical-gate"),
) -> None:
    """Serve the local HTTP API on 127.0.0.1 by default."""
    try:
        import uvicorn
    except ImportError as error:
        _fail(error)
        return
    try:
        from adh.api import create_app

        loaded = load_detector(detector, models_dir=models_dir, device=device)
        gate = load_gate(prefer=semantic, allow_lexical=allow_lexical)
        application = create_app(
            detector=loaded,
            semantic_gate=gate,
            default_detector=detector,
            device=device,
            models_dir=models_dir,
        )
    except AdhError as error:
        _fail(error)
        return
    uvicorn.run(application, host=host, port=port, log_level="info")


@app.command("humanize")
def humanize_cmd(
    text: Optional[str] = typer.Option(None, "--text", help="Text to humanize."),
    file: Optional[Path] = typer.Option(None, "--file", help="UTF-8 file to humanize."),
    detector: str = typer.Option(DEFAULT_MODEL, "--detector", help="Detector name."),
    device: str = typer.Option("auto", "--device"),
    models_dir: Optional[Path] = typer.Option(None, "--models-dir"),
    target: float = typer.Option(30.0, "--target", min=0.0, max=100.0),
    verdict: float = typer.Option(45.0, "--verdict-score", min=0.0, max=100.0),
    max_rounds: int = typer.Option(5, "--max-rounds", min=1, max=20),
    sentence_threshold: float = typer.Option(50.0, "--sentence-threshold"),
    min_semantic: float = typer.Option(0.88, "--min-semantic"),
    max_rewrite_ratio: float = typer.Option(0.4, "--max-rewrite-ratio"),
    best_of: int = typer.Option(3, "--best-of", min=1, max=8),
    verify: Optional[str] = typer.Option(None, "--verify", help="Comma-separated pangram,gptzero."),
    verify_threshold: float = typer.Option(45.0, "--verify-threshold"),
    meaning_gate: str = typer.Option("auto", "--meaning-gate", help="auto, minilm, lexical, or full."),
    deploy_detector: list[str] = typer.Option([], "--deploy-detector", help="Held-out deploy detector(s)."),
    hard_mode: bool = typer.Option(False, "--hard-mode", help="Token-guided decode for stubborn sentences."),
    hard_mode_max_sentences: int = typer.Option(1, "--hard-mode-max-sentences", min=0, max=5),
    enable_logprob_blend: bool = typer.Option(True, "--logprob-blend/--no-logprob-blend"),
    rewriter_model: Optional[str] = typer.Option(None, "--rewriter-model"),
    semantic: str = typer.Option("auto", "--semantic", help="auto, minilm, or lexical."),
    allow_lexical: bool = typer.Option(
        False,
        "--allow-lexical-gate",
        help="Allow the lexical fallback when MiniLM is not installed.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print a RunReport JSON object."),
    output: Optional[Path] = typer.Option(None, "--output", help="Write rewritten text here."),
) -> None:
    """Rewrite only flagged sentences until the detector score drops or rounds end."""
    try:
        payload = _read_input(text, file)
        assert_inner_loop_detector(detector)
        loaded = load_detector(detector, models_dir=models_dir, device=device)
        gate = load_gate(prefer=semantic, allow_lexical=allow_lexical)
        rewriter = load_rewriter(model=rewriter_model)
        verify_detectors = [item.strip() for item in verify.split(",") if item.strip()] if verify else []
        hard_rewriter = None
        if hard_mode:
            from adh.hard import HardModeRewriter

            hard_rewriter = HardModeRewriter()
        report = humanize(
            payload,
            detector=loaded,
            rewriter=rewriter,
            semantic_gate=gate,
            hard_rewriter=hard_rewriter,
            config=EngineConfig(
                target_score=target,
                verdict_score=verdict,
                max_rounds=max_rounds,
                sentence_threshold=sentence_threshold,
                min_semantic_similarity=min_semantic,
                max_rewrite_ratio=max_rewrite_ratio,
                best_of_n=best_of,
                rewriter_model=rewriter_model or "gpt-4o-mini",
                detector=detector,
                meaning_gate_mode=meaning_gate,
                allow_lexical_gate=allow_lexical,
                verify_detectors=verify_detectors,
                verify_threshold=verify_threshold,
                deploy_detectors=deploy_detector,
                hard_mode=hard_mode,
                hard_mode_max_sentences=hard_mode_max_sentences,
                enable_logprob_blend=enable_logprob_blend,
            ),
        )
    except AdhError as error:
        _fail(error)
        return

    if output is not None:
        output.write_text(report.output_text, encoding="utf-8")
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return

    console.print(f"detector: {report.detector}")
    console.print(
        f"score: {report.score_before:.2f} -> {report.score_after:.2f} "
        f"({score_to_label(report.score_after)})"
    )
    console.print(f"semantic: {report.semantic_similarity:.4f}")
    console.print(f"rounds: {report.rounds}  stop: {report.stop_reason}")
    if report.sentences:
        table = Table(title="Flagged sentences")
        table.add_column("#")
        table.add_column("kept")
        table.add_column("before")
        table.add_column("after")
        table.add_column("text", overflow="fold")
        for item in report.sentences:
            table.add_row(
                str(item.i),
                "yes" if item.kept else "no",
                f"{item.score_before:.1f}",
                f"{item.score_after:.1f}",
                item.rewritten[:160],
            )
        console.print(table)
    console.print()
    console.print(report.output_text)


@models_app.command("list")
def models_list(
    models_dir: Optional[Path] = typer.Option(None, "--models-dir"),
) -> None:
    """Show published detectors and whether local artifacts are ready."""
    table = Table(title="Local Raschka detectors")
    table.add_column("name")
    table.add_column("kind")
    table.add_column("ready")
    table.add_column("status")
    table.add_column("hub")
    for row in list_models(models_dir):
        table.add_row(row["name"], row["kind"], row["ready"], row["status"], row["hub"])
    console.print(table)


@models_app.command("fetch")
def models_fetch(
    model: Optional[str] = typer.Option(None, "--model", help="Fetch one model. Default: all."),
    models_dir: Optional[Path] = typer.Option(None, "--models-dir"),
) -> None:
    """Download published weights from the Hugging Face Hub."""
    try:
        names = [model] if model else None
        fetched = fetch_models(names, models_dir=models_dir)
    except AdhError as error:
        _fail(error)
        return
    for spec in fetched:
        console.print(f"ready: {spec.name} -> {spec.artifact_path}")


if __name__ == "__main__":
    app()
