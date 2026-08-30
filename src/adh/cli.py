"""Typer CLI for scoring and humanizing text."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.table import Table

from adh.config import init_config_path, load_config, resolve_adh_config
from adh.doctor import all_passed, run_checks
from adh.errors import error_response
from adh.exceptions import AdhError, InputError
from adh.ids import new_request_id
from adh.jobs.runner import execute_humanize_job
from adh.factory import load_detector, load_gate, load_rewriter
from adh.models import DEFAULT_MODEL, fetch_models, list_models
from adh.profiles import TRY_SAMPLE_TEXT
from adh.report import score_to_label
from adh.service import run_humanize, run_score, humanize_request_from_config

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

_CLI_TO_ADH = {
    "profile": "profile",
    "detector": "detector",
    "device": "device",
    "models_dir": "models_dir",
    "target": "target_score",
    "verdict": "verdict_score",
    "max_rounds": "max_rounds",
    "sentence_threshold": "sentence_threshold",
    "min_semantic": "min_semantic_similarity",
    "max_rewrite_ratio": "max_rewrite_ratio",
    "best_of": "best_of_n",
    "rewriter_model": "rewriter_model",
    "semantic": "semantic",
    "allow_lexical": "allow_lexical_gate",
    "meaning_gate": "meaning_gate_mode",
    "verify": "verify_detectors",
    "verify_threshold": "verify_threshold",
    "deploy_detector": "deploy_detectors",
    "hard_mode": "hard_mode",
    "hard_mode_max_sentences": "hard_mode_max_sentences",
    "prepass": "prepass",
    "prepass_lang": "prepass_lang",
    "prepass_max_paragraphs": "prepass_max_paragraphs",
    "prepass_backend": "prepass_backend",
    "enable_logprob_blend": "enable_logprob_blend",
}

_SERVE_TO_ADH = {
    "detector": "detector",
    "device": "device",
    "models_dir": "models_dir",
    "semantic": "semantic",
    "allow_lexical": "allow_lexical_gate",
}


def _read_input(
    text: str | None,
    file: Path | None,
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


def _fail(error: Exception, *, as_json: bool = False) -> None:
    if as_json and isinstance(error, AdhError):
        payload = {"error": error_response(error, new_request_id())}
        typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(code=1)
    err_console.print(f"[red]error:[/red] {error}")
    raise typer.Exit(code=1)


def _option_from_command_line(ctx: typer.Context, name: str) -> bool:
    getter = getattr(ctx, "get_parameter_source", None)
    if getter is None:
        return False
    source = getter(name)
    if source is None:
        return False
    return getattr(source, "name", str(source)) == "COMMANDLINE"


@app.command()
def score(
    text: str | None = typer.Option(None, "--text", help="Text to score."),
    file: Path | None = typer.Option(None, "--file", exists=False, help="UTF-8 file to score."),
    detector: str = typer.Option(DEFAULT_MODEL, "--detector", help="Detector name."),
    device: str = typer.Option("auto", "--device", help="auto, cpu, cuda, or mps."),
    models_dir: Path | None = typer.Option(None, "--models-dir", help="Local model directory."),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON object."),
) -> None:
    """Score text with a local detector. Does not rewrite."""
    try:
        payload = _read_input(text, file)
        loaded, result = run_score(
            payload,
            detector_name=detector,
            device=device,
            models_dir=models_dir,
        )
    except AdhError as error:
        _fail(error, as_json=as_json)
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
def init(
    force: bool = typer.Option(False, "--force", help="Overwrite an existing adh.yaml."),
    path: Path | None = typer.Option(
        None,
        "--path",
        help="Directory to write adh.yaml (default: current directory).",
    ),
) -> None:
    """Write a starter adh.yaml in the current directory."""
    try:
        destination = path or Path.cwd()
        written = init_config_path(destination, force=force)
    except AdhError as error:
        _fail(error)
        return
    console.print(f"wrote {written}")


@app.command()
def doctor(
    ctx: typer.Context,
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Override profile from adh.yaml for this check.",
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        help="Path to adh.yaml (default: cwd or ADH_CONFIG).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable results."),
) -> None:
    """Validate setup for the configured profile before humanize or CI."""
    try:
        file_cfg = load_config(config_path)
        explicit: set[str] = set()
        if _option_from_command_line(ctx, "profile") and profile is not None:
            explicit.add("profile")
        cfg = resolve_adh_config(
            profile=profile if explicit else None,
            values={"profile": profile} if profile is not None else {},
            explicit=explicit,
            file=file_cfg,
        )
        results = run_checks(cfg)
    except AdhError as error:
        _fail(error, as_json=as_json)
        return

    if as_json:
        typer.echo(json.dumps([item.to_dict() for item in results], indent=2))
    else:
        table = Table(title="ADH doctor")
        table.add_column("check")
        table.add_column("status")
        table.add_column("message", overflow="fold")
        table.add_column("fix", overflow="fold")
        for item in results:
            status = "skip" if item.skipped else ("ok" if item.ok else "FAIL")
            table.add_row(item.name, status, item.message, item.fix or "")
        console.print(table)

    if not all_passed(results):
        raise typer.Exit(code=1)


@app.command()
def serve(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", min=1, max=65535),
    detector: str = typer.Option("fake", "--detector", help="Bound detector for this process."),
    device: str = typer.Option("auto", "--device"),
    models_dir: Path | None = typer.Option(None, "--models-dir"),
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

        file_cfg = load_config()
        explicit = {
            adh_name
            for cli_name, adh_name in _SERVE_TO_ADH.items()
            if _option_from_command_line(ctx, cli_name)
        }
        cfg = resolve_adh_config(
            values={
                "detector": detector,
                "device": device,
                "models_dir": models_dir,
                "semantic": semantic,
                "allow_lexical_gate": allow_lexical,
            },
            explicit=explicit,
            file=file_cfg,
        )
        loaded = load_detector(cfg.detector, models_dir=cfg.models_dir, device=cfg.device)
        gate = load_gate(prefer=cfg.semantic, allow_lexical=cfg.allow_lexical_gate)
        writer = load_rewriter(name=cfg.rewriter, model=cfg.rewriter_model)
        application = create_app(
            detector=loaded,
            rewriter=writer,
            semantic_gate=gate,
            server_config=cfg,
            default_detector=cfg.detector,
            device=cfg.device,
            models_dir=cfg.models_dir,
        )
    except AdhError as error:
        _fail(error)
        return
    uvicorn.run(application, host=host, port=port, log_level="info")


@app.command("try")
def try_cmd() -> None:
    """Run a sample humanize with the zero-key fast profile. No API key required."""
    try:
        report = run_humanize(
            TRY_SAMPLE_TEXT,
            config=resolve_adh_config(profile="fast"),
        )
    except AdhError as error:
        _fail(error)
        return
    typer.echo(report.model_dump_json(indent=2))


@app.command("humanize")
def humanize_cmd(
    ctx: typer.Context,
    text: str | None = typer.Option(None, "--text", help="Text to humanize."),
    file: Path | None = typer.Option(None, "--file", help="UTF-8 file to humanize."),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Preset bundle. Use 'fast' for zero-key test mode.",
    ),
    detector: str = typer.Option(DEFAULT_MODEL, "--detector", help="Detector name."),
    device: str = typer.Option("auto", "--device"),
    models_dir: Path | None = typer.Option(None, "--models-dir"),
    target: float = typer.Option(30.0, "--target", min=0.0, max=100.0),
    verdict: float = typer.Option(45.0, "--verdict-score", min=0.0, max=100.0),
    max_rounds: int = typer.Option(5, "--max-rounds", min=1, max=20),
    sentence_threshold: float = typer.Option(50.0, "--sentence-threshold"),
    min_semantic: float = typer.Option(0.88, "--min-semantic"),
    max_rewrite_ratio: float = typer.Option(0.4, "--max-rewrite-ratio"),
    best_of: int = typer.Option(3, "--best-of", min=1, max=8),
    verify: str | None = typer.Option(None, "--verify", help="Comma-separated pangram,gptzero."),
    verify_threshold: float = typer.Option(45.0, "--verify-threshold"),
    meaning_gate: str = typer.Option("auto", "--meaning-gate", help="auto, minilm, lexical, or full."),
    deploy_detector: list[str] = typer.Option([], "--deploy-detector", help="Held-out deploy detector(s)."),
    hard_mode: bool = typer.Option(False, "--hard-mode", help="Token-guided decode for stubborn sentences."),
    hard_mode_max_sentences: int = typer.Option(1, "--hard-mode-max-sentences", min=0, max=5),
    prepass: str = typer.Option("none", "--prepass", help="none or structural."),
    prepass_lang: str = typer.Option("fi", "--prepass-lang"),
    prepass_max_paragraphs: int = typer.Option(2, "--prepass-max-paragraphs", min=0, max=10),
    prepass_backend: str = typer.Option("llm", "--prepass-backend", help="llm or google."),
    enable_logprob_blend: bool = typer.Option(True, "--logprob-blend/--no-logprob-blend"),
    rewriter_model: str | None = typer.Option(None, "--rewriter-model"),
    semantic: str = typer.Option("auto", "--semantic", help="auto, minilm, or lexical."),
    allow_lexical: bool = typer.Option(
        False,
        "--allow-lexical-gate",
        help="Allow the lexical fallback when MiniLM is not installed.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print a RunReport JSON object."),
    async_mode: bool = typer.Option(
        False,
        "--async",
        help="Run humanize as a background job and wait for the result.",
    ),
    output: Path | None = typer.Option(None, "--output", help="Write rewritten text here."),
) -> None:
    """Rewrite only flagged sentences until the detector score drops or rounds end."""
    try:
        payload = _read_input(text, file)
        verify_detectors = (
            [item.strip() for item in verify.split(",") if item.strip()] if verify else []
        )
        values = {
            "detector": detector,
            "device": device,
            "models_dir": models_dir,
            "target_score": target,
            "verdict_score": verdict,
            "max_rounds": max_rounds,
            "sentence_threshold": sentence_threshold,
            "min_semantic_similarity": min_semantic,
            "max_rewrite_ratio": max_rewrite_ratio,
            "best_of_n": best_of,
            "rewriter_model": rewriter_model,
            "semantic": semantic,
            "allow_lexical_gate": allow_lexical,
            "meaning_gate_mode": meaning_gate,
            "verify_detectors": verify_detectors,
            "verify_threshold": verify_threshold,
            "deploy_detectors": list(deploy_detector),
            "hard_mode": hard_mode,
            "hard_mode_max_sentences": hard_mode_max_sentences,
            "prepass": prepass,
            "prepass_lang": prepass_lang,
            "prepass_max_paragraphs": prepass_max_paragraphs,
            "prepass_backend": prepass_backend,
            "enable_logprob_blend": enable_logprob_blend,
        }
        explicit = {
            adh_name
            for cli_name, adh_name in _CLI_TO_ADH.items()
            if _option_from_command_line(ctx, cli_name)
        }
        file_cfg = load_config()
        resolved = resolve_adh_config(
            profile=profile if _option_from_command_line(ctx, "profile") else None,
            values=values,
            explicit=explicit,
            file=file_cfg,
        )
        if async_mode:
            humanize_request = humanize_request_from_config(
                payload,
                resolved,
                compact=not as_json,
            )
            record = execute_humanize_job(
                humanize_request,
                context={
                    "file_config": file_cfg,
                    "detector": None,
                    "rewriter": None,
                    "semantic_gate": None,
                    "default_detector": detector,
                    "device": device,
                    "models_dir": models_dir,
                },
            )
            if record.status == "failed":
                if as_json:
                    typer.echo(
                        json.dumps(
                            {
                                "job_id": record.job_id,
                                "status": record.status,
                                "error": record.error,
                            },
                            indent=2,
                        )
                    )
                else:
                    message = (record.error or {}).get("message", "job failed")
                    err_console.print(f"[red]error:[/red] {message}")
                raise typer.Exit(code=1)
            report_payload = record.report or {}
            if output is not None:
                output.write_text(str(report_payload.get("output_text", "")), encoding="utf-8")
            if as_json:
                typer.echo(
                    json.dumps(
                        {"job_id": record.job_id, "status": record.status, **report_payload},
                        indent=2,
                    )
                )
                return
            console.print(f"job: {record.job_id}")
            console.print(str(report_payload.get("output_text", "")))
            return

        report = run_humanize(
            payload,
            config=resolved,
        )
    except AdhError as error:
        _fail(error, as_json=as_json)
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
    models_dir: Path | None = typer.Option(None, "--models-dir"),
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
    model: str | None = typer.Option(None, "--model", help="Fetch one model. Default: all."),
    models_dir: Path | None = typer.Option(None, "--models-dir"),
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
