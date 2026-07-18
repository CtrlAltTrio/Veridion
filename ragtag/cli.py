"""Provide operator commands for the ingestion workflow in PRD sections 6 and 8."""

from pathlib import Path

import typer
import yaml

from ragtag.config import settings
from ragtag.pipeline import Pipeline
from ragtag.rag.local import LocalRAG
from ragtag.signals.anomaly import AnomalySignal
from ragtag.signals.influence import InfluenceSignal, Probe
from ragtag.signals.injection import InjectionSignal

app = typer.Typer(help="RAGtag pre-ingestion poisoning detector.")


@app.callback()
def root() -> None:
    """Manage the local RAGtag corpus and poisoning detector."""


@app.command()
def seed() -> None:
    """Build the local corpus index and persist its metadata cache."""

    rag = LocalRAG(rebuild=True)
    typer.echo(f"Seeded {rag.document_count} documents into {rag.chunk_count} chunks.")


@app.command()
def scan(file: Path) -> None:
    """Score one text, Markdown, or PDF document and print its verdict."""

    rag = LocalRAG()
    probes = _load_probes(settings.paths.probes_file)
    signals = [
        AnomalySignal(),
        InjectionSignal(),
        InfluenceSignal(rag, probes, settings),
    ]
    verdict = Pipeline(rag, signals, settings).process(file.read_bytes(), file.name)

    typer.echo(f"Verdict: {verdict.verdict.value}")
    typer.echo(f"Score:   {verdict.score:.3f}")
    typer.echo(f"Reason:  {verdict.explanation}")
    typer.echo("Signals:")
    for name, result in verdict.signals.items():
        typer.echo(f"  {name:<10} {result.score:.3f}  {result.explanation}")


def main() -> None:
    """Run the RAGtag command-line application."""

    app()


def _load_probes(path: Path) -> list[Probe]:
    """Load and validate the configured probe file for Signal C."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Probe.model_validate(item) for item in payload.get("probes", [])]


if __name__ == "__main__":
    main()
