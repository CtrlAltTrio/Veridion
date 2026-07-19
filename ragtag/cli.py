"""Provide operator commands for the ingestion workflow in PRD sections 6 and 8."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import yaml

from ragtag.config import settings
from ragtag.models import VerdictLabel
from ragtag.pipeline import Pipeline
from ragtag.rag import create_target_rag
from ragtag.normalize import extract_text
from ragtag.sealing import verify as verify_evidence
from ragtag.signals.anomaly import AnomalySignal
from ragtag.signals.influence import InfluenceSignal, Probe
from ragtag.signals.injection import InjectionSignal

app = typer.Typer(help="RAGtag pre-ingestion poisoning detector.")
_DEMO_PIPELINE: Pipeline | None = None

_SUPPORTED_DOCUMENTS = {".md", ".pdf", ".txt"}
_EVAL_REPORT_PATH = Path("data/labelled/eval_results.json")
_THRESHOLD_SWEEP_START = 0.35
_THRESHOLD_SWEEP_STOP = 0.95
_THRESHOLD_SWEEP_STEP = 0.05


@app.callback()
def root(
    demo_mode: bool = typer.Option(
        False,
        "--demo-mode",
        help="Load models, indexes, probe answers, and one generation before the command.",
    ),
) -> None:
    """Manage the configured RAGtag corpus and poisoning detector."""

    global _DEMO_PIPELINE
    if demo_mode:
        typer.echo("Pre-warming demo models and caches...")
        _DEMO_PIPELINE = _create_pipeline()
        _DEMO_PIPELINE.rag.embed(["Northwind demo warm-up"])
        _DEMO_PIPELINE.rag.generate(
            "Is the demo model ready?",
            ["The Northwind demo model is ready."],
        )
        typer.echo("Demo mode ready.")


@app.command()
def seed() -> None:
    """Build the local corpus index and persist its metadata cache."""

    rag = create_target_rag(rebuild=True)
    typer.echo(f"Seeded {rag.document_count} documents into {rag.chunk_count} chunks.")


@app.command()
def scan(file: Path) -> None:
    """Score one text, Markdown, or PDF document and print its verdict."""

    verdict = _get_pipeline().process(file.read_bytes(), file.name)

    typer.echo(f"Verdict: {verdict.verdict.value}")
    typer.echo(f"Score:   {verdict.score:.3f}")
    typer.echo(f"Reason:  {verdict.explanation}")
    typer.echo("Signals:")
    for name, result in verdict.signals.items():
        typer.echo(f"  {name:<10} {result.score:.3f}  {result.explanation}")


@app.command("verify")
def verify_command(file: Path, report: Path) -> None:
    """Verify a document against a standalone signed evidence report."""

    try:
        text = extract_text(file)
        evidence = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        typer.echo(f"FAIL: unable to read verification inputs: {error}", err=True)
        raise typer.Exit(code=1) from error

    valid, reason = verify_evidence(text, evidence)
    if valid:
        typer.echo(f"PASS: {reason}")
        return

    typer.echo(f"FAIL: {reason}", err=True)
    raise typer.Exit(code=1)


@app.command("eval")
def evaluate_command(
    output: Path = typer.Option(
        _EVAL_REPORT_PATH,
        "--output",
        "-o",
        help="JSON report consumed by the Streamlit metrics panel.",
    ),
) -> None:
    """Score the labelled set and print aggregate and per-family metrics."""

    pipeline = _get_pipeline()
    report = evaluate_labelled_set(pipeline, settings.paths.labelled_dir)
    _write_eval_report(report, output)
    _print_eval_report(report)
    typer.echo(f"\nWrote dashboard metrics to {output}")


def evaluate_labelled_set(pipeline: Pipeline, labelled_dir: Path) -> dict[str, Any]:
    """Evaluate clean and family-labelled poison documents with one pipeline."""

    records: list[dict[str, Any]] = []
    clean_dir = labelled_dir / "clean"
    poisoned_dir = labelled_dir / "poisoned"
    for path in _document_files(clean_dir):
        records.append(_score_labelled_document(pipeline, path, False, "clean"))
    for path in _document_files(poisoned_dir):
        family = path.relative_to(poisoned_dir).parts[0]
        records.append(_score_labelled_document(pipeline, path, True, family))

    if not records:
        raise typer.BadParameter(f"no labelled documents found under {labelled_dir}")

    aggregate = _classification_metrics(records)
    families = sorted({record["family"] for record in records if record["actual_positive"]})
    per_family = {
        family: _family_recall(records, family)
        for family in families
    }
    threshold_sweep = []
    threshold = _THRESHOLD_SWEEP_START
    while threshold <= _THRESHOLD_SWEEP_STOP + 1e-9:
        metrics = _classification_metrics(records, reject_threshold=threshold)
        threshold_sweep.append({"tau_high": round(threshold, 2), **metrics})
        threshold += _THRESHOLD_SWEEP_STEP

    return {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "decision_rule": "QUARANTINE and REJECT are positive",
        "configured_thresholds": {
            "tau_low": settings.thresholds.tau_low,
            "tau_high": settings.thresholds.tau_high,
        },
        "counts": {
            "total": len(records),
            "clean": sum(not record["actual_positive"] for record in records),
            "poisoned": sum(record["actual_positive"] for record in records),
        },
        "confusion_matrix": aggregate["confusion_matrix"],
        "precision": aggregate["precision"],
        "recall": aggregate["recall"],
        "f1": aggregate["f1"],
        "per_family": per_family,
        "threshold_sweep": threshold_sweep,
        "documents": records,
    }


def _create_pipeline() -> Pipeline:
    """Build the configured local detector once for CLI operations."""

    rag = create_target_rag()
    probes = _load_probes(settings.paths.probes_file)
    return Pipeline(
        rag,
        [
            AnomalySignal(rag, settings),
            InjectionSignal(),
            InfluenceSignal(rag, probes, settings),
        ],
        settings,
    )


def _get_pipeline() -> Pipeline:
    """Reuse the demo-warmed pipeline or create the normal command pipeline."""

    return _DEMO_PIPELINE or _create_pipeline()


def _score_labelled_document(
    pipeline: Pipeline,
    path: Path,
    actual_positive: bool,
    family: str,
) -> dict[str, Any]:
    """Score one labelled file and retain enough detail for later sweeps."""

    verdict = pipeline.process(path.read_bytes(), path.name)
    return {
        "path": path.as_posix(),
        "family": family,
        "actual_positive": actual_positive,
        "predicted_positive": verdict.verdict is not VerdictLabel.ADMIT,
        "verdict": verdict.verdict.value,
        "score": verdict.score,
        "signals": {
            name: result.score
            for name, result in verdict.signals.items()
        },
    }


def _classification_metrics(
    records: list[dict[str, Any]],
    reject_threshold: float | None = None,
) -> dict[str, Any]:
    """Return a confusion matrix and safe precision/recall/F1 metrics."""

    def predicted(record: dict[str, Any]) -> bool:
        if reject_threshold is None:
            return bool(record["predicted_positive"])
        return float(record["score"]) >= reject_threshold

    true_positive = sum(
        bool(record["actual_positive"]) and predicted(record)
        for record in records
    )
    false_positive = sum(
        not bool(record["actual_positive"]) and predicted(record)
        for record in records
    )
    true_negative = sum(
        not bool(record["actual_positive"]) and not predicted(record)
        for record in records
    )
    false_negative = sum(
        bool(record["actual_positive"]) and not predicted(record)
        for record in records
    )
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _family_recall(records: list[dict[str, Any]], family: str) -> dict[str, Any]:
    """Return caught/total/recall for one poison family."""

    family_records = [record for record in records if record["family"] == family]
    caught = sum(bool(record["predicted_positive"]) for record in family_records)
    return {
        "caught": caught,
        "total": len(family_records),
        "recall": _safe_ratio(caught, len(family_records)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Divide metrics safely when a prediction bucket is empty."""

    return float(numerator / denominator) if denominator else 0.0


def _document_files(directory: Path) -> list[Path]:
    """Return supported labelled files recursively in stable order."""

    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUPPORTED_DOCUMENTS
    )


def _write_eval_report(report: dict[str, Any], output: Path) -> None:
    """Atomically write the dashboard's machine-readable metrics report."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)


def _print_eval_report(report: dict[str, Any]) -> None:
    """Print judge-readable aggregate, family, and threshold tables."""

    matrix = report["confusion_matrix"]
    typer.echo("Confusion matrix (QUARANTINE + REJECT = positive)")
    typer.echo("                 Predicted +  Predicted -")
    typer.echo(
        f"Actual poison    {matrix['true_positive']:>11}  {matrix['false_negative']:>11}"
    )
    typer.echo(
        f"Actual clean     {matrix['false_positive']:>11}  {matrix['true_negative']:>11}"
    )
    typer.echo(
        f"\nPrecision {report['precision']:.3f}  "
        f"Recall {report['recall']:.3f}  F1 {report['f1']:.3f}"
    )

    typer.echo("\nRecall by poison family")
    typer.echo("Family                         Caught   Recall")
    for family, metrics in report["per_family"].items():
        typer.echo(
            f"{family:<30} {metrics['caught']:>2}/{metrics['total']:<2}   "
            f"{metrics['recall']:.3f}"
        )

    typer.echo("\nREJECT threshold sweep (tau_high)")
    typer.echo("tau_high  Precision  Recall    F1")
    configured = report["configured_thresholds"]["tau_high"]
    for row in report["threshold_sweep"]:
        marker = " *" if abs(row["tau_high"] - configured) < 1e-9 else ""
        typer.echo(
            f"{row['tau_high']:.2f}      {row['precision']:.3f}      "
            f"{row['recall']:.3f}  {row['f1']:.3f}{marker}"
        )


def main() -> None:
    """Run the RAGtag command-line application."""

    app()


def _load_probes(path: Path) -> list[Probe]:
    """Load and validate the configured probe file for Signal C."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Probe.model_validate(item) for item in payload.get("probes", [])]


if __name__ == "__main__":
    main()
