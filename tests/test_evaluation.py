"""Test labelled metrics and threshold reporting from PRD Tier 2.4."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragtag.cli import _classification_metrics, evaluate_labelled_set
from ragtag.models import SignalResult, Verdict, VerdictLabel


class FakePipeline:
    """Map fixture filenames to deterministic scores and verdicts."""

    def process(self, raw: bytes | str, filename: str | None = None) -> Verdict:
        assert filename is not None
        score = {
            "clean_ok.txt": 0.10,
            "clean_fp.txt": 0.40,
            "caught.txt": 0.80,
            "missed.txt": 0.20,
        }[filename]
        if score >= 0.65:
            label = VerdictLabel.REJECT
        elif score >= 0.35:
            label = VerdictLabel.QUARANTINE
        else:
            label = VerdictLabel.ADMIT
        signals = {
            name: SignalResult(name=name, score=score, explanation=f"{name} result")
            for name in ("anomaly", "injection", "influence")
        }
        return Verdict(
            doc_id=filename,
            verdict=label,
            score=score,
            signals=signals,
        )


def test_labelled_report_contains_matrix_family_and_sweep(tmp_path: Path) -> None:
    labelled = tmp_path / "labelled"
    _write(labelled / "clean" / "clean_ok.txt")
    _write(labelled / "clean" / "clean_fp.txt")
    _write(labelled / "poisoned" / "influence_only" / "caught.txt")
    _write(labelled / "poisoned" / "influence_only" / "missed.txt")

    report = evaluate_labelled_set(FakePipeline(), labelled)

    assert report["counts"] == {"total": 4, "clean": 2, "poisoned": 2}
    assert report["confusion_matrix"] == {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 1,
    }
    assert report["precision"] == pytest.approx(0.5)
    assert report["recall"] == pytest.approx(0.5)
    assert report["f1"] == pytest.approx(0.5)
    assert report["per_family"]["influence_only"] == {
        "caught": 1,
        "total": 2,
        "recall": 0.5,
    }
    thresholds = [row["tau_high"] for row in report["threshold_sweep"]]
    assert thresholds == pytest.approx([0.35 + 0.05 * index for index in range(13)])


def test_zero_denominators_are_safe() -> None:
    metrics = _classification_metrics(
        [{"actual_positive": False, "predicted_positive": False, "score": 0.0}]
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def _write(path: Path) -> None:
    """Create a minimal supported fixture document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture", encoding="utf-8")
