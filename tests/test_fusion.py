"""Test weighted verdict fusion and pipeline resilience from PRD sections 5 and 6."""

from pathlib import Path

import pytest

from ragtag.config import Paths, Settings, SignalWeights, Thresholds
from ragtag.fusion import fuse
from ragtag.models import Document, SignalResult, VerdictLabel
from ragtag.pipeline import Pipeline
from ragtag.signals.base import Signal


@pytest.fixture()
def config(tmp_path: Path) -> Settings:
    return Settings(
        signal_weights=SignalWeights(anomaly=0.25, injection=0.25, influence=0.5),
        thresholds=Thresholds(tau_low=0.35, tau_high=0.65),
        encoder_name="fake",
        ollama_model="fake",
        top_k=5,
        paths=Paths(
            corpus_dir=tmp_path / "corpus",
            probes_file=tmp_path / "probes.yaml",
            attacks_dir=tmp_path / "attacks",
            labelled_dir=tmp_path / "labelled",
            cache_dir=tmp_path / "cache",
            private_key=tmp_path / "key",
        ),
    )


def test_score_below_low_threshold_is_admit(config: Settings) -> None:
    verdict = fuse(_results(influence=0.698), config)

    assert verdict.score == pytest.approx(0.349)
    assert verdict.verdict is VerdictLabel.ADMIT


def test_score_at_low_threshold_is_quarantine(config: Settings) -> None:
    verdict = fuse(_results(influence=0.7), config)

    assert verdict.score == pytest.approx(0.35)
    assert verdict.verdict is VerdictLabel.QUARANTINE


def test_score_at_high_threshold_is_reject(config: Settings) -> None:
    verdict = fuse(_results(anomaly=0.6, influence=1.0), config)

    assert verdict.score == pytest.approx(0.65)
    assert verdict.verdict is VerdictLabel.REJECT


def test_high_influence_alone_is_not_admitted(config: Settings) -> None:
    verdict = fuse(_results(influence=0.9), config)

    assert verdict.score == pytest.approx(0.45)
    assert verdict.verdict is VerdictLabel.QUARANTINE
    assert verdict.explanation.startswith("QUARANTINE: influence explanation")


def test_explanations_are_ordered_by_weighted_contribution(config: Settings) -> None:
    verdict = fuse(_results(anomaly=0.8, injection=0.2, influence=0.7), config)

    assert verdict.explanation == (
        "QUARANTINE: influence explanation; anomaly explanation; "
        "injection explanation"
    )


def test_pipeline_continues_after_signal_failure(
    config: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ragtag.pipeline.seal_document", lambda *_args: {"sealed": True})
    pipeline = Pipeline(
        rag=object(),  # The static test signals do not access the TargetRAG.
        signals=[
            StaticSignal("anomaly", 0.0),
            FailingSignal(),
            StaticSignal("influence", 0.0),
        ],
        config=config,
    )

    verdict = pipeline.process("A harmless internal document.", "safe.txt")

    assert verdict.verdict is VerdictLabel.ADMIT
    assert verdict.evidence == {"sealed": True}
    assert verdict.signals["injection"].score == 0.0
    assert verdict.signals["injection"].details["failed"] is True
    assert "RuntimeError" in verdict.signals["injection"].explanation


class StaticSignal(Signal):
    """Return a fixed score for pipeline orchestration tests."""

    def __init__(self, name: str, score: float) -> None:
        self.name = name
        self._score = score

    def score(self, document: Document) -> SignalResult:
        return SignalResult(
            name=self.name,
            score=self._score,
            explanation=f"{self.name} static explanation",
        )


class FailingSignal(Signal):
    """Raise deterministically to verify signal failure isolation."""

    name = "injection"

    def score(self, document: Document) -> SignalResult:
        raise RuntimeError("deliberate test failure")


def _results(
    anomaly: float = 0.0,
    injection: float = 0.0,
    influence: float = 0.0,
) -> dict[str, SignalResult]:
    """Build a complete result mapping with recognizable explanations."""

    return {
        "anomaly": SignalResult(
            name="anomaly",
            score=anomaly,
            explanation="anomaly explanation",
        ),
        "injection": SignalResult(
            name="injection",
            score=injection,
            explanation="injection explanation",
        ),
        "influence": SignalResult(
            name="influence",
            score=influence,
            explanation="influence explanation",
        ),
    }
