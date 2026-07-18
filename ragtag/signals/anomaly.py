"""Provide the temporary Signal A placeholder for PRD section 5."""

from ragtag.models import Document, SignalResult
from ragtag.signals.base import Signal


class AnomalySignal(Signal):
    """Runnable zero-score placeholder until anomaly detection is implemented."""

    name = "anomaly"

    def score(self, document: Document) -> SignalResult:
        """Return a transparent zero result without performing anomaly logic."""

        return SignalResult(
            name="anomaly",
            score=0.0,
            explanation="temporary anomaly detector found 0 embedding outliers",
            details={"temporary_stub": True},
        )
