"""Provide the temporary Signal B placeholder for PRD section 5."""

from ragtag.models import Document, SignalResult
from ragtag.signals.base import Signal


class InjectionSignal(Signal):
    """Runnable zero-score placeholder until injection detection is implemented."""

    name = "injection"

    def score(self, document: Document) -> SignalResult:
        """Return a transparent zero result without performing injection logic."""

        return SignalResult(
            name="injection",
            score=0.0,
            explanation="temporary injection detector matched 0 instruction patterns",
            details={"temporary_stub": True},
        )
