"""Define the common signal contract required by PRD sections 5 and 6."""

from abc import ABC, abstractmethod

from ragtag.models import Document, SignalResult


class Signal(ABC):
    """Contract implemented by every RAGtag detector signal."""

    @abstractmethod
    def score(self, document: Document) -> SignalResult:
        """Score a document and return an explained signal result."""
