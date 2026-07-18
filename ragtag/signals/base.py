"""Define the common signal contract required by PRD sections 5 and 6."""

from abc import ABC, abstractmethod

from ragtag.models import Document, SignalResult
from ragtag.rag.base import TargetRAG


class Signal(ABC):
    """Contract implemented by every RAGtag detector signal."""

    @abstractmethod
    def score(self, document: Document, rag: TargetRAG) -> SignalResult:
        """Score a document and return an explained signal result."""
