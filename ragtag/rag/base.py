"""Define the target-RAG interface required by PRD sections 5 and 6."""

from abc import ABC, abstractmethod

import numpy as np


class TargetRAG(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (n, d) embeddings using the SAME encoder the RAG retrieves with."""

    @abstractmethod
    def corpus_embeddings(self) -> np.ndarray:
        """(N, d) embeddings of the currently trusted corpus."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 5,
        extra_docs: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Top-k (chunk_text, score).

        If extra_docs are given, retrieve as if they were temporarily part of
        the corpus without mutating it.
        """

    @abstractmethod
    def generate(self, query: str, context: list[str]) -> str:
        """Answer the query grounded in the given context chunks."""
