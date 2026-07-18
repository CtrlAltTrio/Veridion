"""Test calibrated embedding anomaly behavior required by PRD section 5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ragtag.config import Paths, Settings, SignalWeights, Thresholds
from ragtag.models import Document
from ragtag.rag.base import TargetRAG
from ragtag.signals.anomaly import AnomalySignal


class FakeRAG(TargetRAG):
    """Provide a tight trusted cluster and deterministic candidate embeddings."""

    def __init__(self) -> None:
        generator = np.random.default_rng(0)
        corpus = generator.normal(0.0, 0.025, size=(200, 8)).astype(np.float32)
        corpus[:, 0] += 1.0
        self._corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for text in texts:
            if "random" in text:
                vector = np.ones(8, dtype=np.float32)
                vector /= np.linalg.norm(vector)
            else:
                vector = self._corpus[0].copy()
            vectors.append(vector)
        return np.stack(vectors)

    def corpus_embeddings(self) -> np.ndarray:
        return self._corpus.copy()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        extra_docs: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        return []

    def generate(self, query: str, context: list[str]) -> str:
        return ""


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


def test_corpus_like_document_scores_low(config: Settings) -> None:
    signal = AnomalySignal(FakeRAG(), config)

    result = signal.score(_document("trusted corpus document"))

    assert signal.model.n_estimators == 200
    assert signal.model.contamination == 0.05
    assert signal.model.random_state == 0
    assert result.score < 0.2
    assert result.details["percentile"] < 50.0
    assert "percentile of corpus outlierness" in result.explanation


def test_random_text_embedding_scores_high(config: Settings) -> None:
    signal = AnomalySignal(FakeRAG(), config)

    result = signal.score(_document("random unrelated token sequence"))

    assert result.score > 0.8
    assert result.details["percentile"] > 99.0


def test_most_anomalous_chunk_controls_score(config: Settings) -> None:
    signal = AnomalySignal(FakeRAG(), config)
    document = Document(
        doc_id="candidate",
        raw_text="mixed",
        clean_text="mixed",
        chunks=["trusted corpus document", "random unrelated token sequence"],
    )

    result = signal.score(document)

    assert result.score > 0.8
    assert result.details["most_anomalous_chunk"] == 1
    assert len(result.details["raw_scores"]) == 2


def test_matching_corpus_hash_loads_cached_model(
    config: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = AnomalySignal(FakeRAG(), config)
    assert first.cache_path.is_file()

    with caplog.at_level("INFO"):
        second = AnomalySignal(FakeRAG(), config)

    assert second.cache_path == first.cache_path
    assert second.p50 == pytest.approx(first.p50)
    assert second.p99 == pytest.approx(first.p99)
    assert "Loaded anomaly model cache" in caplog.text


def _document(text: str) -> Document:
    """Build a one-chunk candidate for anomaly tests."""

    return Document(
        doc_id="candidate",
        raw_text=text,
        clean_text=text,
        chunks=[text],
    )
