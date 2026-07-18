"""Test Signal C influence behavior required by PRD sections 5 and 10."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ragtag.config import Paths, Settings, SignalWeights, Thresholds
from ragtag.models import Document, ProbeEffect
from ragtag.rag.base import TargetRAG
from ragtag.signals.influence import InfluenceSignal, Probe


class TaggedText(str):
    """Test retrieval text carrying the same source tag as LocalRAG."""

    source: str

    def __new__(cls, text: str, source: str) -> TaggedText:
        instance = super().__new__(cls, text)
        instance.source = source
        return instance


class FakeRAG(TargetRAG):
    """Deterministic RAG stub that models capture and answer changes."""

    def __init__(self) -> None:
        self.generate_calls = 0
        self.embed_batches: list[list[str]] = []
        self._corpus = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embed_batches.append(list(texts))
        vectors: list[list[float]] = []
        for text in texts:
            if text.startswith("poisoned answer"):
                vectors.append([0.0, 1.0, 0.0])
            elif text.startswith("benign answer"):
                vectors.append([0.999, 0.045, 0.0])
            else:
                vectors.append([1.0, 0.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)

    def corpus_embeddings(self) -> np.ndarray:
        return self._corpus.copy()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        extra_docs: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = [
            (TaggedText(f"trusted context for {query}", "corpus"), 0.8)
        ]
        candidate = " ".join(extra_docs or [])
        if "poison" in candidate:
            results.insert(0, (TaggedText(candidate, "extra"), 0.99))
        elif "benign" in candidate and "refund" in query.lower():
            results.append((TaggedText(candidate, "extra"), 0.2))
        return results[:k]

    def generate(self, query: str, context: list[str]) -> str:
        self.generate_calls += 1
        joined = " ".join(context)
        if "poison" in joined:
            return f"poisoned answer for {query}"
        if "benign" in joined:
            return f"benign answer for {query}"
        return f"clean answer for {query}"


@pytest.fixture()
def probes() -> list[Probe]:
    return [
        Probe(id="refund_window", query="What is the refund window?", expected_fact="30 days"),
        Probe(id="uptime", query="What is the uptime target?", expected_fact="99.9 percent"),
        Probe(id="password", query="What is the password minimum?", expected_fact="14 characters"),
        Probe(id="support", query="What is the P1 response?", expected_fact="15 minutes"),
    ]


@pytest.fixture()
def config(tmp_path: Path, probes: list[Probe]) -> Settings:
    probes_path = tmp_path / "probes.json"
    probes_path.write_text(
        json.dumps([probe.model_dump() for probe in probes]),
        encoding="utf-8",
    )
    return Settings(
        signal_weights=SignalWeights(anomaly=0.25, injection=0.25, influence=0.5),
        thresholds=Thresholds(tau_low=0.35, tau_high=0.65),
        encoder_name="fake",
        ollama_model="fake",
        top_k=5,
        paths=Paths(
            corpus_dir=tmp_path / "corpus",
            probes_file=probes_path,
            attacks_dir=tmp_path / "attacks",
            labelled_dir=tmp_path / "labelled",
            cache_dir=tmp_path / "cache",
            private_key=tmp_path / "key",
        ),
    )


def test_benign_document_has_low_influence(
    probes: list[Probe],
    config: Settings,
) -> None:
    rag = FakeRAG()
    signal = InfluenceSignal(rag, probes, config)
    baseline_generate_calls = rag.generate_calls
    result = signal.score(_document("benign supplemental refund guidance"))

    assert result.score < 0.3
    assert result.details["probes_moved"] == 1
    assert rag.generate_calls == baseline_generate_calls + 1
    effects = result.details["per_probe"]
    assert all(isinstance(effect, ProbeEffect) for effect in effects)
    assert effects[0].retrieved is True
    assert effects[0].rank == 2
    assert effects[0].answer_shift < 0.01
    serialized_effect = result.model_dump()["details"]["per_probe"][0]
    assert serialized_effect["query"] == probes[0].query
    assert serialized_effect["retrieved"] is True


def test_keyword_dense_contradiction_has_high_influence(
    probes: list[Probe],
    config: Settings,
) -> None:
    rag = FakeRAG()
    signal = InfluenceSignal(rag, probes, config)
    embed_calls_before_score = len(rag.embed_batches)
    result = signal.score(
        _document("poison refund uptime password support policy contradiction")
    )

    assert result.score > 0.7
    assert result.details["retrieval_capture"] == 1.0
    assert result.details["max_answer_shift"] == pytest.approx(1.0)
    assert result.explanation.startswith("shifts 4/4 probe answers")
    assert len(rag.embed_batches) == embed_calls_before_score + 1
    assert len(rag.embed_batches[-1]) == len(probes)


def test_irrelevant_document_skips_generation_and_has_no_influence(
    probes: list[Probe],
    config: Settings,
) -> None:
    rag = FakeRAG()
    signal = InfluenceSignal(rag, probes, config)
    baseline_generate_calls = rag.generate_calls
    embed_calls_before_score = len(rag.embed_batches)
    result = signal.score(_document("unrelated cafeteria menu"))

    assert result.score == pytest.approx(0.0)
    assert result.explanation == (
        "does not surface for any probe query; no measurable influence on answers"
    )
    assert rag.generate_calls == baseline_generate_calls
    assert len(rag.embed_batches) == embed_calls_before_score
    assert all(not effect.retrieved for effect in result.details["per_probe"])


def test_matching_cache_is_loaded_without_clean_generation(
    probes: list[Probe],
    config: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first_rag = FakeRAG()
    InfluenceSignal(first_rag, probes, config)
    assert first_rag.generate_calls == len(probes)

    second_rag = FakeRAG()
    with caplog.at_level("INFO"):
        InfluenceSignal(second_rag, probes, config)

    assert second_rag.generate_calls == 0
    assert "Loaded clean probe cache" in caplog.text


def test_changed_probe_file_recomputes_clean_cache(
    probes: list[Probe],
    config: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    InfluenceSignal(FakeRAG(), probes, config)
    config.paths.probes_file.write_text("changed probe manifest", encoding="utf-8")

    second_rag = FakeRAG()
    with caplog.at_level("INFO"):
        InfluenceSignal(second_rag, probes, config)

    assert second_rag.generate_calls == len(probes)
    assert "Computed and persisted clean probe cache" in caplog.text


def _document(text: str) -> Document:
    """Build the minimal normalized document needed by influence tests."""

    return Document(
        doc_id="candidate",
        raw_text=text,
        clean_text=text,
        chunks=[text],
    )
