"""Test the pluggable OpenAI-compatible TargetRAG adapter from PRD Tier 2.2."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np

from ragtag.config import Paths, Settings, SignalWeights, Thresholds
from ragtag.rag.openai_compat import OpenAICompatRAG


def test_compatible_adapter_embeds_retrieves_and_generates(tmp_path: Path, monkeypatch) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        if request.url.path == "/v1/embeddings":
            data = []
            for index, text in enumerate(payload["input"]):
                vector = [1.0, 0.0] if "refund" in text.lower() else [0.0, 1.0]
                data.append({"index": index, "embedding": vector})
            return httpx.Response(200, json={"data": data})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Grounded answer"}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    monkeypatch.setattr("ragtag.rag.openai_compat.httpx.Client", lambda **_kwargs: client)
    rag = OpenAICompatRAG(_config(tmp_path), rebuild=True)
    before = rag.corpus_embeddings()

    results = rag.retrieve("refund question", k=2, extra_docs=["refund candidate"])
    answer = rag.generate("What is the policy?", [str(results[0][0])])

    assert answer == "Grounded answer"
    assert results[0][0].source in {"corpus", "extra"}
    assert any(item.source == "extra" for item, _score in results)
    np.testing.assert_array_equal(rag.corpus_embeddings(), before)
    chat_payload = json.loads(requests[-1].content)
    assert chat_payload["temperature"] == 0
    assert chat_payload["max_tokens"] == 128


def test_chat_timeout_returns_stable_fallback(tmp_path: Path, monkeypatch) -> None:
    chat_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_calls
        payload = json.loads(request.content)
        if request.url.path == "/v1/embeddings":
            return httpx.Response(
                200,
                json={"data": [
                    {"index": index, "embedding": [1.0, 0.0]}
                    for index, _text in enumerate(payload["input"])
                ]},
            )
        chat_calls += 1
        raise httpx.ReadTimeout("stage timeout", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    monkeypatch.setattr("ragtag.rag.openai_compat.httpx.Client", lambda **_kwargs: client)
    rag = OpenAICompatRAG(_config(tmp_path), rebuild=True)

    first = rag.generate("question", ["context"])
    second = rag.generate("question", ["context"])

    assert "unavailable" in first.lower()
    assert second == first
    assert chat_calls == 1


def _config(tmp_path: Path) -> Settings:
    """Create an isolated compatible-endpoint configuration and tiny corpus."""

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "policy.txt").write_text("The refund policy is documented.", encoding="utf-8")
    return Settings(
        signal_weights=SignalWeights(anomaly=0.25, injection=0.25, influence=0.5),
        thresholds=Thresholds(tau_low=0.35, tau_high=0.65),
        encoder_name="unused",
        ollama_model="unused",
        rag_backend="openai_compat",
        openai_base_url="http://test",
        openai_chat_model="chat",
        openai_embedding_model="embed",
        top_k=5,
        paths=Paths(
            corpus_dir=corpus,
            probes_file=tmp_path / "probes.yaml",
            attacks_dir=tmp_path / "attacks",
            labelled_dir=tmp_path / "labelled",
            cache_dir=tmp_path / "cache",
            private_key=tmp_path / "key",
        ),
    )
