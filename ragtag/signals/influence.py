"""Measure retrieval and answer influence for Signal C in PRD section 5."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from ragtag.config import Settings
from ragtag.models import Document, ProbeEffect, SignalResult
from ragtag.rag.base import TargetRAG
from ragtag.signals.base import Signal

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "clean_probes.json"
_RETRIEVAL_WEIGHT = 0.4
_MEAN_SHIFT_WEIGHT = 0.4
_MAX_SHIFT_WEIGHT = 0.2


class Probe(BaseModel):
    """One fixed influence query and its corpus-grounded expected fact."""

    id: str
    query: str
    expected_fact: str


class InfluenceSignal(Signal):
    """Quantify a candidate document's marginal effect on probe answers."""

    def __init__(
        self,
        rag: TargetRAG,
        probes: list[Probe],
        config: Settings,
    ) -> None:
        """Initialize the signal and ensure clean-corpus baselines are cached."""

        if not probes:
            raise ValueError("InfluenceSignal requires at least one probe")

        self.rag = rag
        self.probes = [Probe.model_validate(probe) for probe in probes]
        self.config = config
        self.cache_path = self.config.paths.cache_dir / _CACHE_FILENAME
        self._clean_probes: list[dict[str, Any]] = []
        self.ensure_cache()

    def ensure_cache(self) -> None:
        """Load matching clean-probe baselines or compute and persist new ones."""

        corpus_hash = self._corpus_manifest_hash()
        probes_hash = self._probes_file_hash()
        cache_key = f"{corpus_hash}:{probes_hash}"

        cached = self._load_cache(cache_key)
        if cached is not None:
            self._clean_probes = cached
            logger.info("Loaded clean probe cache from %s", self.cache_path)
            return

        clean_rows: list[dict[str, Any]] = []
        clean_answers: list[str] = []
        for probe in self.probes:
            retrieved = self.rag.retrieve(probe.query, k=self.config.top_k)
            context = [str(text) for text, _score in retrieved]
            answer = self.rag.generate(probe.query, context)
            clean_answers.append(answer)
            clean_rows.append(
                {
                    "id": probe.id,
                    "query": probe.query,
                    "expected_fact": probe.expected_fact,
                    "context": [
                        {"text": str(text), "score": float(score)}
                        for text, score in retrieved
                    ],
                    "answer": answer,
                }
            )

        answer_embeddings = self.rag.embed(clean_answers)
        self._validate_embedding_batch(answer_embeddings, len(self.probes))
        for row, embedding in zip(clean_rows, answer_embeddings, strict=True):
            row["embedding"] = np.asarray(embedding, dtype=np.float32).tolist()

        self._write_cache(
            {
                "cache_key": cache_key,
                "corpus_hash": corpus_hash,
                "probes_hash": probes_hash,
                "probes": clean_rows,
            }
        )
        self._clean_probes = clean_rows
        logger.info("Computed and persisted clean probe cache at %s", self.cache_path)

    def score(self, doc: Document) -> SignalResult:
        """Measure retrieval capture and semantic answer shift for ``doc``."""

        pending: list[dict[str, Any]] = []
        answers_to_embed: list[str] = []
        hits = 0

        for probe, clean_row in zip(self.probes, self._clean_probes, strict=True):
            retrieved = self.rag.retrieve(
                probe.query,
                k=self.config.top_k,
                extra_docs=doc.chunks,
            )
            extra_rank = self._first_extra_rank(retrieved)
            answer_before = str(clean_row["answer"])

            if extra_rank is None:
                pending.append(
                    {
                        "probe": probe,
                        "retrieved": False,
                        "rank": None,
                        "answer_before": answer_before,
                        "answer_after": answer_before,
                        "embedding_index": None,
                    }
                )
                continue

            hits += 1
            context = [str(text) for text, _score in retrieved]
            answer_after = self.rag.generate(probe.query, context)
            embedding_index = len(answers_to_embed)
            answers_to_embed.append(answer_after)
            pending.append(
                {
                    "probe": probe,
                    "retrieved": True,
                    "rank": extra_rank,
                    "answer_before": answer_before,
                    "answer_after": answer_after,
                    "embedding_index": embedding_index,
                }
            )

        if answers_to_embed:
            candidate_embeddings = self.rag.embed(answers_to_embed)
            self._validate_embedding_batch(candidate_embeddings, len(answers_to_embed))
        else:
            candidate_embeddings = np.empty((0, 0), dtype=np.float32)

        effects: list[ProbeEffect] = []
        shifts: list[float] = []
        for row, clean_row in zip(pending, self._clean_probes, strict=True):
            embedding_index = row["embedding_index"]
            if embedding_index is None:
                shift = 0.0
            else:
                shift = self._cosine_distance(
                    np.asarray(clean_row["embedding"], dtype=np.float32),
                    candidate_embeddings[embedding_index],
                )
            shifts.append(shift)
            probe = row["probe"]
            effects.append(
                ProbeEffect(
                    query=probe.query,
                    retrieved=row["retrieved"],
                    rank=row["rank"],
                    answer_shift=shift,
                    answer_before=row["answer_before"],
                    answer_after=row["answer_after"],
                )
            )

        probe_count = len(self.probes)
        retrieval_capture = hits / probe_count
        mean_shift = float(np.mean(shifts))
        max_shift = max(shifts)
        influence = (
            _RETRIEVAL_WEIGHT * retrieval_capture
            + _MEAN_SHIFT_WEIGHT * mean_shift
            + _MAX_SHIFT_WEIGHT * max_shift
        )

        if hits == 0:
            explanation = (
                "does not surface for any probe query; no measurable influence "
                "on answers"
            )
        else:
            retrieved_effects = [effect for effect in effects if effect.retrieved]
            largest_effect = max(
                retrieved_effects,
                key=lambda effect: effect.answer_shift,
            )
            explanation = (
                f"shifts {hits}/{probe_count} probe answers "
                f"(largest shift {max_shift:.2f} on '{largest_effect.query}')"
            )

        return SignalResult(
            name="influence",
            score=float(np.clip(influence, 0.0, 1.0)),
            explanation=explanation,
            details={
                "retrieval_capture": retrieval_capture,
                "mean_answer_shift": mean_shift,
                "max_answer_shift": max_shift,
                "probes_moved": hits,
                "probes_total": probe_count,
                "per_probe": effects,
            },
        )

    def _corpus_manifest_hash(self) -> str:
        """Hash the trusted corpus embedding matrix as a portable manifest."""

        embeddings = np.ascontiguousarray(
            self.rag.corpus_embeddings(),
            dtype=np.float32,
        )
        digest = hashlib.sha256()
        digest.update(str(embeddings.shape).encode("ascii"))
        digest.update(b"\0float32\0")
        digest.update(embeddings.tobytes())
        return digest.hexdigest()

    def _probes_file_hash(self) -> str:
        """Hash exact probe-file bytes, with a deterministic in-memory fallback."""

        probes_path = Path(self.config.paths.probes_file)
        if probes_path.is_file():
            content = probes_path.read_bytes()
        else:
            content = json.dumps(
                [probe.model_dump() for probe in self.probes],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _load_cache(self, cache_key: str) -> list[dict[str, Any]] | None:
        """Return validated cached rows when their key and probes still match."""

        if not self.cache_path.is_file():
            return None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            rows = payload["probes"]
            if not isinstance(rows, list):
                return None
            if payload.get("cache_key") != cache_key or len(rows) != len(self.probes):
                return None
            for probe, row in zip(self.probes, rows, strict=True):
                if not isinstance(row, dict):
                    return None
                if row.get("id") != probe.id or row.get("query") != probe.query:
                    return None
                if row.get("expected_fact") != probe.expected_fact:
                    return None
                if not isinstance(row.get("answer"), str):
                    return None
                if not isinstance(row.get("embedding"), list):
                    return None
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return None
        return rows

    def _write_cache(self, payload: dict[str, Any]) -> None:
        """Atomically persist clean-probe data as human-inspectable JSON."""

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.cache_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.cache_path)

    @staticmethod
    def _first_extra_rank(results: list[tuple[str, float]]) -> int | None:
        """Return the one-based rank of the first temporary retrieval result."""

        for rank, (text, _score) in enumerate(results, start=1):
            if getattr(text, "source", None) == "extra" or bool(
                getattr(text, "is_extra", False)
            ):
                return rank
        return None

    @staticmethod
    def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
        """Return cosine distance clipped to the model's documented 0..1 range."""

        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0 if np.array_equal(left, right) else 1.0
        similarity = float(np.dot(left, right) / (left_norm * right_norm))
        return float(np.clip(1.0 - similarity, 0.0, 1.0))

    @staticmethod
    def _validate_embedding_batch(embeddings: np.ndarray, expected_rows: int) -> None:
        """Reject malformed encoder output before it can corrupt cache alignment."""

        if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
            raise ValueError(
                "TargetRAG.embed returned an invalid batch shape: "
                f"expected {expected_rows} rows, got {embeddings.shape}"
            )
