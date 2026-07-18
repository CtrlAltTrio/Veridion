"""Detect embedding-space anomalies for Signal A in PRD section 5."""

from __future__ import annotations

import hashlib
import logging
import pickle
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from ragtag.config import Settings
from ragtag.models import Document, SignalResult
from ragtag.rag.base import TargetRAG
from ragtag.signals.base import Signal

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1
_N_ESTIMATORS = 200
_CONTAMINATION = 0.05
_RANDOM_STATE = 0


class AnomalySignal(Signal):
    """Calibrated IsolationForest detector over the trusted corpus embedding space."""

    name = "anomaly"

    def __init__(self, rag: TargetRAG, config: Settings) -> None:
        """Load or fit the deterministic corpus anomaly model and calibration."""

        self.rag = rag
        self.config = config
        corpus_embeddings = np.ascontiguousarray(
            self.rag.corpus_embeddings(),
            dtype=np.float32,
        )
        if corpus_embeddings.ndim != 2 or corpus_embeddings.shape[0] == 0:
            raise ValueError(
                "AnomalySignal requires a non-empty 2D corpus embedding matrix"
            )

        self._embedding_dimension = corpus_embeddings.shape[1]
        self.corpus_hash = self._corpus_hash(corpus_embeddings)
        self.cache_path = self.config.paths.cache_dir / (
            f"anomaly_iforest_v{_CACHE_VERSION}_{self.corpus_hash}.pkl"
        )
        self.model: IsolationForest
        self.corpus_raw_scores: np.ndarray
        self.p50: float
        self.p99: float

        if not self._load_cache():
            self._fit(corpus_embeddings)
            self._write_cache()
            logger.info("Fitted and cached anomaly model at %s", self.cache_path)

    def score(self, document: Document) -> SignalResult:
        """Score every candidate chunk and calibrate the most anomalous one."""

        if not document.chunks:
            return SignalResult(
                name="anomaly",
                score=0.0,
                explanation="embedding sits at the 0.0th percentile of corpus outlierness",
                details={"raw_scores": [], "percentile": 0.0},
            )

        candidate_embeddings = np.asarray(
            self.rag.embed(document.chunks),
            dtype=np.float32,
        )
        if candidate_embeddings.ndim != 2:
            raise ValueError("TargetRAG.embed must return a 2D embedding matrix")
        if candidate_embeddings.shape != (
            len(document.chunks),
            self._embedding_dimension,
        ):
            raise ValueError(
                "candidate embedding shape does not match chunks and corpus dimension: "
                f"got {candidate_embeddings.shape}"
            )

        raw_scores = -self.model.score_samples(candidate_embeddings)
        max_index = int(np.argmax(raw_scores))
        max_raw = float(raw_scores[max_index])
        denominator = self.p99 - self.p50
        if denominator <= np.finfo(np.float64).eps:
            calibrated = 1.0 if max_raw > self.p50 else 0.0
        else:
            calibrated = float(
                np.clip((max_raw - self.p50) / denominator, 0.0, 1.0)
            )

        percentile = 100.0 * float(np.mean(self.corpus_raw_scores <= max_raw))
        return SignalResult(
            name="anomaly",
            score=calibrated,
            explanation=(
                f"embedding sits at the {percentile:.1f}th percentile "
                "of corpus outlierness"
            ),
            details={
                "raw_scores": raw_scores.astype(float).tolist(),
                "max_raw_score": max_raw,
                "most_anomalous_chunk": max_index,
                "percentile": percentile,
                "p50": self.p50,
                "p99": self.p99,
                "corpus_hash": self.corpus_hash,
            },
        )

    def _fit(self, corpus_embeddings: np.ndarray) -> None:
        """Fit the specified forest and derive corpus calibration statistics."""

        self.model = IsolationForest(
            n_estimators=_N_ESTIMATORS,
            contamination=_CONTAMINATION,
            random_state=_RANDOM_STATE,
        )
        self.model.fit(corpus_embeddings)
        self.corpus_raw_scores = np.asarray(
            -self.model.score_samples(corpus_embeddings),
            dtype=np.float64,
        )
        self.p50 = float(np.percentile(self.corpus_raw_scores, 50))
        self.p99 = float(np.percentile(self.corpus_raw_scores, 99))

    def _load_cache(self) -> bool:
        """Load a validated fitted model and its calibration distribution."""

        if not self.cache_path.is_file():
            return False
        try:
            payload: dict[str, Any] = pickle.loads(self.cache_path.read_bytes())
            if payload.get("version") != _CACHE_VERSION:
                return False
            if payload.get("corpus_hash") != self.corpus_hash:
                return False
            model = payload["model"]
            raw_scores = np.asarray(payload["corpus_raw_scores"], dtype=np.float64)
            p50 = float(payload["p50"])
            p99 = float(payload["p99"])
            if not isinstance(model, IsolationForest):
                return False
            if raw_scores.ndim != 1 or raw_scores.size == 0:
                return False
            if int(model.n_features_in_) != self._embedding_dimension:
                return False
            if not np.isfinite(raw_scores).all():
                return False
            if not np.isfinite([p50, p99]).all() or p99 < p50:
                return False
        except (
            OSError,
            EOFError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ):
            return False

        self.model = model
        self.corpus_raw_scores = raw_scores
        self.p50 = p50
        self.p99 = p99
        logger.info("Loaded anomaly model cache from %s", self.cache_path)
        return True

    def _write_cache(self) -> None:
        """Atomically persist the fitted forest and calibration data."""

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _CACHE_VERSION,
            "corpus_hash": self.corpus_hash,
            "model": self.model,
            "corpus_raw_scores": self.corpus_raw_scores,
            "p50": self.p50,
            "p99": self.p99,
        }
        temporary_path = self.cache_path.with_suffix(".pkl.tmp")
        temporary_path.write_bytes(
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        )
        temporary_path.replace(self.cache_path)

    @staticmethod
    def _corpus_hash(corpus_embeddings: np.ndarray) -> str:
        """Return a deterministic hash of corpus embedding shape and values."""

        digest = hashlib.sha256()
        digest.update(str(corpus_embeddings.shape).encode("ascii"))
        digest.update(b"\0float32\0")
        digest.update(corpus_embeddings.tobytes())
        return digest.hexdigest()
