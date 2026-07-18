"""Orchestrate the seven-stage ingestion gate described in PRD section 6."""

from __future__ import annotations

import logging
from typing import Literal, cast

from ragtag.config import Settings
from ragtag.fusion import fuse
from ragtag.models import SignalResult, Verdict, VerdictLabel
from ragtag.normalize import build_document, extract_text
from ragtag.rag.base import TargetRAG
from ragtag.sealing import seal_document
from ragtag.signals.base import Signal

logger = logging.getLogger(__name__)

_SIGNAL_NAMES = ("anomaly", "injection", "influence")
SignalName = Literal["anomaly", "injection", "influence"]


class Pipeline:
    """Run normalization, independent signals, fusion, and admission sealing."""

    def __init__(
        self,
        rag: TargetRAG,
        signals: list[Signal],
        config: Settings,
    ) -> None:
        """Store the target RAG, detector signals, and validated configuration."""

        self.rag = rag
        self.signals = signals
        self.config = config

    def process(
        self,
        raw_bytes_or_text: bytes | str,
        filename: str | None = None,
    ) -> Verdict:
        """Score one document without allowing an individual signal to abort it."""

        if isinstance(raw_bytes_or_text, bytes):
            raw_text = extract_text(raw_bytes_or_text, filename)
        elif isinstance(raw_bytes_or_text, str):
            raw_text = raw_bytes_or_text
        else:
            raise TypeError("raw_bytes_or_text must be bytes or str")

        document = build_document(raw_text, filename)
        results: dict[str, SignalResult] = {}
        for index, signal in enumerate(self.signals):
            signal_name = self._signal_name(signal, index)
            try:
                result = signal.score(document)
            except Exception as error:
                logger.exception("%s signal failed; continuing pipeline", signal_name)
                result = SignalResult(
                    name=signal_name,
                    score=0.0,
                    explanation=(
                        f"{signal_name} signal failed with "
                        f"{type(error).__name__}: {error}; assigned score 0.00"
                    ),
                    details={"failed": True, "error_type": type(error).__name__},
                )
            results[result.name] = result

        for signal_name in _SIGNAL_NAMES:
            if signal_name not in results:
                results[signal_name] = SignalResult(
                    name=signal_name,
                    score=0.0,
                    explanation=(
                        f"{signal_name} signal was not configured; assigned score 0.00"
                    ),
                    details={"missing": True},
                )

        verdict = fuse(results, self.config).model_copy(
            update={"doc_id": document.doc_id}
        )
        if verdict.verdict is VerdictLabel.ADMIT:
            try:
                evidence = seal_document(document, verdict)
            except NotImplementedError:
                logger.info("Evidence sealing is not implemented; returning ADMIT unsealed")
            except Exception:
                logger.exception("Evidence sealing failed; returning ADMIT unsealed")
            else:
                if evidence is not None:
                    verdict = verdict.model_copy(update={"evidence": evidence})
        return verdict

    @staticmethod
    def _signal_name(signal: Signal, index: int) -> SignalName:
        """Resolve a stable name even when a test or plugin signal later fails."""

        declared_name = getattr(signal, "name", None)
        if declared_name in _SIGNAL_NAMES:
            return cast(SignalName, declared_name)

        class_name = type(signal).__name__.lower()
        for signal_name in _SIGNAL_NAMES:
            if signal_name in class_name:
                return cast(SignalName, signal_name)

        fallback_index = min(index, len(_SIGNAL_NAMES) - 1)
        return cast(SignalName, _SIGNAL_NAMES[fallback_index])
