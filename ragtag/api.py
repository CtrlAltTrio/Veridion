"""Expose the asynchronous REST API contract specified in PRD section 9."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ragtag.config import settings
from ragtag.models import Verdict
from ragtag.normalize import build_document, extract_text
from ragtag.pipeline import Pipeline
from ragtag.rag import create_target_rag
from ragtag.sealing import verify as verify_evidence
from ragtag.signals.anomaly import AnomalySignal
from ragtag.signals.influence import InfluenceSignal, Probe
from ragtag.signals.injection import InjectionSignal


class SubmitBody(BaseModel):
    """JSON document accepted by ``POST /submit``."""

    text: str
    filename: str | None = None


class SubmitResponse(BaseModel):
    """Acknowledgement returned before background scoring completes."""

    doc_id: str
    status: Literal["scoring"] = "scoring"


class VerifyBody(BaseModel):
    """Document and evidence supplied for an offline integrity check."""

    text: str
    evidence: dict[str, Any]


class VerifyResponse(BaseModel):
    """Result of comparing a document with its sealed digest."""

    valid: bool
    reason: str


class ProbeSetBody(BaseModel):
    """Replacement or extension requested for the active probe set."""

    probes: list[Probe]
    mode: Literal["replace", "extend"] = "replace"


class ProbeSetResponse(BaseModel):
    """Summary of the active in-memory probe set."""

    n_probes: int
    mode: Literal["replace", "extend"]


PipelineFactory = Callable[[], Pipeline]


def _build_pipeline() -> Pipeline:
    """Construct the one process-wide local pipeline and warm its caches."""

    rag = create_target_rag()
    probes = _read_probes(settings.paths.probes_file)
    return Pipeline(
        rag,
        [
            AnomalySignal(rag, settings),
            InjectionSignal(),
            InfluenceSignal(rag, probes, settings),
        ],
        settings,
    )


def create_app(pipeline_factory: PipelineFactory = _build_pipeline) -> FastAPI:
    """Create a FastAPI application with a single lifespan-scoped pipeline."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.pipeline = await asyncio.to_thread(pipeline_factory)
        application.state.verdicts: dict[str, Verdict] = {}
        application.state.pending: set[str] = set()
        application.state.errors: dict[str, str] = {}
        application.state.tasks: set[asyncio.Task[None]] = set()
        yield
        tasks = tuple(application.state.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    application = FastAPI(
        title="RAGtag API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.post("/submit", response_model=SubmitResponse)
    async def submit(request: Request) -> SubmitResponse:
        raw, filename = await _submission_payload(request)
        raw_text = await asyncio.to_thread(_extract_submission, raw, filename)
        doc_id = build_document(raw_text, filename).doc_id

        if doc_id not in request.app.state.pending:
            request.app.state.pending.add(doc_id)
            request.app.state.errors.pop(doc_id, None)
            task = asyncio.create_task(
                _score_document(request.app, doc_id, raw, filename),
                name=f"ragtag-score-{doc_id[:12]}",
            )
            request.app.state.tasks.add(task)
            task.add_done_callback(request.app.state.tasks.discard)
        return SubmitResponse(doc_id=doc_id)

    @application.get("/verdict/{doc_id}")
    async def verdict(doc_id: str, request: Request) -> Response:
        result = request.app.state.verdicts.get(doc_id)
        if result is not None:
            return Response(
                content=json.dumps(_verdict_payload(result)),
                media_type="application/json",
            )
        if doc_id in request.app.state.pending:
            return Response(
                content=json.dumps({"doc_id": doc_id, "status": "scoring"}),
                media_type="application/json",
                status_code=status.HTTP_202_ACCEPTED,
            )
        error = request.app.state.errors.get(doc_id)
        if error is not None:
            raise HTTPException(status_code=500, detail=f"scoring failed: {error}")
        raise HTTPException(status_code=404, detail="document not found")

    @application.post("/verify", response_model=VerifyResponse)
    async def verify(body: VerifyBody) -> VerifyResponse:
        valid, reason = await asyncio.to_thread(
            verify_evidence,
            body.text,
            body.evidence,
        )
        return VerifyResponse(valid=valid, reason=reason)

    @application.get("/corpus/stats")
    async def corpus_stats(request: Request) -> dict[str, int | str]:
        rag = request.app.state.pipeline.rag
        return {
            "n_docs": int(getattr(rag, "document_count", 0)),
            "n_probes": len(_influence_signal(request.app.state.pipeline).probes),
            "encoder": settings.encoder_name,
        }

    @application.post("/probes", response_model=ProbeSetResponse)
    async def probes(body: ProbeSetBody, request: Request) -> ProbeSetResponse:
        if not body.probes:
            raise HTTPException(status_code=422, detail="at least one probe is required")
        pipeline = request.app.state.pipeline
        influence = _influence_signal(pipeline)
        active = list(body.probes)
        if body.mode == "extend":
            by_id = {probe.id: probe for probe in influence.probes}
            by_id.update({probe.id: probe for probe in body.probes})
            active = list(by_id.values())
        replacement = await asyncio.to_thread(
            InfluenceSignal,
            pipeline.rag,
            active,
            pipeline.config,
        )
        pipeline.signals = [
            replacement if signal is influence else signal
            for signal in pipeline.signals
        ]
        return ProbeSetResponse(n_probes=len(active), mode=body.mode)

    return application


async def _score_document(
    application: FastAPI,
    doc_id: str,
    raw: bytes | str,
    filename: str | None,
) -> None:
    """Run synchronous model work in a thread and update the in-memory job store."""

    try:
        result = await asyncio.to_thread(
            application.state.pipeline.process,
            raw,
            filename,
        )
    except Exception as error:
        application.state.errors[doc_id] = f"{type(error).__name__}: {error}"
    else:
        application.state.verdicts[doc_id] = result
    finally:
        application.state.pending.discard(doc_id)


async def _submission_payload(request: Request) -> tuple[bytes | str, str | None]:
    """Parse JSON or multipart input without adding a multipart dependency."""

    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        try:
            body = SubmitBody.model_validate(await request.json())
        except Exception as error:
            raise HTTPException(status_code=422, detail="invalid JSON submission") from error
        return body.text, body.filename

    if content_type.startswith("multipart/form-data"):
        body = await request.body()
        message = BytesParser(policy=email_policy).parsebytes(
            b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body
        )
        for part in message.iter_parts():
            if part.get_param("name", header="content-disposition") != "file":
                continue
            filename = part.get_filename()
            payload = part.get_payload(decode=True)
            if filename and payload is not None:
                return payload, Path(filename).name
        raise HTTPException(status_code=422, detail="multipart submission requires a file")

    raise HTTPException(
        status_code=415,
        detail="use application/json or multipart/form-data",
    )


def _extract_submission(raw: bytes | str, filename: str | None) -> str:
    """Extract uploaded bytes while leaving submitted JSON text unchanged."""

    if isinstance(raw, str):
        return raw
    return extract_text(raw, filename)


def _read_probes(path: Path) -> list[Probe]:
    """Load the configured fixed probe set."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Probe.model_validate(item) for item in payload.get("probes", [])]


def _influence_signal(pipeline: Pipeline) -> InfluenceSignal:
    """Return the pipeline's influence signal or fail with a startup error."""

    for signal in pipeline.signals:
        if isinstance(signal, InfluenceSignal):
            return signal
    raise RuntimeError("pipeline has no influence signal")


def _verdict_payload(verdict: Verdict) -> dict[str, Any]:
    """Map internal rich signal details onto the stable PRD section 9 shape."""

    payload: dict[str, Any] = {
        "doc_id": verdict.doc_id,
        "verdict": verdict.verdict.value,
        "score": verdict.score,
        "signals": {},
        "evidence": verdict.evidence,
    }
    for name, result in verdict.signals.items():
        signal_payload: dict[str, Any] = {
            "score": result.score,
            "explanation": result.explanation,
        }
        if name == "injection":
            signal_payload["matches"] = [
                match.get("matched_text", "")
                for match in result.details.get("matches", [])
                if isinstance(match, dict)
            ]
        elif name == "influence":
            signal_payload.update(
                {
                    "probes_moved": result.details.get("probes_moved", 0),
                    "probes_total": result.details.get("probes_total", 0),
                    "per_probe": [
                        {
                            "query": effect.query,
                            "retrieved": effect.retrieved,
                            "rank": effect.rank,
                            "shift": effect.answer_shift,
                            "answer_before": effect.answer_before,
                            "answer_after": effect.answer_after,
                        }
                        for effect in result.details.get("per_probe", [])
                    ],
                }
            )
        payload["signals"][name] = signal_payload
    return payload


app = create_app()
