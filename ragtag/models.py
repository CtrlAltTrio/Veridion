"""Define boundary schemas for detector results from PRD sections 5 and 9."""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel


class VerdictLabel(str, Enum):
    ADMIT = "ADMIT"
    QUARANTINE = "QUARANTINE"
    REJECT = "REJECT"


class Document(BaseModel):
    doc_id: str
    raw_text: str
    clean_text: str
    chunks: list[str]
    filename: Optional[str] = None
    unicode_flags: list[str] = []


class ProbeEffect(BaseModel):
    query: str
    retrieved: bool
    rank: Optional[int]
    answer_shift: float
    answer_before: str
    answer_after: str


class SignalResult(BaseModel):
    name: Literal["anomaly", "injection", "influence"]
    score: float
    explanation: str
    details: dict = {}


class Verdict(BaseModel):
    doc_id: str
    verdict: VerdictLabel
    score: float
    signals: dict[str, SignalResult]
    evidence: Optional[dict] = None
