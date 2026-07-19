"""Sign and verify tamper-evident admission reports for PRD Tier 2.1."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from ragtag.config import settings
from ragtag.models import Document, Verdict
from ragtag.normalize import clean


def seal(doc: Document, verdict: Verdict) -> dict[str, Any]:
    """Return an Ed25519-signed evidence report for one admitted document."""

    private_key = _load_or_create_private_key()
    canonical = _canonical_json(
        text=doc.clean_text,
        verdict=verdict.verdict.value,
        score=verdict.score,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    signature = private_key.sign(bytes.fromhex(digest)).hex()
    public_key = private_key.public_key().public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw,
    )

    signal_scores = {
        name: round(result.score, 6)
        for name, result in sorted(verdict.signals.items())
    }
    return {
        "canonical": canonical,
        "sha256": digest,
        "signature": signature,
        "public_key": public_key.hex(),
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "verdict": verdict.verdict.value,
        "score": round(verdict.score, 6),
        "signals": signal_scores,
    }


def verify(text: str, evidence: dict[str, Any]) -> tuple[bool, str]:
    """Verify the current text against an offline evidence report.

    Hash comparison happens before signature validation so a document edit has
    a specific, operator-friendly failure reason.
    """

    try:
        verdict = str(evidence["verdict"])
        score = float(evidence["score"])
        expected_digest = str(evidence["sha256"])
    except (KeyError, TypeError, ValueError):
        return False, "evidence report is missing canonical verdict, score, or hash data"

    canonical = _canonical_json(
        text=clean(text),
        verdict=verdict,
        score=score,
    )
    actual_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if not _constant_time_equal(actual_digest, expected_digest):
        return False, "hash mismatch: the document content has changed"

    try:
        public_key_bytes = bytes.fromhex(str(evidence["public_key"]))
        signature = bytes.fromhex(str(evidence["signature"]))
        public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        public_key.verify(signature, bytes.fromhex(actual_digest))
    except (KeyError, TypeError, ValueError, InvalidSignature):
        return False, "signature invalid: the evidence report is not authentic"

    return True, "hash and Ed25519 signature are valid"


def seal_document(doc: Document, verdict: Verdict) -> dict[str, Any]:
    """Backward-compatible pipeline entry point for :func:`seal`."""

    return seal(doc, verdict)


def _canonical_json(text: str, verdict: str, score: float) -> str:
    """Serialize exactly the canonical fields defined by TECH_SPEC section 8."""

    return json.dumps(
        {
            "text": text,
            "verdict": verdict,
            "score": round(score, 6),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _load_or_create_private_key() -> Ed25519PrivateKey:
    """Load the operator key or create it atomically with mode ``0600``."""

    key_path = Path(settings.paths.private_key).expanduser()
    key_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    try:
        key_bytes = key_path.read_bytes()
    except FileNotFoundError:
        private_key = Ed25519PrivateKey.generate()
        key_bytes = private_key.private_bytes(
            encoding=Encoding.Raw,
            format=PrivateFormat.Raw,
            encryption_algorithm=NoEncryption(),
        )
        try:
            descriptor = os.open(
                key_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            key_bytes = key_path.read_bytes()
        else:
            with os.fdopen(descriptor, "wb") as key_file:
                key_file.write(key_bytes)

    os.chmod(key_path, 0o600)
    try:
        return Ed25519PrivateKey.from_private_bytes(key_bytes)
    except ValueError as error:
        raise ValueError(f"invalid Ed25519 private key at {key_path}") from error


def _constant_time_equal(left: str, right: str) -> bool:
    """Compare two ASCII digest strings without early-exit timing leakage."""

    return hmac.compare_digest(left, right)
