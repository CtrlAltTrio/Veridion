"""Exercise the asynchronous PRD section 9 submission and integrity contract."""

from __future__ import annotations

import hashlib
import time

from fastapi.testclient import TestClient

from ragtag.api import create_app
from ragtag.models import SignalResult, Verdict, VerdictLabel


class FakePipeline:
    """Small synchronous scorer used to prove API background execution."""

    def process(self, raw: bytes | str, filename: str | None = None) -> Verdict:
        text = raw.decode() if isinstance(raw, bytes) else raw
        digest = hashlib.sha256(text.encode()).hexdigest()
        signals = {
            name: SignalResult(
                name=name,
                score=0.0,
                explanation=f"{name} clean at 0.00",
            )
            for name in ("anomaly", "injection", "influence")
        }
        return Verdict(
            doc_id=digest,
            verdict=VerdictLabel.ADMIT,
            score=0.0,
            signals=signals,
        )


def test_json_submit_is_scored_and_retrievable() -> None:
    with TestClient(create_app(lambda: FakePipeline())) as client:
        submitted = client.post(
            "/submit",
            json={"text": "trusted policy", "filename": "policy.txt"},
        )
        assert submitted.status_code == 200
        assert submitted.json()["status"] == "scoring"

        result = _wait_for_verdict(client, submitted.json()["doc_id"])
        assert result["verdict"] == "ADMIT"
        assert set(result["signals"]) == {"anomaly", "injection", "influence"}
        assert result["evidence"] is None


def test_multipart_file_submission() -> None:
    with TestClient(create_app(lambda: FakePipeline())) as client:
        response = client.post(
            "/submit",
            files={"file": ("candidate.txt", b"candidate text", "text/plain")},
        )
        assert response.status_code == 200
        result = _wait_for_verdict(client, response.json()["doc_id"])
        assert result["doc_id"] == hashlib.sha256(b"candidate text").hexdigest()


def test_verify_uses_shared_evidence_verifier(monkeypatch) -> None:
    text = "sealed document"
    monkeypatch.setattr(
        "ragtag.api.verify_evidence",
        lambda current, evidence: (
            current == text and evidence.get("token") == "valid",
            "verified by shared sealing module",
        ),
    )
    with TestClient(create_app(lambda: FakePipeline())) as client:
        valid = client.post("/verify", json={"text": text, "evidence": {"token": "valid"}})
        invalid = client.post("/verify", json={"text": text + "!", "evidence": {"token": "valid"}})

    assert valid.json() == {
        "valid": True,
        "reason": "verified by shared sealing module",
    }
    assert invalid.json()["valid"] is False


def _wait_for_verdict(client: TestClient, doc_id: str) -> dict[str, object]:
    """Poll the deliberately asynchronous job API for a bounded interval."""

    for _ in range(50):
        response = client.get(f"/verdict/{doc_id}")
        if response.status_code == 200:
            return response.json()
        assert response.status_code == 202
        time.sleep(0.01)
    raise AssertionError("background verdict did not complete")
