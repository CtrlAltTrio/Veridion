"""Test signed evidence and offline verification from TECH_SPEC section 8."""

from __future__ import annotations

import json
import stat
from pathlib import Path

from typer.testing import CliRunner

from ragtag.cli import app
from ragtag.models import SignalResult, Verdict, VerdictLabel
from ragtag.normalize import build_document
from ragtag.sealing import seal, verify


def test_seal_and_verify_round_trip(tmp_path: Path, monkeypatch) -> None:
    key_path = tmp_path / ".ragtag" / "key"
    monkeypatch.setattr("ragtag.sealing.settings.paths.private_key", key_path)
    document = build_document("Approved Northwind policy.", "policy.txt")

    evidence = seal(document, _admit_verdict(document.doc_id))
    valid, reason = verify(document.raw_text, evidence)

    assert valid is True
    assert "signature" in reason
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert json.loads(evidence["canonical"])["text"] == document.clean_text
    assert len(evidence["sha256"]) == 64
    assert len(evidence["signature"]) == 128
    assert len(evidence["public_key"]) == 64
    assert evidence["timestamp"].endswith("Z")
    assert evidence["signals"] == {
        "anomaly": 0.01,
        "influence": 0.03,
        "injection": 0.02,
    }


def test_single_character_edit_reports_hash_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "ragtag.sealing.settings.paths.private_key",
        tmp_path / "key",
    )
    document = build_document("Approved Northwind policy.", "policy.txt")
    evidence = seal(document, _admit_verdict(document.doc_id))

    valid, reason = verify("Approved Northwind policy!", evidence)

    assert valid is False
    assert "hash mismatch" in reason


def test_tampered_signature_has_specific_reason(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "ragtag.sealing.settings.paths.private_key",
        tmp_path / "key",
    )
    document = build_document("Approved Northwind policy.", "policy.txt")
    evidence = seal(document, _admit_verdict(document.doc_id))
    evidence["signature"] = "00" * 64

    valid, reason = verify(document.raw_text, evidence)

    assert valid is False
    assert "signature invalid" in reason


def test_verify_cli_reports_pass_and_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "ragtag.sealing.settings.paths.private_key",
        tmp_path / "key",
    )
    document_path = tmp_path / "policy.txt"
    document_path.write_text("Approved Northwind policy.", encoding="utf-8")
    document = build_document(document_path.read_text(), document_path.name)
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(seal(document, _admit_verdict(document.doc_id))),
        encoding="utf-8",
    )
    runner = CliRunner()

    passed = runner.invoke(app, ["verify", str(document_path), str(report_path)])
    document_path.write_text("Approved Northwind policy!", encoding="utf-8")
    failed = runner.invoke(app, ["verify", str(document_path), str(report_path)])

    assert passed.exit_code == 0
    assert "PASS:" in passed.stdout
    assert failed.exit_code == 1
    assert "FAIL: hash mismatch" in failed.output


def _admit_verdict(doc_id: str) -> Verdict:
    """Build a complete admitted verdict with stable signal scores."""

    scores = {"anomaly": 0.01, "injection": 0.02, "influence": 0.03}
    return Verdict(
        doc_id=doc_id,
        verdict=VerdictLabel.ADMIT,
        score=0.0225,
        signals={
            name: SignalResult(
                name=name,
                score=score,
                explanation=f"{name} score is {score:.2f}",
            )
            for name, score in scores.items()
        },
    )
