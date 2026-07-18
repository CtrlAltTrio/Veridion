"""Test normalization and extraction behavior required by PRD section 6 step 2."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from ragtag.normalize import (
    build_document,
    chunk,
    clean,
    detect_unicode_anomalies,
    extract_text,
)


@pytest.mark.parametrize("suffix", [".txt", ".md"])
def test_extract_text_file_formats(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"policy{suffix}"
    path.write_text("Refunds are available for 30 days.", encoding="utf-8")

    assert extract_text(path) == "Refunds are available for 30 days."


def test_extract_text_bytes_requires_filename() -> None:
    assert extract_text(b"# Policy", "policy.md") == "# Policy"
    with pytest.raises(ValueError, match="filename is required"):
        extract_text(b"missing type")


def test_extract_text_pdf_bytes() -> None:
    extracted = extract_text(_make_pdf("Hello from PDF"), "sample.pdf")

    assert "Hello from PDF" in extracted


def test_detects_zero_width_characters() -> None:
    flags = detect_unicode_anomalies("a\u200bb\u200cc\u200dd\ufeffe")

    assert "4 zero-width spaces" in flags


def test_detects_rtl_overrides() -> None:
    flags = detect_unicode_anomalies("safe\u202etext\u202c")

    assert "2 RTL override characters" in flags


def test_detects_unicode_tag_characters() -> None:
    flags = detect_unicode_anomalies("safe\U000e0061\U000e007ftext")

    assert "2 unicode tag characters" in flags


def test_detects_cyrillic_homoglyph() -> None:
    flags = detect_unicode_anomalies("pаypal")  # The second character is Cyrillic.

    assert "homoglyph: Cyrillic а -> Latin a" in flags


def test_detects_nfkc_homoglyph() -> None:
    flags = detect_unicode_anomalies("fullwidth ａ")

    assert "homoglyph: Fullwidth ａ -> Latin a" in flags


def test_clean_removes_anomalies_and_collapses_whitespace() -> None:
    raw = "  Refund\u200b\u202e  pоlicy.\U000e0061\nValid\tfor 30 days.  "

    assert clean(raw) == "Refund policy. Valid for 30 days."


def test_chunk_preserves_sentence_boundaries_and_whole_sentence_overlap() -> None:
    text = "Alpha sentence. Beta sentence. Gamma sentence."

    assert chunk(text, size=30, overlap=14) == [
        "Alpha sentence. Beta sentence.",
        "Beta sentence. Gamma sentence.",
    ]


def test_chunk_splits_oversized_sentence_without_exceeding_size() -> None:
    chunks = chunk("alpha beta gamma delta epsilon.", size=12, overlap=0)

    assert chunks == ["alpha beta", "gamma delta", "epsilon."]
    assert all(len(part) <= 12 for part in chunks)


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_chunk_rejects_invalid_boundaries(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk("Text.", size=size, overlap=overlap)


def test_build_document_populates_normalized_fields_and_flags() -> None:
    raw = "Refund pоlicy.\u200b"

    document = build_document(raw, "policy.md")

    assert document.raw_text == raw
    assert document.clean_text == "Refund policy."
    assert document.chunks == ["Refund policy."]
    assert document.filename == "policy.md"
    assert len(document.doc_id) == 64
    assert "1 zero-width space" in document.unicode_flags
    assert "homoglyph: Cyrillic о -> Latin o" in document.unicode_flags


def _make_pdf(text: str) -> bytes:
    """Create a minimal single-page PDF containing simple ASCII text."""

    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    pdf = io.BytesIO()
    pdf.write(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(pdf.tell())
        pdf.write(f"{number} 0 obj\n".encode("ascii"))
        pdf.write(body)
        pdf.write(b"\nendobj\n")

    xref_offset = pdf.tell()
    pdf.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return pdf.getvalue()
