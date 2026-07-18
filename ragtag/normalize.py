"""Normalize, inspect, extract, and chunk documents for PRD section 6 step 2."""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from ragtag.models import Document

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff\u2060]")
_RTL_OVERRIDE_RE = re.compile(r"[\u202a-\u202e]")
_TAG_CHAR_RE = re.compile(r"[\U000e0000-\U000e007f]")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

# Common cross-script characters used to disguise Latin text. NFKC does not
# fold these characters, so they supplement compatibility normalization.
_CONFUSABLES: dict[str, str] = {
    # Cyrillic lowercase
    "а": "a",
    "е": "e",
    "і": "i",
    "ј": "j",
    "о": "o",
    "р": "p",
    "с": "c",
    "ѕ": "s",
    "у": "y",
    "х": "x",
    # Cyrillic uppercase
    "А": "A",
    "В": "B",
    "Е": "E",
    "І": "I",
    "Ј": "J",
    "К": "K",
    "М": "M",
    "Н": "H",
    "О": "O",
    "Р": "P",
    "С": "C",
    "Т": "T",
    "Х": "X",
    # Greek characters commonly substituted into Latin words
    "Α": "A",
    "Β": "B",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "H",
    "Ι": "I",
    "Κ": "K",
    "Μ": "M",
    "Ν": "N",
    "Ο": "O",
    "Ρ": "P",
    "Τ": "T",
    "Υ": "Y",
    "Χ": "X",
    "α": "a",
    "ι": "i",
    "ο": "o",
    "ρ": "p",
    "υ": "u",
    "χ": "x",
}


def extract_text(
    path_or_bytes: str | Path | bytes,
    filename: str | None = None,
) -> str:
    """Extract UTF-8 text from a ``.txt``, ``.md``, or ``.pdf`` document.

    ``filename`` supplies the file type for byte input and may override a path's
    name when the caller is processing an uploaded file.
    """

    if isinstance(path_or_bytes, bytes):
        if not filename:
            raise ValueError("filename is required when extracting from bytes")
        source_name = filename
        data = path_or_bytes
        path: Path | None = None
    else:
        path = Path(path_or_bytes)
        source_name = filename or path.name
        data = None

    suffix = Path(source_name).suffix.lower()
    if suffix in {".txt", ".md"}:
        if data is not None:
            return data.decode("utf-8-sig")
        assert path is not None
        return path.read_text(encoding="utf-8-sig")

    if suffix == ".pdf":
        if data is not None:
            pdf_source: Path | io.BytesIO = io.BytesIO(data)
        else:
            assert path is not None
            pdf_source = path
        reader = PdfReader(pdf_source)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    raise ValueError(f"unsupported document type: {suffix or '<none>'}")


def detect_unicode_anomalies(text: str) -> list[str]:
    """Return human-readable descriptions of hidden and confusable Unicode."""

    flags: list[str] = []

    zero_width_count = len(_ZERO_WIDTH_RE.findall(text))
    if zero_width_count:
        noun = "space" if zero_width_count == 1 else "spaces"
        flags.append(f"{zero_width_count} zero-width {noun}")

    rtl_count = len(_RTL_OVERRIDE_RE.findall(text))
    if rtl_count:
        noun = "character" if rtl_count == 1 else "characters"
        flags.append(f"{rtl_count} RTL override {noun}")

    tag_count = len(_TAG_CHAR_RE.findall(text))
    if tag_count:
        noun = "character" if tag_count == 1 else "characters"
        flags.append(f"{tag_count} unicode tag {noun}")

    seen_substitutions: set[tuple[str, str]] = set()
    for character in text:
        replacement = _normalized_character(character)
        substitution = (character, replacement)
        if replacement == character or substitution in seen_substitutions:
            continue
        seen_substitutions.add(substitution)
        flags.append(f"homoglyph: {_describe_substitution(character, replacement)}")

    return flags


def clean(text: str) -> str:
    """Remove detected controls, normalize confusables, and collapse whitespace."""

    without_controls = _ZERO_WIDTH_RE.sub("", text)
    without_controls = _RTL_OVERRIDE_RE.sub("", without_controls)
    without_controls = _TAG_CHAR_RE.sub("", without_controls)
    normalized = unicodedata.normalize("NFKC", without_controls)
    normalized = "".join(_CONFUSABLES.get(character, character) for character in normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def chunk(text: str, size: int, overlap: int) -> list[str]:
    """Split text into character-bounded, sentence-aware overlapping chunks.

    Whole sentences are retained whenever they fit within ``size``. Oversized
    sentences fall back to word-aware splitting. Overlap consists of complete
    trailing sentence units and never makes a chunk exceed ``size``.
    """

    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")

    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return []

    units: list[str] = []
    for sentence in _SENTENCE_BOUNDARY_RE.split(collapsed):
        units.extend(_split_oversized_unit(sentence, size))

    chunks: list[str] = []
    start = 0
    while start < len(units):
        end = start
        current: list[str] = []
        while end < len(units):
            candidate = " ".join([*current, units[end]])
            if len(candidate) > size:
                break
            current.append(units[end])
            end += 1

        chunks.append(" ".join(current))
        if end == len(units):
            break

        overlap_start = end
        while overlap_start > start:
            candidate_overlap = " ".join(units[overlap_start - 1 : end])
            if len(candidate_overlap) > overlap:
                break
            overlap_start -= 1

        # Guarantee that the next chunk contains new material. If the overlap
        # plus the next unit is too large, discard overlap units until it fits.
        while (
            overlap_start < end
            and len(" ".join(units[overlap_start : end + 1])) > size
        ):
            overlap_start += 1
        start = overlap_start

    return chunks


def build_document(raw: str, filename: str | None = None) -> Document:
    """Build a normalized ``Document`` with deterministic content identity."""

    clean_text = clean(raw)
    return Document(
        doc_id=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        raw_text=raw,
        clean_text=clean_text,
        chunks=chunk(clean_text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP),
        filename=filename,
        unicode_flags=detect_unicode_anomalies(raw),
    )


def _normalized_character(character: str) -> str:
    """Return the compatibility/confusable-normalized form of one character."""

    return _CONFUSABLES.get(character, unicodedata.normalize("NFKC", character))


def _describe_substitution(character: str, replacement: str) -> str:
    """Describe a Unicode substitution in concise, judge-readable language."""

    name = unicodedata.name(character, "Unicode character")
    script = name.split(maxsplit=1)[0].title()
    if replacement.isascii() and replacement.isalpha():
        target = f"Latin {replacement}"
    else:
        target = repr(replacement)
    return f"{script} {character} -> {target}"


def _split_oversized_unit(unit: str, size: int) -> list[str]:
    """Split one oversized sentence on words, then hard-split oversized words."""

    if len(unit) <= size:
        return [unit]

    pieces: list[str] = []
    current = ""
    for word in unit.split():
        if len(word) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(word[index : index + size] for index in range(0, len(word), size))
            continue

        candidate = f"{current} {word}".strip()
        if len(candidate) <= size:
            current = candidate
        else:
            pieces.append(current)
            current = word

    if current:
        pieces.append(current)
    return pieces
