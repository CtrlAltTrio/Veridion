"""Detect instruction injection and Unicode obfuscation for PRD section 5."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TypedDict

from ragtag.models import Document, SignalResult
from ragtag.signals.base import Signal

_REGEX_FLAGS = re.IGNORECASE | re.MULTILINE

PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = tuple(
    (re.compile(pattern, _REGEX_FLAGS), weight)
    for pattern, weight in (
        (r"ignore (all |any )?previous instructions", 1.0),
        (r"disregard (the )?(above|prior|previous)", 1.0),
        (r"^\s*(SYSTEM|ASSISTANT|USER)\s*:", 0.9),
        (r"your new (task|instruction|role) is", 0.9),
        (r"you are now (a|an|the)", 0.7),
        (r"do not (mention|reveal|tell)", 0.7),
        (r"(always|instead) (answer|respond|say)", 0.6),
        (r"---\s*(END|BEGIN)\s+(CONTEXT|PROMPT|SYSTEM)", 0.8),
        (
            r"important:? (instruction|note) (for|to) (the )?"
            r"(ai|assistant|model)",
            0.9,
        ),
    )
)

_UNICODE_FLAG_WEIGHTS = {
    "zero-width": 0.6,
    "rtl override": 0.7,
    "unicode tag": 0.9,
    "homoglyph:": 0.5,
}


class InjectionMatch(TypedDict):
    """Serializable evidence for one heuristic hit."""

    kind: str
    matched_text: str
    weight: float


class InjectionSignal(Signal):
    """Score model-directed language with saturating weighted heuristics."""

    name = "injection"

    def score(self, document: Document) -> SignalResult:
        """Return every regex and Unicode hit with a saturated suspicion score."""

        matches: list[InjectionMatch] = []
        for pattern, weight in PATTERNS:
            raw_matches = [match.group(0) for match in pattern.finditer(document.raw_text)]
            raw_match_counts = Counter(
                " ".join(matched_text.split()).casefold()
                for matched_text in raw_matches
            )
            for matched_text in raw_matches:
                matches.append(
                    {
                        "kind": "pattern",
                        "matched_text": matched_text,
                        "weight": weight,
                    }
                )

            for match in pattern.finditer(document.clean_text):
                matched_text = match.group(0)
                normalized_match = " ".join(matched_text.split()).casefold()
                if raw_match_counts[normalized_match]:
                    raw_match_counts[normalized_match] -= 1
                    continue
                matches.append(
                    {
                        "kind": "pattern",
                        "matched_text": matched_text,
                        "weight": weight,
                    }
                )

        for unicode_flag in document.unicode_flags:
            normalized_flag = unicode_flag.casefold()
            for flag_class, weight in _UNICODE_FLAG_WEIGHTS.items():
                if flag_class in normalized_flag:
                    matches.append(
                        {
                            "kind": "unicode",
                            "matched_text": unicode_flag,
                            "weight": weight,
                        }
                    )
                    break

        score = 1.0 - math.prod(1.0 - match["weight"] for match in matches)
        if not matches:
            explanation = "no language directed at the model detected"
        else:
            strongest = max(matches, key=lambda match: match["weight"])
            strongest_text = " ".join(strongest["matched_text"].split())
            additional_count = len(matches) - 1
            suffix = f" (plus {additional_count} more)" if additional_count else ""
            explanation = (
                "contains an instruction aimed at the model: "
                f"'{strongest_text}'{suffix}"
            )

        return SignalResult(
            name="injection",
            score=score,
            explanation=explanation,
            details={"matches": matches},
        )
