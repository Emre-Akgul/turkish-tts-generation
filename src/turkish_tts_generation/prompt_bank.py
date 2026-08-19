"""Validation for versioned Turkish arena prompt banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

VERSION = "v1"
REVIEW_STATUS = "automated_only"
CATEGORY_QUOTAS = {
    "everyday": 72,
    "assistant": 48,
    "informational": 36,
    "announcement": 24,
    "normalization": 36,
    "pronunciation": 24,
}
LENGTH_QUOTAS = {"micro": 36, "short": 108, "medium": 72, "long": 24}
LENGTH_RANGES = {"micro": (2, 5), "short": (6, 12), "medium": (13, 22), "long": (23, 35)}
TURKISH_ALPHABET = set("abcçdefgğhıijklmnoöprsştuüvyz")
ID_PATTERN = re.compile(r"tr-arena-v1-(\d{4})$")
SPACE_PATTERN = re.compile(r"\s+")


class PromptBankError(ValueError):
    """Raised when an arena prompt bank violates its published contract."""


@dataclass(frozen=True, slots=True)
class PromptBankSummary:
    """Validated prompt-bank identity and distribution."""

    count: int
    sha256: str
    categories: dict[str, int]
    lengths: dict[str, int]


def _normalized_text(value: str) -> str:
    return SPACE_PATTERN.sub(" ", value.casefold()).strip(" .,!?:;\"'“”‘’")


def _word_count(text: str) -> int:
    return len(text.split())


def _validate_row(row: Any, *, line_number: int) -> list[str]:
    if not isinstance(row, dict):
        raise PromptBankError(f"line {line_number}: expected a JSON object")
    required = {"id", "text", "category", "length_bucket", "tags", "version", "review_status"}
    missing = sorted(required - row.keys())
    if missing:
        raise PromptBankError(f"line {line_number}: missing fields: {', '.join(missing)}")
    for name in ("id", "text", "category", "length_bucket", "version", "review_status"):
        if not isinstance(row[name], str) or not row[name].strip():
            raise PromptBankError(f"line {line_number}: {name} must be a non-empty string")
    if row["version"] != VERSION:
        raise PromptBankError(f"line {line_number}: version must be {VERSION}")
    if row["review_status"] != REVIEW_STATUS:
        raise PromptBankError(f"line {line_number}: review_status must be {REVIEW_STATUS}")
    if row["category"] not in CATEGORY_QUOTAS:
        raise PromptBankError(f"line {line_number}: unknown category {row['category']!r}")
    if row["length_bucket"] not in LENGTH_RANGES:
        raise PromptBankError(f"line {line_number}: unknown length bucket {row['length_bucket']!r}")
    if row["text"] != unicodedata.normalize("NFC", row["text"]):
        raise PromptBankError(f"line {line_number}: text is not Unicode NFC")
    if "\n" in row["text"] or "\r" in row["text"] or "\t" in row["text"]:
        raise PromptBankError(f"line {line_number}: text contains control whitespace")
    minimum, maximum = LENGTH_RANGES[row["length_bucket"]]
    words = _word_count(row["text"])
    if not minimum <= words <= maximum:
        raise PromptBankError(
            f"line {line_number}: {words} words does not match {row['length_bucket']} ({minimum}-{maximum})"
        )
    tags = row["tags"]
    if not isinstance(tags, list) or not tags or any(not isinstance(tag, str) or not tag for tag in tags):
        raise PromptBankError(f"line {line_number}: tags must be a non-empty string list")
    if len(tags) != len(set(tags)):
        raise PromptBankError(f"line {line_number}: tags must be unique")
    expected = row.get("expected_reading")
    if expected is not None and (not isinstance(expected, str) or not expected.strip()):
        raise PromptBankError(f"line {line_number}: expected_reading must be null or a non-empty string")
    return tags


def load_and_validate_prompt_bank(path: Path) -> PromptBankSummary:
    """Validate the v1 schema, quotas, IDs, originality checks, and alphabet coverage."""
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PromptBankError(f"line {line_number}: invalid JSON") from error
        _validate_row(row, line_number=line_number)
        rows.append(row)

    expected_count = sum(CATEGORY_QUOTAS.values())
    if len(rows) != expected_count:
        raise PromptBankError(f"expected {expected_count} prompts, found {len(rows)}")
    ids = [row["id"] for row in rows]
    expected_ids = [f"tr-arena-v1-{index:04d}" for index in range(1, expected_count + 1)]
    if ids != expected_ids:
        raise PromptBankError("prompt IDs must be unique, ordered, and contiguous from tr-arena-v1-0001")

    normalized = [_normalized_text(row["text"]) for row in rows]
    if len(normalized) != len(set(normalized)):
        raise PromptBankError("prompt texts must be unique after normalization")
    for left_index, left in enumerate(normalized):
        for right_index in range(left_index + 1, len(normalized)):
            right = normalized[right_index]
            if min(len(left), len(right)) >= 24 and SequenceMatcher(None, left, right).ratio() >= 0.92:
                raise PromptBankError(f"near-duplicate prompts: {ids[left_index]} and {ids[right_index]}")

    categories = Counter(str(row["category"]) for row in rows)
    lengths = Counter(str(row["length_bucket"]) for row in rows)
    if dict(categories) != CATEGORY_QUOTAS:
        raise PromptBankError(f"category quotas do not match: {dict(categories)}")
    if dict(lengths) != LENGTH_QUOTAS:
        raise PromptBankError(f"length quotas do not match: {dict(lengths)}")

    letters = {character for row in rows for character in row["text"].casefold() if character.isalpha()}
    missing_letters = sorted(TURKISH_ALPHABET - letters)
    if missing_letters:
        raise PromptBankError(f"missing Turkish letters: {', '.join(missing_letters)}")

    return PromptBankSummary(
        count=len(rows),
        sha256=hashlib.sha256(raw).hexdigest(),
        categories=dict(categories),
        lengths=dict(lengths),
    )


def main(args: list[str] | None = None) -> int:
    """Validate a prompt bank from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    options = parser.parse_args(args)
    try:
        summary = load_and_validate_prompt_bank(options.path)
    except (OSError, UnicodeError, PromptBankError) as error:
        print(f"error: {error}")
        return 2
    print(f"prompts={summary.count} sha256={summary.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
