"""Arena prompt-bank validation tests."""

import json
from pathlib import Path

import pytest

from turkish_tts_generation.prompt_bank import PromptBankError, load_and_validate_prompt_bank

PROMPT_BANK = Path(__file__).parents[1] / "data" / "turkish_arena_v1.jsonl"


def test_v1_prompt_bank_contract() -> None:
    summary = load_and_validate_prompt_bank(PROMPT_BANK)

    assert summary.count == 240
    assert sum(summary.categories.values()) == 240
    assert sum(summary.lengths.values()) == 240
    assert len(summary.sha256) == 64


def test_rejects_changed_prompt_order(tmp_path: Path) -> None:
    rows = [json.loads(line) for line in PROMPT_BANK.read_text(encoding="utf-8").splitlines()]
    rows[0], rows[1] = rows[1], rows[0]
    changed = tmp_path / "changed.jsonl"
    changed.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(PromptBankError, match="ordered"):
        load_and_validate_prompt_bank(changed)
