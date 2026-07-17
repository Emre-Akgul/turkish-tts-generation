"""Configuration parsing tests."""

from pathlib import Path

import pytest

from turkish_tts_generation.config import ConfigError, load_config

VALID_CONFIG = """
dataset:
  path: org/dataset
  split: test
  text_column: text
  limit: 2
targets:
  - name: target-a
    engine: noop
    model_id: model-a
    batch_size: 2
output:
  root: outputs
  run_name: test-run
"""


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "generation.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_config(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, VALID_CONFIG))

    assert config.dataset.path == "org/dataset"
    assert config.dataset.text_column == "text"
    assert config.targets[0].batch_size == 2
    assert config.output.audio_format == "wav"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("  text_column: text\n", "", "text_column"),
        ("  limit: 2\n", "  limit: 0\n", "limit"),
        (
            "  - name: target-a\n    engine: noop\n    model_id: model-a\n    batch_size: 2\n",
            "  - name: target-a\n    engine: noop\n    model_id: model-a\n    batch_size: 2\n"
            "  - name: target-a\n    engine: noop\n    model_id: model-b\n",
            "unique",
        ),
    ],
)
def test_reject_invalid_config(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(_write_config(tmp_path, VALID_CONFIG.replace(old, new)))
