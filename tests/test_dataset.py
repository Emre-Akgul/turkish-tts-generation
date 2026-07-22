"""Dataset selection and normalization tests."""

from typing import Any

import pytest

from turkish_tts_generation.config import ConfigError, DatasetConfig
from turkish_tts_generation.dataset import load_samples


class FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.column_names = list(rows[0])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def test_deterministic_shuffle_limit_and_column_mapping() -> None:
    dataset = FakeDataset([{"uid": f"id-{index}", "sentence": f"text {index}"} for index in range(6)])
    config = DatasetConfig(
        path="org/dataset",
        split="test",
        text_column="sentence",
        id_column="uid",
        shuffle=True,
        seed=7,
        limit=3,
    )

    first = load_samples(config, loader=lambda *_args, **_kwargs: dataset)
    second = load_samples(config, loader=lambda *_args, **_kwargs: dataset)

    assert first == second
    assert len(first) == 3
    assert first[0].sample_id.startswith("id-")
    assert first[0].source_index != 0


def test_generate_id_from_source_index() -> None:
    dataset = FakeDataset([{"text": "Merhaba"}])
    config = DatasetConfig(path="org/dataset", text_column="text")

    assert load_samples(config, loader=lambda *_args, **_kwargs: dataset)[0].sample_id == "row-00000000"


def test_reject_missing_text_column() -> None:
    dataset = FakeDataset([{"sentence": "Merhaba"}])
    config = DatasetConfig(path="org/dataset", text_column="text")

    with pytest.raises(ConfigError, match="missing configured columns"):
        load_samples(config, loader=lambda *_args, **_kwargs: dataset)


def test_conditioning_column_mapping() -> None:
    dataset = FakeDataset([{"text": "Merhaba", "audio": "/tmp/ref.wav", "transcript": "Selam", "speaker": 7}])
    config = DatasetConfig(
        path="org/dataset",
        text_column="text",
        reference_audio_column="audio",
        reference_text_column="transcript",
        speaker_id_column="speaker",
    )

    sample = load_samples(config, loader=lambda *_args, **_kwargs: dataset)[0]
    assert sample.reference_audio == "/tmp/ref.wav"
    assert sample.reference_text == "Selam"
    assert sample.speaker_id == "7"
