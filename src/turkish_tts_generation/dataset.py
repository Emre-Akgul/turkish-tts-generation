"""Hugging Face dataset loading and text normalization."""

import os
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from datasets import load_dataset

from turkish_tts_generation.config import ConfigError, DatasetConfig
from turkish_tts_generation.contracts import GenerationItem, TextSample

DatasetLoader = Callable[..., Any]


def load_samples(config: DatasetConfig, *, loader: DatasetLoader | None = None) -> list[TextSample]:
    """Load and deterministically select normalized text samples."""
    dataset_loader = loader or load_dataset
    dataset = dataset_loader(
        config.path,
        config.subset,
        split=config.split,
        revision=config.revision,
        token=os.getenv("HF_TOKEN"),
    )
    columns = set(dataset.column_names)
    required_columns = {config.text_column}
    if config.id_column is not None:
        required_columns.add(config.id_column)
    for column in (
        config.reference_audio_column,
        config.reference_text_column,
        config.speaker_id_column,
    ):
        if column is not None:
            required_columns.add(column)
    missing_columns = sorted(required_columns - columns)
    if missing_columns:
        msg = f"dataset is missing configured columns: {', '.join(missing_columns)}"
        raise ConfigError(msg)

    indices = list(range(len(dataset)))
    if config.shuffle:
        random.Random(config.seed).shuffle(indices)
    if config.limit is not None:
        indices = indices[: config.limit]

    samples = [_normalize_row(dataset[index], index, config) for index in indices]
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        msg = "selected dataset rows contain duplicate sample IDs"
        raise ConfigError(msg)
    return samples


def _normalize_row(row: Mapping[str, Any], source_index: int, config: DatasetConfig) -> TextSample:
    text = row[config.text_column]
    if not isinstance(text, str) or not text.strip():
        msg = f"dataset row {source_index} has empty or non-string text"
        raise ConfigError(msg)
    if config.id_column is None:
        sample_id = f"row-{source_index:08d}"
    else:
        raw_id = row[config.id_column]
        if raw_id is None or not str(raw_id).strip():
            msg = f"dataset row {source_index} has an empty sample ID"
            raise ConfigError(msg)
        sample_id = str(raw_id).strip()
    return TextSample(
        sample_id=sample_id,
        text=text.strip(),
        source_index=source_index,
        reference_audio=_optional_row_string(row, config.reference_audio_column),
        reference_text=_optional_row_string(row, config.reference_text_column),
        speaker_id=_optional_row_string(row, config.speaker_id_column),
    )


def _optional_row_string(row: Mapping[str, Any], column: str | None) -> str | None:
    if column is None or row[column] is None:
        return None
    value = str(row[column]).strip()
    return value or None


def batches(items: Sequence[GenerationItem], batch_size: int) -> list[Sequence[GenerationItem]]:
    """Split generation items into stable batches."""
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]
