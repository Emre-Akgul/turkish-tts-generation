"""Typed generation-job configuration loaded from YAML."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml


class ConfigError(ValueError):
    """Raised when a generation configuration is invalid."""


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Hugging Face dataset source and deterministic selection settings."""

    path: str
    text_column: str
    subset: str | None = None
    revision: str | None = None
    split: str = "train"
    id_column: str | None = None
    reference_audio_column: str | None = None
    reference_text_column: str | None = None
    speaker_id_column: str | None = None
    shuffle: bool = False
    seed: int = 42
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class TargetConfig:
    """One named inference-engine and model combination."""

    name: str
    engine: str
    model_id: str
    batch_size: int = 1
    device: str = "auto"
    dtype: str = "auto"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Artifact and manifest output settings."""

    root: Path = Path("outputs")
    run_name: str = "default"
    audio_format: str = "wav"
    resume: bool = True


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Complete generation job configuration."""

    dataset: DatasetConfig
    targets: tuple[TargetConfig, ...]
    output: OutputConfig


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"Expected a mapping at {location}"
        raise ConfigError(msg)
    return cast("dict[str, Any]", value)


def _required_string(data: dict[str, Any], key: str, location: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"{location}.{key} must be a non-empty string"
        raise ConfigError(msg)
    return value.strip()


def _optional_string(data: dict[str, Any], key: str, location: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        msg = f"{location}.{key} must be null or a non-empty string"
        raise ConfigError(msg)
    return value.strip()


def _path_component(value: str, location: str) -> str:
    if value in {".", ".."} or "/" in value or "\\" in value:
        msg = f"{location} must be a single path-safe name"
        raise ConfigError(msg)
    return value


def _parse_dataset(value: object) -> DatasetConfig:
    data = _mapping(value, "dataset")
    limit = data.get("limit")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        msg = "dataset.limit must be null or a positive integer"
        raise ConfigError(msg)
    seed = data.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool):
        msg = "dataset.seed must be an integer"
        raise ConfigError(msg)
    shuffle = data.get("shuffle", False)
    if not isinstance(shuffle, bool):
        msg = "dataset.shuffle must be a boolean"
        raise ConfigError(msg)
    return DatasetConfig(
        path=_required_string(data, "path", "dataset"),
        subset=_optional_string(data, "subset", "dataset"),
        revision=_optional_string(data, "revision", "dataset"),
        split=_required_string(data, "split", "dataset"),
        text_column=_required_string(data, "text_column", "dataset"),
        id_column=_optional_string(data, "id_column", "dataset"),
        reference_audio_column=_optional_string(data, "reference_audio_column", "dataset"),
        reference_text_column=_optional_string(data, "reference_text_column", "dataset"),
        speaker_id_column=_optional_string(data, "speaker_id_column", "dataset"),
        shuffle=shuffle,
        seed=seed,
        limit=limit,
    )


def _parse_target(value: object, index: int) -> TargetConfig:
    location = f"targets[{index}]"
    data = _mapping(value, location)
    batch_size = data.get("batch_size", 1)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        msg = f"{location}.batch_size must be a positive integer"
        raise ConfigError(msg)
    options = data.get("options", {})
    if not isinstance(options, dict):
        msg = f"{location}.options must be a mapping"
        raise ConfigError(msg)
    return TargetConfig(
        name=_path_component(_required_string(data, "name", location), f"{location}.name"),
        engine=_required_string(data, "engine", location),
        model_id=_required_string(data, "model_id", location),
        batch_size=batch_size,
        device=_required_string(data, "device", location) if "device" in data else "auto",
        dtype=_required_string(data, "dtype", location) if "dtype" in data else "auto",
        options=options,
    )


def _parse_output(value: object) -> OutputConfig:
    data = _mapping(value, "output")
    root = data.get("root", "outputs")
    if not isinstance(root, str) or not root.strip():
        msg = "output.root must be a non-empty path string"
        raise ConfigError(msg)
    resume = data.get("resume", True)
    if not isinstance(resume, bool):
        msg = "output.resume must be a boolean"
        raise ConfigError(msg)
    audio_format = _required_string(data, "audio_format", "output") if "audio_format" in data else "wav"
    normalized_format = audio_format.removeprefix(".")
    if not normalized_format or any(character in normalized_format for character in "/\\"):
        msg = "output.audio_format must be a file extension, not a path"
        raise ConfigError(msg)
    return OutputConfig(
        root=Path(root),
        run_name=_path_component(_required_string(data, "run_name", "output"), "output.run_name"),
        audio_format=normalized_format,
        resume=resume,
    )


def load_config(path: Path) -> GenerationConfig:
    """Load and validate a generation job from YAML."""
    with path.open(encoding="utf-8") as file:
        content = yaml.safe_load(file)
    data = _mapping(content, "root")
    targets_value = data.get("targets")
    if not isinstance(targets_value, list) or not targets_value:
        msg = "targets must be a non-empty list"
        raise ConfigError(msg)
    targets = tuple(_parse_target(value, index) for index, value in enumerate(targets_value))
    names = [target.name for target in targets]
    if len(names) != len(set(names)):
        msg = "target names must be unique"
        raise ConfigError(msg)
    return GenerationConfig(
        dataset=_parse_dataset(data.get("dataset")),
        targets=targets,
        output=_parse_output(data.get("output")),
    )
