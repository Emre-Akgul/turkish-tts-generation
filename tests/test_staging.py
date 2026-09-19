"""One-model staged arena runner tests."""

from pathlib import Path
from typing import Any

import pytest

from turkish_tts_generation.staging import SMOKE_SAMPLE_IDS, stage_target

CONFIG = Path(__file__).parents[1] / "configs" / "arena-v1.yaml"


def test_stage_builds_smoke_full_and_cleanup_commands(tmp_path: Path) -> None:
    commands: list[tuple[str, ...]] = []
    cleanup_calls: list[tuple[set[str], dict[str, Any]]] = []

    def command_runner(command: Any, _environment: dict[str, str]) -> None:
        commands.append(tuple(command))

    def cleanup_runner(_config: Any, completed: set[str], **kwargs: Any) -> list[Path]:
        cleanup_calls.append((completed, kwargs))
        return []

    stage_target(
        CONFIG,
        "supertonic-3",
        model_root=tmp_path / "models",
        runtime_root=tmp_path / "runtimes",
        command_runner=command_runner,
        cleanup_runner=cleanup_runner,
    )

    assert len(commands) == 4
    assert commands[0][1].endswith("setup_runtime.py")
    assert "supertonic-3" in commands[1]
    assert commands[2].count("--sample-id") == len(SMOKE_SAMPLE_IDS)
    # discovered folds in any other target whose manifest.jsonl already exists
    # under the real configured output root, so this only asserts a lower bound.
    assert cleanup_calls[0][0] >= {"supertonic-3"}


def test_stage_skips_cleanup_when_requested(tmp_path: Path) -> None:
    commands: list[tuple[str, ...]] = []

    def command_runner(command: Any, _environment: dict[str, str]) -> None:
        commands.append(tuple(command))

    def cleanup_runner(*_args: Any, **_kwargs: Any) -> list[Path]:
        pytest.fail("cleanup must not run when cleanup is skipped")

    removed = stage_target(
        CONFIG,
        "supertonic-3",
        model_root=tmp_path / "models",
        runtime_root=tmp_path / "runtimes",
        cleanup=False,
        command_runner=command_runner,
        cleanup_runner=cleanup_runner,
    )

    assert len(commands) == 4
    assert removed == []


def test_stage_rejects_unknown_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown configured target"):
        stage_target(CONFIG, "missing", model_root=tmp_path / "models", runtime_root=tmp_path / "runtimes")
