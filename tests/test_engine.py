"""Inference engine registry tests."""

from pathlib import Path

import pytest

from turkish_tts_generation.engine import EngineRegistry, NoopEngine, SubprocessEngine, create_default_registry


def test_register_and_create_engine() -> None:
    registry = EngineRegistry()
    registry.register("noop", NoopEngine)

    assert isinstance(registry.create("NOOP"), NoopEngine)
    assert registry.names == ("noop",)


def test_reject_duplicate_and_unknown_engines() -> None:
    registry = EngineRegistry()
    registry.register("noop", NoopEngine)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("noop", NoopEngine)
    with pytest.raises(ValueError, match="unknown engine"):
        registry.create("missing")


def test_default_registry_has_every_architecture() -> None:
    registry = create_default_registry()

    assert set(registry.names) == {
        "chatterbox",
        "f5-tts",
        "freya",
        "moss-tts",
        "noop",
        "omnivoice",
        "supertonic",
        "voxcpm",
        "xtts",
        "fish-speech",
    }
    assert isinstance(registry.create("voxcpm"), SubprocessEngine)


def test_subprocess_engine_uses_runtime_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TTS_RUNTIME_ROOT", str(tmp_path))
    expected = tmp_path / "voxcpm" / ".venv" / "bin" / "python"
    expected.parent.mkdir(parents=True)
    expected.touch()

    assert SubprocessEngine("voxcpm")._python_executable() == expected
