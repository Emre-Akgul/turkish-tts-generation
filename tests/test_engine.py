"""Inference engine registry tests."""

import pytest

from turkish_tts_generation.engine import EngineRegistry, NoopEngine


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
