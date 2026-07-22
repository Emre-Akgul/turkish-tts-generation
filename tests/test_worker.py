"""Dependency-free worker helper tests."""

from turkish_tts_generation.worker import _reference


def test_reference_audio_is_optional_and_prefers_explicit_values() -> None:
    options = {"_default_reference_audio": "/models/default.wav"}

    assert _reference({}, options) == "/models/default.wav"
    assert _reference({}, {**options, "reference_audio": "/target.wav"}) == "/target.wav"
    assert _reference({"reference_audio": "/sample.wav"}, options) == "/sample.wav"
