"""Supported model catalog tests."""

import pytest

from turkish_tts_generation.models import SUPPORTED_MODELS, resolve_model


def test_all_requested_models_are_registered() -> None:
    assert len(SUPPORTED_MODELS) == 9
    assert resolve_model("Trendyol/Trendyol-TTS").engine == "voxcpm"
    assert resolve_model("voxcpm2").engine == "voxcpm"
    assert len({model.engine for model in SUPPORTED_MODELS}) == 8
    assert resolve_model("orkhon-tts").default_reference_audio == "xtts-v2/samples/tr_sample.wav"
    assert resolve_model("moss-tts-nano-100m").default_reference_audio == "xtts-v2/samples/tr_sample.wav"
    assert resolve_model("xtts-v2").default_reference_audio == "xtts-v2/samples/tr_sample.wav"


def test_resolve_is_case_insensitive_and_rejects_unknown() -> None:
    assert resolve_model("FREYA-TTS").model_id == "freyavoice/Freya-TTS"
    with pytest.raises(ValueError, match="unsupported model"):
        resolve_model("missing")
