"""Supported model catalog tests."""

import pytest

from turkish_tts_generation.models import SUPPORTED_MODELS, resolve_model


def test_all_requested_models_are_registered() -> None:
    assert len(SUPPORTED_MODELS) == 19
    assert resolve_model("Trendyol/Trendyol-TTS").engine == "voxcpm"
    assert resolve_model("voxcpm2").engine == "voxcpm"
    assert len({model.engine for model in SUPPORTED_MODELS}) == 17
    assert resolve_model("s2-pro").engine == "fish-speech"
    assert resolve_model("orkhon-tts").default_reference_audio == "xtts-v2/samples/tr_sample.wav"
    assert resolve_model("moss-tts-nano-100m").default_reference_audio == "xtts-v2/samples/tr_sample.wav"
    assert resolve_model("xtts-v2").default_reference_audio == "xtts-v2/samples/tr_sample.wav"
    assert resolve_model("piper-tr-dfki").engine == "piper"
    assert resolve_model("mms-tts-tur").engine == "mms-tts"
    assert resolve_model("anka-tts-v0.1").engine == "anka-tts"
    assert resolve_model("kizagan-tts-v1").engine == "voxcpm"
    assert resolve_model("pocket-tts-tr").engine == "pocket-tts"
    assert resolve_model("pocket-tts-tr").default_reference_audio == "xtts-v2/samples/tr_sample.wav"
    assert resolve_model("kani-tts-400m-0.3-tr").engine == "kani-tts"
    assert resolve_model("higgs-tts-3-4b").engine == "higgs"
    assert resolve_model("firered-tts3").engine == "firered"
    assert resolve_model("moss-tts-local-v1.5").engine == "moss-tts-v1.5"
    assert all(len(model.revision) == 40 for model in SUPPORTED_MODELS)


def test_resolve_is_case_insensitive_and_rejects_unknown() -> None:
    assert resolve_model("FREYA-TTS").model_id == "freyavoice/Freya-TTS"
    with pytest.raises(ValueError, match="unsupported model"):
        resolve_model("missing")
