"""Supported model catalog and shared inference architectures."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """A supported checkpoint and the engine capable of loading it."""

    key: str
    model_id: str
    engine: str
    directory: str
    companion_directory: str | None = None
    default_reference_audio: str | None = None


SUPPORTED_MODELS = (
    ModelDefinition("trendyol-tts", "Trendyol/Trendyol-TTS", "voxcpm", "trendyol-tts"),
    ModelDefinition(
        "chatterbox-multilingual-v3",
        "ResembleAI/chatterbox",
        "chatterbox",
        "chatterbox-multilingual-v3",
    ),
    ModelDefinition("voxcpm2", "openbmb/VoxCPM2", "voxcpm", "voxcpm2"),
    ModelDefinition(
        "orkhon-tts",
        "hcsolakoglu/Orkhon-TTS",
        "f5-tts",
        "orkhon-tts",
        companion_directory="vocos-mel-24khz",
        default_reference_audio="xtts-v2/samples/tr_sample.wav",
    ),
    ModelDefinition(
        "moss-tts-nano-100m",
        "OpenMOSS-Team/MOSS-TTS-Nano-100M",
        "moss-tts",
        "moss-tts-nano-100m",
        companion_directory="moss-audio-tokenizer-nano",
        default_reference_audio="xtts-v2/samples/tr_sample.wav",
    ),
    ModelDefinition("supertonic-3", "Supertone/supertonic-3", "supertonic", "supertonic-3"),
    ModelDefinition(
        "xtts-v2",
        "coqui/XTTS-v2",
        "xtts",
        "xtts-v2",
        default_reference_audio="xtts-v2/samples/tr_sample.wav",
    ),
    ModelDefinition("omnivoice", "k2-fsa/OmniVoice", "omnivoice", "omnivoice"),
    ModelDefinition(
        "freya-tts",
        "freyavoice/Freya-TTS",
        "freya",
        "freya-tts",
        companion_directory="voxcpm2",
    ),
)


def resolve_model(value: str) -> ModelDefinition:
    """Resolve a short key or Hugging Face repository ID."""
    normalized = value.casefold()
    for model in SUPPORTED_MODELS:
        if normalized in {model.key.casefold(), model.model_id.casefold()}:
            return model
    supported = ", ".join(model.key for model in SUPPORTED_MODELS)
    raise ValueError(f"unsupported model '{value}' (supported: {supported})")
