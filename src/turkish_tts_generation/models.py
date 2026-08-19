"""Supported model catalog and shared inference architectures."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """A supported checkpoint and the engine capable of loading it."""

    key: str
    model_id: str
    engine: str
    directory: str
    revision: str
    companion_directory: str | None = None
    companion_revision: str | None = None
    default_reference_audio: str | None = None


SUPPORTED_MODELS = (
    ModelDefinition(
        "trendyol-tts", "Trendyol/Trendyol-TTS", "voxcpm", "trendyol-tts", "66a80184b286390800ee7c1a95228cc839cd59ef"
    ),
    ModelDefinition(
        "chatterbox-multilingual-v3",
        "ResembleAI/chatterbox",
        "chatterbox",
        "chatterbox-multilingual-v3",
        "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
    ),
    ModelDefinition("voxcpm2", "openbmb/VoxCPM2", "voxcpm", "voxcpm2", "32279effe8c19989596f05d353d1447f51d9e915"),
    ModelDefinition(
        "orkhon-tts",
        "hcsolakoglu/Orkhon-TTS",
        "f5-tts",
        "orkhon-tts",
        "9c29a360503ccf26a53d3ae7a1c827d3da98d595",
        companion_directory="vocos-mel-24khz",
        companion_revision="0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21",
        default_reference_audio="xtts-v2/samples/tr_sample.wav",
    ),
    ModelDefinition(
        "moss-tts-nano-100m",
        "OpenMOSS-Team/MOSS-TTS-Nano-100M",
        "moss-tts",
        "moss-tts-nano-100m",
        "44502f80dbf9743528fa921cc544d662c685ebec",
        companion_directory="moss-audio-tokenizer-nano",
        companion_revision="6aa02b01e445cc585582cf0ba480bc3ea6c8dd68",
        default_reference_audio="xtts-v2/samples/tr_sample.wav",
    ),
    ModelDefinition(
        "supertonic-3",
        "Supertone/supertonic-3",
        "supertonic",
        "supertonic-3",
        "3cadd1ee6394adea1bd021217a0e650ede09a323",
    ),
    ModelDefinition(
        "xtts-v2",
        "coqui/XTTS-v2",
        "xtts",
        "xtts-v2",
        "6c2b0d75eae4b7047358e3b6bd9325f857d43f77",
        default_reference_audio="xtts-v2/samples/tr_sample.wav",
    ),
    ModelDefinition(
        "omnivoice", "k2-fsa/OmniVoice", "omnivoice", "omnivoice", "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
    ),
    ModelDefinition(
        "freya-tts",
        "freyavoice/Freya-TTS",
        "freya",
        "freya-tts",
        "d124e07493615208f58bdd21d432736849ee4230",
        companion_directory="voxcpm2",
        companion_revision="32279effe8c19989596f05d353d1447f51d9e915",
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
