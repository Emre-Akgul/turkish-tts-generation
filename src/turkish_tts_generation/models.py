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
    ModelDefinition(
        "s2-pro",
        "fishaudio/s2-pro",
        "fish-speech",
        "s2-pro",
        "1de9996b6be38b745688de084d87a5633f714e4e",
        default_reference_audio="xtts-v2/samples/tr_sample.wav",
    ),
    ModelDefinition(
        "piper-tr-dfki",
        "rhasspy/piper-voices",
        "piper",
        "piper-tr-dfki",
        "8914c16824264dfe6425deffca679ce9bb1ab371",
    ),
    ModelDefinition(
        "mms-tts-tur",
        "facebook/mms-tts-tur",
        "mms-tts",
        "mms-tts-tur",
        "7e364479c307f06733ca865b0a5269e0209cf82d",
    ),
    ModelDefinition(
        "anka-tts-v0.1",
        "krmkayabasi/Anka-TTS",
        "anka-tts",
        "anka-tts",
        "f1ce92d4eeb02ab0d57b537493b6dd365b24607b",
    ),
    ModelDefinition(
        "kizagan-tts-v1",
        "AlicanKiraz0/Kizagan-TTS-v1.0",
        "voxcpm",
        "kizagan-tts-v1",
        "a22aac87ee3d84c3757f06efec5cfa08f426e02b",
    ),
    ModelDefinition(
        "pocket-tts-tr",
        "kaanhgunay/pocket-tts-tr",
        "pocket-tts",
        "pocket-tts-tr",
        "e5aa490d9aa6075cf047e4286fb57cecb99aa1ed",
        default_reference_audio="xtts-v2/samples/tr_sample.wav",
    ),
    ModelDefinition(
        "kani-tts-400m-0.3-tr",
        "Anilosan15/kani-tts-400m-0.3-tr",
        "kani-tts",
        "kani-tts-400m-0.3-tr",
        "a8b45f2cbb094a0c7b174f0e4e46caa2acfb6294",
    ),
    ModelDefinition(
        "higgs-tts-3-4b",
        "bosonai/higgs-tts-3-4b",
        "higgs",
        "higgs-tts-3-4b",
        "239f63fb7b02b1aa085f98d9efae5e35cc5523e8",
        default_reference_audio="xtts-v2/samples/tr_sample.wav",
    ),
    ModelDefinition(
        "firered-tts3",
        "FireRedTeam/FireRedTTS3",
        "firered",
        "firered-tts3",
        "dcf1bdcd1b8b25b382fa84c3e34eb82e3054a610",
        default_reference_audio="xtts-v2/samples/tr_sample.wav",
    ),
    ModelDefinition(
        "moss-tts-local-v1.5",
        "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        "moss-tts-v1.5",
        "moss-tts-local-v1.5",
        "be7766a6735b98bd793f7c79fb720b4d0f5d13b8",
        companion_directory="moss-audio-tokenizer-v2",
        companion_revision="f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169",
        default_reference_audio="xtts-v2/samples/tr_sample.wav",
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
