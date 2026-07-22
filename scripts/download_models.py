"""Download and verify every supported checkpoint without installing engines."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class Download:
    repo_id: str
    directory: str
    required: tuple[str, ...]
    allow: tuple[str, ...] | None = None
    ignore: tuple[str, ...] | None = None


DOWNLOADS = (
    Download(
        "Trendyol/Trendyol-TTS", "trendyol-tts", ("model.safetensors", "audiovae.pth"), ignore=("lora_adapter/**",)
    ),
    Download(
        "ResembleAI/chatterbox",
        "chatterbox-multilingual-v3",
        ("t3_mtl23ls_v3.safetensors", "s3gen.pt", "ve.pt", "conds.pt"),
        allow=(
            "t3_mtl23ls_v3.safetensors",
            "s3gen.pt",
            "ve.pt",
            "conds.pt",
            "grapheme_mtl_merged_expanded_v1.json",
            "Cangjie5_TC.json",
        ),
    ),
    Download("openbmb/VoxCPM2", "voxcpm2", ("model.safetensors", "audiovae.pth")),
    Download("hcsolakoglu/Orkhon-TTS", "orkhon-tts", ("orkhon_tts.pt", "vocab.txt")),
    Download("charactr/vocos-mel-24khz", "vocos-mel-24khz", ("pytorch_model.bin", "config.yaml")),
    Download("OpenMOSS-Team/MOSS-TTS-Nano-100M", "moss-tts-nano-100m", ("pytorch_model.bin",)),
    Download(
        "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        "moss-audio-tokenizer-nano",
        ("model-00001-of-00001.safetensors", "model.safetensors.index.json"),
    ),
    Download("Supertone/supertonic-3", "supertonic-3", ("onnx/tts.json",)),
    Download("coqui/XTTS-v2", "xtts-v2", ("model.pth", "config.json")),
    Download("k2-fsa/OmniVoice", "omnivoice", ("model.safetensors", "audio_tokenizer/model.safetensors")),
    Download("freyavoice/Freya-TTS", "freya-tts", ("model.safetensors", "config.json")),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    args.model_root.mkdir(parents=True, exist_ok=True)
    for download in DOWNLOADS:
        destination = args.model_root / download.directory
        print(f"Downloading {download.repo_id} -> {destination}", flush=True)
        snapshot_download(
            repo_id=download.repo_id,
            local_dir=destination,
            allow_patterns=download.allow,
            ignore_patterns=download.ignore,
        )
        missing = [name for name in download.required if not (destination / name).is_file()]
        if missing:
            raise RuntimeError(f"{download.repo_id} is missing: {', '.join(missing)}")
    print(f"All {len(DOWNLOADS)} repositories are ready in {args.model_root}")


if __name__ == "__main__":
    main()
