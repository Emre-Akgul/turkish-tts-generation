"""Download and verify selected pinned checkpoints without installing engines."""

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


@dataclass(frozen=True)
class Download:
    key: str
    repo_id: str
    directory: str
    revision: str
    required: tuple[str, ...]
    allow: tuple[str, ...] | None = None
    ignore: tuple[str, ...] | None = None


DOWNLOADS = (
    Download(
        "trendyol-tts",
        "Trendyol/Trendyol-TTS",
        "trendyol-tts",
        "66a80184b286390800ee7c1a95228cc839cd59ef",
        ("model.safetensors", "audiovae.pth"),
        ignore=("lora_adapter/**",),
    ),
    Download(
        "chatterbox-multilingual-v3",
        "ResembleAI/chatterbox",
        "chatterbox-multilingual-v3",
        "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
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
    Download(
        "voxcpm2",
        "openbmb/VoxCPM2",
        "voxcpm2",
        "32279effe8c19989596f05d353d1447f51d9e915",
        ("model.safetensors", "audiovae.pth"),
    ),
    Download(
        "orkhon-tts",
        "hcsolakoglu/Orkhon-TTS",
        "orkhon-tts",
        "9c29a360503ccf26a53d3ae7a1c827d3da98d595",
        ("orkhon_tts.pt", "vocab.txt"),
    ),
    Download(
        "vocos-mel-24khz",
        "charactr/vocos-mel-24khz",
        "vocos-mel-24khz",
        "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21",
        ("pytorch_model.bin", "config.yaml"),
    ),
    Download(
        "moss-tts-nano-100m",
        "OpenMOSS-Team/MOSS-TTS-Nano-100M",
        "moss-tts-nano-100m",
        "44502f80dbf9743528fa921cc544d662c685ebec",
        ("pytorch_model.bin",),
    ),
    Download(
        "moss-audio-tokenizer-nano",
        "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        "moss-audio-tokenizer-nano",
        "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68",
        ("model-00001-of-00001.safetensors", "model.safetensors.index.json"),
    ),
    Download(
        "supertonic-3",
        "Supertone/supertonic-3",
        "supertonic-3",
        "3cadd1ee6394adea1bd021217a0e650ede09a323",
        ("onnx/tts.json",),
    ),
    Download(
        "xtts-v2",
        "coqui/XTTS-v2",
        "xtts-v2",
        "6c2b0d75eae4b7047358e3b6bd9325f857d43f77",
        ("model.pth", "config.json", "samples/tr_sample.wav"),
    ),
    Download(
        "omnivoice",
        "k2-fsa/OmniVoice",
        "omnivoice",
        "c5fdb5ccb189668d56333f77ba2629f4cd7535f4",
        ("model.safetensors", "audio_tokenizer/model.safetensors"),
    ),
    Download(
        "freya-tts",
        "freyavoice/Freya-TTS",
        "freya-tts",
        "d124e07493615208f58bdd21d432736849ee4230",
        ("model.safetensors", "config.json"),
    ),
    Download(
        "xtts-reference",
        "coqui/XTTS-v2",
        "xtts-v2",
        "6c2b0d75eae4b7047358e3b6bd9325f857d43f77",
        ("samples/tr_sample.wav",),
        allow=("samples/tr_sample.wav",),
    ),
    Download(
        "voxcpm2-audiovae",
        "openbmb/VoxCPM2",
        "voxcpm2",
        "32279effe8c19989596f05d353d1447f51d9e915",
        ("audiovae.pth",),
        allow=("audiovae.pth",),
    ),
)

PRIMARY_KEYS = {
    "trendyol-tts",
    "chatterbox-multilingual-v3",
    "voxcpm2",
    "orkhon-tts",
    "moss-tts-nano-100m",
    "supertonic-3",
    "xtts-v2",
    "omnivoice",
    "freya-tts",
}
DEPENDENCIES = {
    "orkhon-tts": ("vocos-mel-24khz", "xtts-reference"),
    "moss-tts-nano-100m": ("moss-audio-tokenizer-nano", "xtts-reference"),
    "freya-tts": ("voxcpm2-audiovae",),
}


def _selected(keys: list[str]) -> tuple[Download, ...]:
    selected_keys = set(keys or PRIMARY_KEYS)
    for key in tuple(selected_keys):
        selected_keys.update(DEPENDENCIES.get(key, ()))
    if "xtts-v2" in selected_keys:
        selected_keys.discard("xtts-reference")
    if "voxcpm2" in selected_keys:
        selected_keys.discard("voxcpm2-audiovae")
    return tuple(download for download in DOWNLOADS if download.key in selected_keys)


def _snapshot_bytes(download: Download) -> int:
    info = HfApi().model_info(download.repo_id, revision=download.revision, files_metadata=True)
    return sum(int(sibling.size or 0) for sibling in info.siblings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=sorted(PRIMARY_KEYS), default=[])
    parser.add_argument("--headroom-gib", type=float, default=8.0)
    args = parser.parse_args()
    model_root = args.model_root.expanduser().resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    downloads = _selected(args.model)
    required = sum(_snapshot_bytes(download) for download in downloads) + int(args.headroom_gib * 1024**3)
    available = shutil.disk_usage(model_root).free
    if available < required:
        raise RuntimeError(f"insufficient space: need {required / 1024**3:.1f} GiB, have {available / 1024**3:.1f} GiB")
    for download in downloads:
        destination = model_root / download.directory
        print(f"Downloading {download.repo_id} -> {destination}", flush=True)
        snapshot_download(
            repo_id=download.repo_id,
            revision=download.revision,
            local_dir=destination,
            allow_patterns=download.allow,
            ignore_patterns=download.ignore,
        )
        missing = [name for name in download.required if not (destination / name).is_file()]
        if missing:
            raise RuntimeError(f"{download.repo_id} is missing: {', '.join(missing)}")
    print(f"All {len(downloads)} repositories are ready in {model_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - CLI boundary reports third-party network failures cleanly.
        print(f"error: {type(error).__name__}: {error}")
        raise SystemExit(2) from None
