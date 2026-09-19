"""Pinned top-level dependencies for isolated inference runtimes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

RUNTIME_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "voxcpm": ("voxcpm==2.0.3", "soundfile==0.14.0"),
    "chatterbox": ("chatterbox-tts==0.1.7", "setuptools<81"),
    "f5-tts": ("f5-tts==1.1.22", "torch==2.5.1", "torchaudio==2.5.1", "soundfile==0.14.0", "datasets>=4.0"),
    "moss-tts": (
        "git+https://github.com/OpenMOSS/MOSS-TTS-Nano.git@cc7bdf19c7639c0870dab22045a33b442760f6be",
        "soundfile==0.14.0",
    ),
    "supertonic": ("supertonic==1.3.1",),
    "xtts": (
        "TTS==0.22.0",
        "transformers==4.43.4",
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "soundfile==0.14.0",
    ),
    "omnivoice": (
        "git+https://github.com/k2-fsa/OmniVoice.git@38e992bc60f85548faeb77e8fa70158ba71deb30",
        "soundfile==0.14.0",
    ),
    "freya": (
        "torch",
        "numpy",
        "einops",
        "soundfile==0.14.0",
        "librosa",
        "huggingface_hub",
        "safetensors",
        "voxcpm==2.0.3",
    ),
    "fish-speech": (
        "git+https://github.com/fishaudio/fish-speech.git@e5e292632cb11e7a27b2b7487f58f612bc101e13",
        # Without an explicit floor, uv's resolver backtracks to the oldest
        # transformers (4.12.2) satisfying some unrelated constraint elsewhere in
        # the tree, which drags in tokenizers==0.10.3 -- a Rust extension with no
        # prebuilt wheel for modern Python that also fails to compile on any
        # single Rust toolchain (its code predates a lint now deny-by-default,
        # but its unpinned transitive deps need a newer Cargo than that implies).
        "transformers>=4.40,<4.58",
        "soundfile==0.14.0",
    ),
    "piper": ("piper-tts==1.8.0", "soundfile==0.14.0"),
    "mms-tts": (
        "transformers==4.43.4",
        "torch==2.5.1",
        "soundfile==0.14.0",
    ),
    "anka-tts": (
        "anka-tts[tts]==0.1.6",
        "f5-tts==1.1.22",
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "soundfile==0.14.0",
        "datasets>=4.0",
    ),
    "pocket-tts": ("pocket-tts[audio]==3.1.0", "pyyaml>=6.0"),
    "kani-tts": ("kani-tts==1.0.1", "soundfile==0.14.0"),
    # Higgs TTS 3 ships weights + config only (no direct from_pretrained inference
    # path); the model card's own AGENTS.md directs self-hosting through this
    # SGLang-Omni server, which exposes an OpenAI-compatible /v1/audio/speech API.
    "higgs": (
        "git+https://github.com/sgl-project/sglang-omni.git@ebd577ea0696a1510fdec8aad1ca8b44b5b3522f",
        "requests",
    ),
    "firered": (
        # flash_attn is installed separately below, after torch: its setup.py
        # imports torch at build time but doesn't declare it as a build
        # dependency, so it can't be resolved together with the rest in one
        # isolated-build install.
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "torchcodec==0.7.0",
        "transformers==5.6.2",
        "einops==0.8.2",
        "python-dotenv",
        "regex",
        "wetext",
        "fasttext",
        "faster-whisper",
        "soundfile==0.14.0",
    ),
    "moss-tts-v1.5": (
        "transformers>=5.0",
        "torch",
        "torchaudio",
        # Recent torchaudio delegates file loading to torchcodec; without it,
        # torchaudio.load() (used internally to encode the reference clip)
        # raises ImportError at the first real generate() call.
        "torchcodec",
        "einops",
        "soundfile==0.14.0",
    ),
}

FREYA_SOURCE_URL = "https://github.com/freyavoiceai/FreyaTTS.git"
FREYA_SOURCE_REVISION = "146d36c1cb6660646be57d31339db4eed9315de3"

FIRERED_SOURCE_URL = "https://github.com/FireRedTeam/FireRedTTS3.git"
FIRERED_SOURCE_REVISION = "7a1f3a7282ff184cc1c7f070556baaf5f08b5216"


def runtime_lock_sha256(engine: str) -> str:
    """Return a stable identity for one engine's pinned requirements."""
    project_root = Path(__file__).resolve().parents[2]
    lock_root = Path(os.getenv("TTS_RUNTIME_LOCK_ROOT", project_root / "runtime-locks")).expanduser()
    lock_file = lock_root / f"{engine}.txt"
    if lock_file.is_file():
        return hashlib.sha256(lock_file.read_bytes()).hexdigest()
    content = json.dumps(RUNTIME_REQUIREMENTS[engine], separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()
