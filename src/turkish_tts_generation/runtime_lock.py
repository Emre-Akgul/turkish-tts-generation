"""Pinned top-level dependencies for isolated inference runtimes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

RUNTIME_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "voxcpm": ("voxcpm==2.0.3", "soundfile==0.14.0"),
    "chatterbox": ("chatterbox-tts==0.1.7",),
    "f5-tts": ("f5-tts==1.1.22", "soundfile==0.14.0"),
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
}

FREYA_SOURCE_URL = "https://github.com/freyavoiceai/FreyaTTS.git"
FREYA_SOURCE_REVISION = "146d36c1cb6660646be57d31339db4eed9315de3"


def runtime_lock_sha256(engine: str) -> str:
    """Return a stable identity for one engine's pinned requirements."""
    project_root = Path(__file__).resolve().parents[2]
    lock_root = Path(os.getenv("TTS_RUNTIME_LOCK_ROOT", project_root / "runtime-locks")).expanduser()
    lock_file = lock_root / f"{engine}.txt"
    if lock_file.is_file():
        return hashlib.sha256(lock_file.read_bytes()).hexdigest()
    content = json.dumps(RUNTIME_REQUIREMENTS[engine], separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()
