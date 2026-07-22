"""Create one isolated uv environment for an inference engine."""

import argparse
import subprocess
from pathlib import Path

DEPENDENCIES = {
    "voxcpm": ("voxcpm", "soundfile"),
    "chatterbox": ("chatterbox-tts",),
    "f5-tts": ("f5-tts", "soundfile"),
    "moss-tts": ("git+https://github.com/OpenMOSS/MOSS-TTS-Nano.git", "soundfile"),
    "supertonic": ("supertonic",),
    "xtts": ("TTS==0.22.0", "soundfile"),
    "omnivoice": ("git+https://github.com/k2-fsa/OmniVoice.git", "soundfile"),
    "freya": ("torch", "numpy", "einops", "soundfile", "librosa", "huggingface_hub", "safetensors", "voxcpm"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=(*DEPENDENCIES, "all"))
    args = parser.parse_args()
    engines = DEPENDENCIES if args.engine == "all" else (args.engine,)
    for engine in engines:
        root = Path(".runtimes") / engine
        python = root / ".venv" / "bin" / "python"
        root.mkdir(parents=True, exist_ok=True)
        if not python.is_file():
            subprocess.run(("uv", "venv", str(root / ".venv"), "--python", "3.11"), check=True)
        subprocess.run(("uv", "pip", "install", "--python", str(python), *DEPENDENCIES[engine]), check=True)
        if engine == "freya":
            _install_freya_source(root, python)
        print(f"Ready: {engine} ({python})")


def _install_freya_source(root: Path, python: Path) -> None:
    source = root / "Freya-TTS"
    if not source.is_dir():
        subprocess.run(
            ("git", "clone", "--depth", "1", "https://github.com/freyavoice/Freya-TTS.git", str(source)),
            check=True,
        )
    site_packages = subprocess.run(
        (str(python), "-c", "import site; print(site.getsitepackages()[0])"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    Path(site_packages, "freya_tts_source.pth").write_text(str(source.resolve()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
