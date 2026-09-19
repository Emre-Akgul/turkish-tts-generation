"""Create one isolated uv environment for an inference engine."""

import argparse
import shutil
import subprocess
from pathlib import Path

from turkish_tts_generation.runtime_lock import (
    FIRERED_SOURCE_REVISION,
    FIRERED_SOURCE_URL,
    FREYA_SOURCE_REVISION,
    FREYA_SOURCE_URL,
    RUNTIME_REQUIREMENTS,
)

# Packages forced to a specific version by a post-install fix below, applied every
# run regardless of the install source. Excluded from the frozen lock file: pinning
# them there too would make a later `uv pip install -r lockfile` resolve them
# together with the engine's own package, which conflicts with its own sub-pins.
POST_INSTALL_OVERRIDES = {
    "kani-tts": {"transformers", "nvidia-cudnn-cu13"},
    "pocket-tts": {"nvidia-cudnn-cu13"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=(*RUNTIME_REQUIREMENTS, "all"))
    parser.add_argument("--runtime-root", type=Path, default=Path(".runtimes"))
    parser.add_argument("--lock-root", type=Path, default=Path("runtime-locks"))
    parser.add_argument("--headroom-gib", type=float, default=8.0)
    args = parser.parse_args()
    engines = RUNTIME_REQUIREMENTS if args.engine == "all" else (args.engine,)
    runtime_root = args.runtime_root.expanduser().resolve()
    lock_root = args.lock_root.expanduser().resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    lock_root.mkdir(parents=True, exist_ok=True)
    for engine in engines:
        available = shutil.disk_usage(runtime_root).free
        required = int(args.headroom_gib * 1024**3)
        if available < required:
            raise RuntimeError(
                f"insufficient runtime headroom: need {required / 1024**3:.1f} GiB, have {available / 1024**3:.1f} GiB"
            )
        root = runtime_root / engine
        python = root / ".venv" / "bin" / "python"
        root.mkdir(parents=True, exist_ok=True)
        if not python.is_file():
            subprocess.run(("uv", "venv", str(root / ".venv"), "--python", "3.11"), check=True)
        lock_file = lock_root / f"{engine}.txt"
        requirements = ("-r", str(lock_file)) if lock_file.is_file() else RUNTIME_REQUIREMENTS[engine]
        subprocess.run(("uv", "pip", "install", "--python", str(python), *requirements), check=True)
        if engine == "freya":
            _install_source(root, python, FREYA_SOURCE_URL, FREYA_SOURCE_REVISION, "FreyaTTS", "freya_tts_source.pth")
        if engine == "firered":
            _install_source(
                root, python, FIRERED_SOURCE_URL, FIRERED_SOURCE_REVISION, "FireRedTTS3", "firered_tts3_source.pth"
            )
        if engine == "kani-tts":
            _fix_kani_tts_transformers(python)
        if engine in ("kani-tts", "pocket-tts"):
            _fix_broken_cudnn(python)
        if not lock_file.is_file():
            frozen = subprocess.run(
                ("uv", "pip", "freeze", "--python", str(python)),
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            excluded = POST_INSTALL_OVERRIDES.get(engine, set())
            if excluded:
                frozen = "\n".join(
                    line for line in frozen.splitlines() if line.split("==")[0].strip() not in excluded
                )
                frozen += "\n"
            temporary = lock_file.with_suffix(".txt.tmp")
            source_header = ""
            if engine == "freya":
                source_header = f"# Freya source: {FREYA_SOURCE_URL}@{FREYA_SOURCE_REVISION}\n"
            elif engine == "firered":
                source_header = f"# FireRedTTS3 source: {FIRERED_SOURCE_URL}@{FIRERED_SOURCE_REVISION}\n"
            temporary.write_text(source_header + frozen, encoding="utf-8")
            temporary.replace(lock_file)
        print(f"Ready: {engine} ({python})")


def _fix_kani_tts_transformers(python: Path) -> None:
    # nemo-toolkit==2.4.0 pins transformers==4.51.3, but kani-tts's own LFM2 model
    # code needs transformers.utils.TransformersKwargs and LFM2 support, both added
    # later. Force a newer transformers without re-resolving nemo's other pins.
    subprocess.run(
        ("uv", "pip", "install", "--python", str(python), "--no-deps", "transformers==4.55.0"),
        check=True,
    )


def _fix_broken_cudnn(python: Path) -> None:
    # torch==2.14.0 exact-pins nvidia-cudnn-cu13==9.24.0.43, which fails to
    # initialize on this machine's driver (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
    # from cudnnFinalize on the very first conv). 9.20.0.48, used by other engines
    # here, loads and runs fine. Force it without re-resolving torch's other pins.
    subprocess.run(
        ("uv", "pip", "install", "--python", str(python), "--no-deps", "nvidia-cudnn-cu13==9.20.0.48"),
        check=True,
    )


def _install_source(root: Path, python: Path, url: str, revision: str, directory_name: str, pth_name: str) -> None:
    # For source repos with no setup.py/pyproject.toml: clone and add a .pth file
    # so the package imports directly from the checkout instead of site-packages.
    source = root / directory_name
    if not source.is_dir():
        subprocess.run(("git", "clone", url, str(source)), check=True)
    subprocess.run(("git", "checkout", "--detach", revision), cwd=source, check=True)
    site_packages = subprocess.run(
        (str(python), "-c", "import site; print(site.getsitepackages()[0])"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    Path(site_packages, pth_name).write_text(str(source.resolve()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {type(error).__name__}: {error}")
        raise SystemExit(2) from None
