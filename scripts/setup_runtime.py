"""Create one isolated uv environment for an inference engine."""

import argparse
import shutil
import subprocess
from pathlib import Path

from turkish_tts_generation.runtime_lock import FREYA_SOURCE_REVISION, FREYA_SOURCE_URL, RUNTIME_REQUIREMENTS


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
            _install_freya_source(root, python)
        if not lock_file.is_file():
            frozen = subprocess.run(
                ("uv", "pip", "freeze", "--python", str(python)),
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            temporary = lock_file.with_suffix(".txt.tmp")
            source_header = f"# Freya source: {FREYA_SOURCE_URL}@{FREYA_SOURCE_REVISION}\n" if engine == "freya" else ""
            temporary.write_text(source_header + frozen, encoding="utf-8")
            temporary.replace(lock_file)
        print(f"Ready: {engine} ({python})")


def _install_freya_source(root: Path, python: Path) -> None:
    source = root / "FreyaTTS"
    if not source.is_dir():
        subprocess.run(("git", "clone", FREYA_SOURCE_URL, str(source)), check=True)
    subprocess.run(("git", "checkout", "--detach", FREYA_SOURCE_REVISION), cwd=source, check=True)
    site_packages = subprocess.run(
        (str(python), "-c", "import site; print(site.getsitepackages()[0])"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    Path(site_packages, "freya_tts_source.pth").write_text(str(source.resolve()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {type(error).__name__}: {error}")
        raise SystemExit(2) from None
