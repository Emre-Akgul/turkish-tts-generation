"""Run one fully verified, recoverable TTS arena model stage."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from turkish_tts_generation.cleanup import cleanup_completed_assets
from turkish_tts_generation.config import load_config
from turkish_tts_generation.models import resolve_model

SMOKE_SAMPLE_IDS = (
    "tr-arena-v1-0001",
    "tr-arena-v1-0051",
    "tr-arena-v1-0073",
    "tr-arena-v1-0105",
    "tr-arena-v1-0121",
    "tr-arena-v1-0136",
    "tr-arena-v1-0157",
    "tr-arena-v1-0170",
    "tr-arena-v1-0181",
    "tr-arena-v1-0200",
    "tr-arena-v1-0217",
    "tr-arena-v1-0232",
)

CommandRunner = Callable[[Sequence[str], dict[str, str]], None]
CleanupRunner = Callable[..., list[Path]]


def _run(command: Sequence[str], environment: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)  # noqa: S603


def stage_target(
    config_path: Path,
    target_name: str,
    *,
    model_root: Path,
    runtime_root: Path,
    completed_targets: set[str] | None = None,
    lock_root: Path | None = None,
    headroom_gib: float = 8.0,
    cleanup: bool = True,
    normalize: bool = True,
    command_runner: CommandRunner = _run,
    cleanup_runner: CleanupRunner = cleanup_completed_assets,
) -> list[Path]:
    """Install, download, smoke, generate, optionally normalize, verify, and clean one target."""
    config = load_config(config_path)
    targets = {target.name: target for target in config.targets}
    try:
        target = targets[target_name]
    except KeyError as error:
        raise ValueError(f"unknown configured target: {target_name}") from error
    model = resolve_model(target.model_id)
    project_root = Path(__file__).resolve().parents[2]
    resolved_models = model_root.expanduser().resolve()
    resolved_runtimes = runtime_root.expanduser().resolve()
    resolved_locks = (lock_root or project_root / "runtime-locks").expanduser().resolve()
    resolved_models.mkdir(parents=True, exist_ok=True)
    resolved_runtimes.mkdir(parents=True, exist_ok=True)
    resolved_locks.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["TTS_MODEL_ROOT"] = str(resolved_models)
    environment["TTS_RUNTIME_ROOT"] = str(resolved_runtimes)
    environment["TTS_RUNTIME_LOCK_ROOT"] = str(resolved_locks)
    python = sys.executable
    command_runner(
        (
            python,
            str(project_root / "scripts" / "setup_runtime.py"),
            model.engine,
            "--runtime-root",
            str(resolved_runtimes),
            "--lock-root",
            str(resolved_locks),
            "--headroom-gib",
            str(headroom_gib),
        ),
        environment,
    )
    command_runner(
        (
            python,
            str(project_root / "scripts" / "download_models.py"),
            "--model-root",
            str(resolved_models),
            "--model",
            model.key,
            "--headroom-gib",
            str(headroom_gib),
        ),
        environment,
    )
    smoke_command = [
        python,
        "-m",
        "turkish_tts_generation",
        "--config",
        str(config_path),
        "--target",
        target_name,
    ]
    for sample_id in SMOKE_SAMPLE_IDS:
        smoke_command.extend(("--sample-id", sample_id))
    command_runner(tuple(smoke_command), environment)
    command_runner(
        (
            python,
            "-m",
            "turkish_tts_generation",
            "--config",
            str(config_path),
            "--target",
            target_name,
        ),
        environment,
    )
    if normalize:
        command_runner(
            (
                python,
                "-m",
                "turkish_tts_generation.arena_audio",
                "--config",
                str(config_path),
                "--target",
                target_name,
            ),
            environment,
        )
    run_root = config.output.root / config.output.run_name
    discovered = {
        configured.name
        for configured in config.targets
        if (run_root / configured.name / "arena-manifest.jsonl").is_file()
    }
    completed = set(completed_targets or ()) | discovered | {target_name}
    if not cleanup or not normalize:
        return []
    return cleanup_runner(
        config,
        completed,
        model_root=resolved_models,
        runtime_root=resolved_runtimes,
        expected_count=240,
    )


def main(args: list[str] | None = None) -> int:
    """Run a staged arena target from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/arena-v1.yaml"))
    parser.add_argument("--target", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--lock-root", type=Path)
    parser.add_argument("--completed-target", action="append", default=[])
    parser.add_argument("--headroom-gib", type=float, default=8.0)
    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument("--skip-normalize", action="store_true")
    options = parser.parse_args(args)
    try:
        removed = stage_target(
            options.config,
            options.target,
            model_root=options.model_root,
            runtime_root=options.runtime_root,
            completed_targets=set(options.completed_target),
            lock_root=options.lock_root,
            headroom_gib=options.headroom_gib,
            cleanup=not options.skip_cleanup,
            normalize=not options.skip_normalize,
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}")
        return 2
    for path in removed:
        print(f"removed={path}")
    print(f"completed={options.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
