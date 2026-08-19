"""Verified cleanup of recoverable model and runtime staging assets."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from turkish_tts_generation.arena_audio import sha256_file
from turkish_tts_generation.config import GenerationConfig, load_config
from turkish_tts_generation.models import resolve_model

RETAINED_DEPENDENCY_FILES = {
    "voxcpm2": {"audiovae.pth"},
    "xtts-v2": {"samples/tr_sample.wav"},
}


def _direct_child(path: Path, root: Path) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved_path = path.expanduser().resolve()
    if resolved_path == resolved_root or resolved_path.parent != resolved_root:
        raise ValueError(f"cleanup path must be a direct child of staging root: {resolved_path}")
    return resolved_path


def safe_remove_staged_directory(path: Path, root: Path, *, verified: bool, still_required: bool) -> bool:
    """Remove an explicit recoverable staging directory after safety gates pass."""
    resolved = _direct_child(path, root)
    if not verified:
        raise ValueError("cleanup requires verified output manifests")
    if still_required:
        raise ValueError(f"staging asset is still required: {resolved.name}")
    if not resolved.exists():
        return False
    if resolved.is_symlink() or not resolved.is_dir():
        raise ValueError(f"cleanup target must be a real directory: {resolved}")
    shutil.rmtree(resolved)
    return True


def verify_target_completion(config: GenerationConfig, target_name: str, *, expected_count: int = 240) -> None:
    """Verify every retained raw and arena asset before allowing cleanup."""
    run_root = (config.output.root / config.output.run_name).resolve()
    manifest = run_root / target_name / "arena-manifest.jsonl"
    if not manifest.is_file():
        raise ValueError(f"arena completion manifest is missing: {manifest}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_count or len({row.get("sample_id") for row in rows}) != expected_count:
        raise ValueError(f"arena manifest for {target_name} must contain {expected_count} unique samples")
    for row in rows:
        if row.get("target_name") != target_name:
            raise ValueError(f"arena manifest contains the wrong target: {target_name}")
        for path_key, hash_key in (("raw_path", "raw_sha256"), ("arena_path", "normalized_sha256")):
            asset = (run_root / str(row[path_key])).resolve()
            try:
                asset.relative_to(run_root)
            except ValueError as error:
                raise ValueError(f"manifest asset is outside run root: {asset}") from error
            if not asset.is_file() or sha256_file(asset) != row.get(hash_key):
                raise ValueError(f"manifest asset is missing or changed: {asset}")


def _requirements(config: GenerationConfig) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    model_requirements: dict[str, set[str]] = defaultdict(set)
    runtime_requirements: dict[str, set[str]] = defaultdict(set)
    owners: dict[str, str] = {}
    for target in config.targets:
        model = resolve_model(target.model_id)
        owners[model.directory] = target.name
        model_requirements[model.directory].add(target.name)
        if model.companion_directory:
            model_requirements[model.companion_directory].add(target.name)
        if model.default_reference_audio:
            model_requirements[Path(model.default_reference_audio).parts[0]].add(target.name)
        runtime_requirements[model.engine].add(target.name)
    return dict(model_requirements), dict(runtime_requirements), owners


def _prune_to_retained_files(directory: Path, retained: set[str]) -> None:
    keep = {(directory / relative).resolve() for relative in retained}
    missing = [path for path in keep if not path.is_file()]
    if missing:
        raise ValueError(f"required retained dependency is missing: {missing[0]}")
    for path in sorted(directory.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        resolved = path.resolve()
        if path.is_symlink() and resolved not in keep or path.is_file() and resolved not in keep:
            path.unlink()
        elif path.is_dir() and not any(candidate == resolved or resolved in candidate.parents for candidate in keep):
            path.rmdir()


def cleanup_completed_assets(
    config: GenerationConfig,
    completed_targets: set[str],
    *,
    model_root: Path,
    runtime_root: Path,
    expected_count: int = 240,
) -> list[Path]:
    """Clean all no-longer-needed staged assets after verifying completed targets."""
    configured = {target.name for target in config.targets}
    unknown = completed_targets - configured
    if unknown:
        raise ValueError(f"unknown completed targets: {', '.join(sorted(unknown))}")
    for target_name in completed_targets:
        verify_target_completion(config, target_name, expected_count=expected_count)

    model_requirements, runtime_requirements, owners = _requirements(config)
    removed: list[Path] = []
    for directory, required_by in model_requirements.items():
        path = _direct_child(model_root / directory, model_root)
        remaining = required_by - completed_targets
        if not remaining:
            if safe_remove_staged_directory(path, model_root, verified=True, still_required=False):
                removed.append(path)
        elif owners.get(directory) in completed_targets and directory in RETAINED_DEPENDENCY_FILES and path.is_dir():
            _prune_to_retained_files(path, RETAINED_DEPENDENCY_FILES[directory])

    for engine, required_by in runtime_requirements.items():
        path = _direct_child(runtime_root / engine, runtime_root)
        if not (required_by - completed_targets) and safe_remove_staged_directory(
            path, runtime_root, verified=True, still_required=False
        ):
            removed.append(path)
    return removed


def main(args: list[str] | None = None) -> int:
    """Verify completed targets and safely clean their recoverable staging assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--completed-target", action="append", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=240)
    options = parser.parse_args(args)
    try:
        removed = cleanup_completed_assets(
            load_config(options.config),
            set(options.completed_target),
            model_root=options.model_root,
            runtime_root=options.runtime_root,
            expected_count=options.expected_count,
        )
    except (KeyError, OSError, ValueError) as error:
        print(f"error: {error}")
        return 2
    for path in removed:
        print(f"removed={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
