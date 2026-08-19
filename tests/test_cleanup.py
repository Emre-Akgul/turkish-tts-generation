"""Verified staged-asset cleanup tests."""

from pathlib import Path

import pytest

from turkish_tts_generation.cleanup import _requirements, safe_remove_staged_directory
from turkish_tts_generation.config import load_config

ARENA_CONFIG = Path(__file__).parents[1] / "configs" / "arena-v1.yaml"


def test_safe_cleanup_requires_verification_and_no_dependents(tmp_path: Path) -> None:
    root = tmp_path / "models"
    target = root / "model"
    target.mkdir(parents=True)
    (target / "weights.bin").write_bytes(b"weights")

    with pytest.raises(ValueError, match="verified"):
        safe_remove_staged_directory(target, root, verified=False, still_required=False)
    with pytest.raises(ValueError, match="still required"):
        safe_remove_staged_directory(target, root, verified=True, still_required=True)

    assert safe_remove_staged_directory(target, root, verified=True, still_required=False)
    assert not target.exists()


def test_safe_cleanup_refuses_root_nested_and_symlink_targets(tmp_path: Path) -> None:
    root = tmp_path / "models"
    nested = root / "model" / "nested"
    nested.mkdir(parents=True)

    with pytest.raises(ValueError, match="direct child"):
        safe_remove_staged_directory(root, root, verified=True, still_required=False)
    with pytest.raises(ValueError, match="direct child"):
        safe_remove_staged_directory(nested, root, verified=True, still_required=False)

    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="direct child"):
        safe_remove_staged_directory(link, root, verified=True, still_required=False)


def test_shared_asset_and_runtime_dependencies_are_retained() -> None:
    models, runtimes, _owners = _requirements(load_config(ARENA_CONFIG))

    assert models["xtts-v2"] == {"xtts-v2", "orkhon", "moss-nano"}
    assert models["voxcpm2"] == {"voxcpm2", "freya"}
    assert runtimes["voxcpm"] == {"trendyol", "voxcpm2"}
