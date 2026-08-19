"""Publish the Turkish TTS Arena v1 prompts and generated audio to the Hugging Face Hub.

Only targets with a complete, hash-verified arena manifest are included; a target
still generating or with unverifiable assets is skipped with a message explaining
why. Safe to re-run at any point during or after a generation run.
"""

import argparse
import json
from pathlib import Path

from datasets import Audio, Dataset

from turkish_tts_generation.cleanup import verify_target_completion
from turkish_tts_generation.config import load_config


def _load_hf_token(env_path: Path) -> str:
    if not env_path.is_file():
        raise SystemExit(f"missing env file: {env_path}")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip() == "HF_TOKEN":
            return value.strip().strip('"').strip("'")
    raise SystemExit(f"HF_TOKEN not found in {env_path}")


def _load_prompts(path: Path) -> dict[str, dict]:
    prompts: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompts[row["id"]] = row
    return prompts


def _ready_targets(config, expected_count: int) -> list[str]:
    ready = []
    for target in config.targets:
        try:
            verify_target_completion(config, target.name, expected_count=expected_count)
        except ValueError as error:
            print(f"skipping {target.name}: {error}")
            continue
        ready.append(target.name)
    return ready


def build_rows(config, prompts: dict[str, dict], target_names: list[str]) -> list[dict]:
    run_root = (config.output.root / config.output.run_name).resolve()
    rows: list[dict] = []
    for target_name in target_names:
        manifest_path = run_root / target_name / "arena-manifest.jsonl"
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = prompts.get(record["sample_id"], {})
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "text": record["text"],
                    "category": prompt.get("category"),
                    "length_bucket": prompt.get("length_bucket"),
                    "tags": prompt.get("tags", []),
                    "target_name": target_name,
                    "model_id": record["model_id"],
                    "audio": str(run_root / record["arena_path"]),
                    "sample_rate": record["sample_rate"],
                    "duration_seconds": record["duration_seconds"],
                    "integrated_lufs": record["integrated_lufs"],
                    "true_peak_dbtp": record["true_peak_dbtp"],
                    "checkpoint_revision": record.get("checkpoint_revision"),
                    "runtime_lock_sha256": record.get("runtime_lock_sha256"),
                    "seed": record.get("seed"),
                    "generation_options": json.dumps(record.get("generation_options") or {}, sort_keys=True),
                    "normalized_sha256": record["normalized_sha256"],
                }
            )
    return rows


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/arena-v1.yaml"))
    parser.add_argument("--prompt-bank", type=Path, default=Path("data/turkish_arena_v1.jsonl"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--username", default="EmreAkgul", help="Hugging Face user or organization namespace.")
    parser.add_argument("--repo-name", default="turkish-tts-arena-v1", help="Dataset repository name.")
    parser.add_argument("--expected-count", type=int, default=240)
    parser.add_argument("--public", action="store_true", help="Publish publicly instead of privately.")
    options = parser.parse_args(args)
    repo_id = f"{options.username}/{options.repo_name}"

    token = _load_hf_token(options.env_file)
    config = load_config(options.config)
    prompts = _load_prompts(options.prompt_bank)

    ready = _ready_targets(config, options.expected_count)
    if not ready:
        print("no targets are complete and verified yet; nothing to publish")
        return 1
    print(f"publishing {len(ready)} target(s): {', '.join(ready)}")

    rows = build_rows(config, prompts, ready)
    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("audio", Audio())
    dataset.push_to_hub(repo_id, token=token, private=not options.public)
    print(f"published {len(rows)} rows to https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
