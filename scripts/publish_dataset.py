"""Publish each Turkish TTS Arena v1 model as a Hugging Face dataset subset.

Only targets with a complete, hash-verified generation manifest are included; a
target still generating or with unverifiable assets is skipped with a message
explaining why. Safe to re-run at any point during or after a generation run.
"""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Audio, Dataset

from turkish_tts_generation.config import load_config


def _load_hf_token(env_path: Path) -> str | None:
    if not env_path.is_file():
        # huggingface_hub falls back to the token saved by `hf auth login`.
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip() == "HF_TOKEN":
            return value.strip().strip('"').strip("'")
    return None


def _load_prompts(path: Path) -> dict[str, dict]:
    prompts: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompts[row["id"]] = row
    return prompts


def _ready_targets(config, expected_count: int, selected: set[str]) -> list[str]:
    run_root = (config.output.root / config.output.run_name).resolve()
    ready = []
    for target in config.targets:
        if selected and target.name not in selected:
            continue
        manifest = run_root / target.name / "manifest.jsonl"
        try:
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
            by_id = {row["sample_id"]: row for row in rows}
            if len(rows) != expected_count or len(by_id) != expected_count:
                raise ValueError(f"expected {expected_count} unique records, found {len(rows)}/{len(by_id)}")
            for row in rows:
                if row["status"] not in {"succeeded", "skipped"}:
                    raise ValueError(f"unsuccessful sample {row['sample_id']}")
                path = Path(row["output_path"])
                if not path.is_file():
                    raise ValueError(f"missing audio {path}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != row.get("raw_sha256"):
                    raise ValueError(f"hash mismatch for {path}")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"skipping {target.name}: {error}")
            continue
        ready.append(target.name)
    return ready


def build_rows(config, prompts: dict[str, dict], target_names: list[str]) -> list[dict]:
    run_root = (config.output.root / config.output.run_name).resolve()
    rows: list[dict] = []
    for target_name in target_names:
        manifest_path = run_root / target_name / "manifest.jsonl"
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
                    "audio": str(Path(record["output_path"]).resolve()),
                    "sample_rate": record["sample_rate"],
                    "duration_seconds": record["duration_seconds"],
                    "checkpoint_revision": record.get("checkpoint_revision"),
                    "runtime_lock_sha256": record.get("runtime_lock_sha256"),
                    "seed": record.get("seed"),
                    "generation_options": json.dumps(record.get("generation_options") or {}, sort_keys=True),
                    "raw_sha256": record["raw_sha256"],
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
    parser.add_argument("--target", action="append", default=[], help="Publish only this model subset; repeatable.")
    parser.add_argument("--public", action="store_true", help="Publish publicly instead of privately.")
    parser.add_argument("--dry-run", action="store_true", help="Build and report subsets without uploading them.")
    options = parser.parse_args(args)
    repo_id = f"{options.username}/{options.repo_name}"

    token = _load_hf_token(options.env_file)
    config = load_config(options.config)
    prompts = _load_prompts(options.prompt_bank)

    selected = set(options.target)
    configured = {target.name for target in config.targets}
    unknown = selected - configured
    if unknown:
        parser.error(f"unknown target(s): {', '.join(sorted(unknown))}")

    ready = _ready_targets(config, options.expected_count, selected)
    if not ready:
        print("no targets are complete and verified yet; nothing to publish")
        return 1
    action = "building" if options.dry_run else "publishing"
    print(f"{action} {len(ready)} model subset(s): {', '.join(ready)}")

    total = 0
    for index, target_name in enumerate(ready):
        rows = build_rows(config, prompts, [target_name])
        if len(rows) != options.expected_count:
            raise SystemExit(f"{target_name}: expected {options.expected_count} rows, found {len(rows)}")
        dataset = Dataset.from_list(rows).cast_column("audio", Audio())
        total += len(rows)
        if options.dry_run:
            print(f"ready subset={target_name} rows={len(rows)}")
            continue
        dataset.push_to_hub(
            repo_id,
            config_name=target_name,
            set_default=index == 0,
            split="train",
            token=token,
            private=not options.public,
        )
        print(f"published subset={target_name} rows={len(rows)}")
    destination = f"https://huggingface.co/datasets/{repo_id}"
    print(f"{'validated' if options.dry_run else 'published'} {total} rows across {len(ready)} subsets: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
