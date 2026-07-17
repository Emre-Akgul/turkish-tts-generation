"""JSON Lines manifest persistence."""

import json
from dataclasses import asdict
from pathlib import Path

from turkish_tts_generation.contracts import ManifestRecord, ManifestStatus


def read_manifest(path: Path) -> list[ManifestRecord]:
    """Read a manifest, returning an empty list when it does not exist."""
    if not path.exists():
        return []
    records: list[ManifestRecord] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                data["status"] = ManifestStatus(data["status"])
                records.append(ManifestRecord(**data))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                msg = f"invalid manifest record at {path}:{line_number}"
                raise ValueError(msg) from error
    return records


def write_manifest(path: Path, records: list[ManifestRecord]) -> None:
    """Atomically replace a JSON Lines manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    temporary_path.replace(path)
