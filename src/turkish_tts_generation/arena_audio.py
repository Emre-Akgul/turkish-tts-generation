"""Prepare standardized, hashed WAV assets for the TTS arena."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from turkish_tts_generation.config import GenerationConfig, TargetConfig, load_config
from turkish_tts_generation.contracts import ManifestStatus
from turkish_tts_generation.io import read_manifest

LOUDNESS_I = -23.0
TRUE_PEAK = -1.0
LOUDNESS_RANGE = 7.0
SAMPLE_RATE = 48_000
JSON_BLOCK = re.compile(r"\{[^{]*\"input_i\".*?\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class ArenaAudioRecord:
    """One verified raw-to-arena conversion with complete provenance."""

    sample_id: str
    text: str
    target_name: str
    model_id: str
    prompt_bank_sha256: str | None
    checkpoint_revision: str | None
    runtime_lock_sha256: str | None
    generation_options: dict[str, object] | None
    seed: int | None
    raw_path: str
    raw_sha256: str
    arena_path: str
    normalized_sha256: str
    sample_rate: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    duration_seconds: float
    integrated_lufs: float
    true_peak_dbtp: float


def sha256_file(path: Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffmpeg(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown ffmpeg error"
        raise RuntimeError(detail)
    return result


def measure_loudness(path: Path, *, ffmpeg: str = "ffmpeg") -> dict[str, float]:
    """Measure integrated loudness and true peak with FFmpeg's EBU R128 filter."""
    result = _ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            f"loudnorm=I={LOUDNESS_I}:TP={TRUE_PEAK}:LRA={LOUDNESS_RANGE}:print_format=json",
            "-f",
            "null",
            "-",
        ]
    )
    matches = JSON_BLOCK.findall(result.stderr)
    if not matches:
        raise RuntimeError("ffmpeg did not report loudness measurements")
    values = json.loads(matches[-1])
    try:
        return {"integrated_lufs": float(values["input_i"]), "true_peak_dbtp": float(values["input_tp"])}
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("ffmpeg returned invalid loudness measurements") from error


def normalize_audio(source: Path, destination: Path, *, ffmpeg: str = "ffmpeg") -> None:
    """Run deterministic two-pass EBU R128 normalization into an uncompressed WAV."""
    first = _ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(source),
            "-af",
            f"loudnorm=I={LOUDNESS_I}:TP={TRUE_PEAK}:LRA={LOUDNESS_RANGE}:print_format=json",
            "-f",
            "null",
            "-",
        ]
    )
    matches = JSON_BLOCK.findall(first.stderr)
    if not matches:
        raise RuntimeError("ffmpeg did not report first-pass loudness measurements")
    measured = json.loads(matches[-1])
    if not math.isfinite(float(measured["input_i"])):
        raise RuntimeError(f"generated audio is silent, cannot normalize: {source}")
    parameters = (
        f"loudnorm=I={LOUDNESS_I}:TP={TRUE_PEAK}:LRA={LOUDNESS_RANGE}"
        f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true:print_format=summary"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.wav")
    _ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-y",
            "-i",
            str(source),
            "-af",
            parameters,
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ]
    )
    temporary.replace(destination)


def _audio_properties(path: Path, *, ffmpeg: str) -> dict[str, int | float]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        frames = audio.getnframes()
    if channels != 1 or sample_rate != SAMPLE_RATE or sample_width != 2 or frames <= 0:
        raise RuntimeError(f"invalid standardized audio properties: {path}")
    loudness = measure_loudness(path, ffmpeg=ffmpeg)
    integrated = loudness["integrated_lufs"]
    peak = loudness["true_peak_dbtp"]
    if not -24.0 <= integrated <= -22.0:
        raise RuntimeError(f"integrated loudness outside tolerance ({integrated} LUFS): {path}")
    if peak > -0.8:
        raise RuntimeError(f"true peak exceeds tolerance ({peak} dBTP): {path}")
    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "sample_width_bytes": sample_width,
        "frame_count": frames,
        "duration_seconds": frames / sample_rate,
        "integrated_lufs": integrated,
        "true_peak_dbtp": peak,
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise RuntimeError(f"asset path is outside run root: {path}") from error


def _write_records(path: Path, records: list[ArenaAudioRecord]) -> None:
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_target(
    config: GenerationConfig,
    target: TargetConfig,
    *,
    force: bool = False,
    ffmpeg: str = "ffmpeg",
) -> list[ArenaAudioRecord]:
    """Normalize all successful raw records for one target and write its release manifest."""
    run_root = config.output.root / config.output.run_name
    target_root = run_root / target.name
    raw_records = read_manifest(target_root / "manifest.jsonl")
    eligible = [record for record in raw_records if record.status in {ManifestStatus.SUCCEEDED, ManifestStatus.SKIPPED}]
    if not eligible:
        raise RuntimeError(f"target has no successful raw records: {target.name}")
    if len({record.sample_id for record in eligible}) != len(eligible):
        raise RuntimeError(f"target manifest contains duplicate successful sample IDs: {target.name}")

    arena_manifest = target_root / "arena-manifest.jsonl"
    previous: dict[str, dict[str, object]] = {}
    if arena_manifest.is_file():
        for line in arena_manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict) and isinstance(row.get("sample_id"), str):
                    previous[row["sample_id"]] = row

    records: list[ArenaAudioRecord] = []
    for raw_record in sorted(eligible, key=lambda record: record.source_index):
        source = Path(raw_record.output_path)
        if not source.is_file():
            raise RuntimeError(f"raw audio is missing: {source}")
        raw_hash = sha256_file(source)
        destination = target_root / "arena_audio" / source.name
        prior = previous.get(raw_record.sample_id)
        reusable = (
            not force
            and destination.is_file()
            and prior is not None
            and prior.get("raw_sha256") == raw_hash
            and prior.get("normalized_sha256") == sha256_file(destination)
        )
        if not reusable:
            normalize_audio(source, destination, ffmpeg=ffmpeg)
        properties = _audio_properties(destination, ffmpeg=ffmpeg)
        records.append(
            ArenaAudioRecord(
                sample_id=raw_record.sample_id,
                text=raw_record.text,
                target_name=raw_record.target_name,
                model_id=raw_record.model_id,
                prompt_bank_sha256=raw_record.prompt_bank_sha256,
                checkpoint_revision=raw_record.checkpoint_revision,
                runtime_lock_sha256=raw_record.runtime_lock_sha256,
                generation_options=raw_record.generation_options,
                seed=raw_record.seed,
                raw_path=_relative(source, run_root),
                raw_sha256=raw_hash,
                arena_path=_relative(destination, run_root),
                normalized_sha256=sha256_file(destination),
                sample_rate=int(properties["sample_rate"]),
                channels=int(properties["channels"]),
                sample_width_bytes=int(properties["sample_width_bytes"]),
                frame_count=int(properties["frame_count"]),
                duration_seconds=float(properties["duration_seconds"]),
                integrated_lufs=float(properties["integrated_lufs"]),
                true_peak_dbtp=float(properties["true_peak_dbtp"]),
            )
        )
    _write_records(arena_manifest, records)
    return records


def main(args: list[str] | None = None) -> int:
    """Prepare arena audio for selected configured targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    options = parser.parse_args(args)
    try:
        config = load_config(options.config)
        selected = set(options.target)
        unknown = selected - {target.name for target in config.targets}
        if unknown:
            raise ValueError(f"unknown configured targets: {', '.join(sorted(unknown))}")
        targets = [target for target in config.targets if not selected or target.name in selected]
        total = sum(
            len(prepare_target(config, target, force=options.force, ffmpeg=options.ffmpeg)) for target in targets
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 2
    print(f"prepared={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
