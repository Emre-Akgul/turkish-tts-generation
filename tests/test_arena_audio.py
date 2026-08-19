"""Arena audio normalization and manifest tests."""

import math
import struct
import wave
from pathlib import Path

from turkish_tts_generation.arena_audio import SAMPLE_RATE, measure_loudness, normalize_audio, prepare_target
from turkish_tts_generation.config import DatasetConfig, GenerationConfig, OutputConfig, TargetConfig
from turkish_tts_generation.contracts import ManifestRecord, ManifestStatus
from turkish_tts_generation.io import write_manifest


def _tone(path: Path) -> None:
    rate = 16_000
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        frames = [int(1500 * math.sin(2 * math.pi * 220 * index / rate)) for index in range(rate * 3)]
        audio.writeframes(b"".join(struct.pack("<h", frame) for frame in frames))


def test_two_pass_normalization_produces_arena_wav(tmp_path: Path) -> None:
    source = tmp_path / "raw.wav"
    destination = tmp_path / "arena.wav"
    _tone(source)

    normalize_audio(source, destination)

    with wave.open(str(destination), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == SAMPLE_RATE
        assert audio.getsampwidth() == 2
        assert audio.getnframes() > 0
    loudness = measure_loudness(destination)
    assert -24.0 <= loudness["integrated_lufs"] <= -22.0
    assert loudness["true_peak_dbtp"] <= -0.8


def test_prepare_target_writes_relative_hashed_manifest(tmp_path: Path) -> None:
    output = OutputConfig(root=tmp_path / "outputs", run_name="arena")
    target = TargetConfig(name="model", engine="test", model_id="test")
    config = GenerationConfig(dataset=DatasetConfig(path="json", text_column="text"), targets=(target,), output=output)
    target_root = output.root / output.run_name / target.name
    raw = target_root / "audio" / "sample.wav"
    raw.parent.mkdir(parents=True)
    _tone(raw)
    write_manifest(
        target_root / "manifest.jsonl",
        [
            ManifestRecord(
                run_name="arena",
                dataset_id="json",
                dataset_subset=None,
                dataset_revision=None,
                dataset_split="train",
                source_index=0,
                sample_id="sample",
                text="Merhaba dünya.",
                target_name="model",
                engine="test",
                model_id="test",
                output_path=str(raw),
                status=ManifestStatus.SUCCEEDED,
            )
        ],
    )

    records = prepare_target(config, target)

    assert len(records) == 1
    assert records[0].raw_path == "model/audio/sample.wav"
    assert records[0].arena_path == "model/arena_audio/sample.wav"
    assert len(records[0].raw_sha256) == 64
    assert len(records[0].normalized_sha256) == 64
    assert (target_root / "arena-manifest.jsonl").is_file()
