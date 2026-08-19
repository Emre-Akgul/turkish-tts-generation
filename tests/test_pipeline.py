"""Generation runner behavior tests."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from turkish_tts_generation.config import DatasetConfig, GenerationConfig, OutputConfig, TargetConfig
from turkish_tts_generation.contracts import EngineResult, GenerationItem, ManifestStatus, TextSample
from turkish_tts_generation.engine import EngineRegistry
from turkish_tts_generation.io import read_manifest
from turkish_tts_generation.pipeline import GenerationRunner

SAMPLES = [TextSample(sample_id=f"sample-{index}", text=f"Text {index}", source_index=index) for index in range(3)]


def _config(tmp_path: Path, *, engine: str = "test") -> GenerationConfig:
    return GenerationConfig(
        dataset=DatasetConfig(path="org/dataset", text_column="text"),
        targets=(TargetConfig(name="target", engine=engine, model_id="model", batch_size=2),),
        output=OutputConfig(root=tmp_path / "outputs", run_name="run"),
    )


class SuccessfulEngine:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.loaded = False
        self.unloaded = False

    def load(self, target: TargetConfig) -> None:
        self.loaded = True

    def generate(self, batch: Sequence[GenerationItem]) -> Sequence[EngineResult]:
        self.batch_sizes.append(len(batch))
        results = []
        for item in batch:
            item.output_path.write_bytes(b"audio")
            results.append(
                EngineResult(
                    sample_id=item.sample.sample_id,
                    sample_rate=24_000,
                    duration_seconds=1.0,
                    inference_seconds=0.2,
                )
            )
        return results

    def unload(self) -> None:
        self.unloaded = True


class FlakyEngine(SuccessfulEngine):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate(self, batch: Sequence[GenerationItem]) -> Sequence[EngineResult]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("batch failed")
        return super().generate(batch)


def _registry(engine: SuccessfulEngine) -> EngineRegistry:
    registry = EngineRegistry()
    registry.register("test", lambda: engine)
    return registry


def test_dry_run_plans_targets_without_creating_engine_or_audio(tmp_path: Path) -> None:
    registry = EngineRegistry()
    registry.register("test", lambda: pytest.fail("dry-run created an engine"))
    config = _config(tmp_path)
    config = GenerationConfig(
        dataset=config.dataset,
        targets=(
            config.targets[0],
            TargetConfig(name="target-two", engine="test", model_id="other-model", batch_size=3),
        ),
        output=config.output,
    )
    runner = GenerationRunner(config, registry, sample_loader=lambda _config: SAMPLES)

    summary = runner.run(dry_run=True)

    target_dir = tmp_path / "outputs" / "run" / "target"
    records = read_manifest(target_dir / "plan.jsonl")
    second_records = read_manifest(tmp_path / "outputs" / "run" / "target-two" / "plan.jsonl")
    assert summary.planned == 6
    assert {record.status for record in records} == {ManifestStatus.PLANNED}
    assert {record.target_name for record in second_records} == {"target-two"}
    assert not (target_dir / "audio").exists()


def test_batch_generation_and_manifest(tmp_path: Path) -> None:
    engine = SuccessfulEngine()
    runner = GenerationRunner(_config(tmp_path), _registry(engine), sample_loader=lambda _config: SAMPLES)

    summary = runner.run()

    records = read_manifest(tmp_path / "outputs" / "run" / "target" / "manifest.jsonl")
    assert summary.succeeded == 3
    assert engine.batch_sizes == [2, 1]
    assert engine.loaded and engine.unloaded
    assert all(Path(record.output_path).is_file() for record in records)
    assert all(record.sample_rate == 24_000 for record in records)
    assert all(record.raw_sha256 is not None for record in records)


def test_batch_failure_is_recorded_and_next_batch_continues(tmp_path: Path) -> None:
    runner = GenerationRunner(_config(tmp_path), _registry(FlakyEngine()), sample_loader=lambda _config: SAMPLES)

    summary = runner.run()

    records = read_manifest(tmp_path / "outputs" / "run" / "target" / "manifest.jsonl")
    assert summary.failed == 2
    assert summary.succeeded == 1
    assert [record.status for record in records] == [
        ManifestStatus.FAILED,
        ManifestStatus.FAILED,
        ManifestStatus.SUCCEEDED,
    ]


def test_resume_skips_successes_and_retries_missing_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first_engine = SuccessfulEngine()
    GenerationRunner(config, _registry(first_engine), sample_loader=lambda _config: SAMPLES).run()
    manifest_path = tmp_path / "outputs" / "run" / "target" / "manifest.jsonl"
    original_records = read_manifest(manifest_path)

    skipped_engine = SuccessfulEngine()
    skipped = GenerationRunner(config, _registry(skipped_engine), sample_loader=lambda _config: SAMPLES).run()
    assert skipped.skipped == 3
    assert not skipped_engine.loaded

    Path(original_records[0].output_path).unlink()
    retry_engine = SuccessfulEngine()
    retried = GenerationRunner(config, _registry(retry_engine), sample_loader=lambda _config: SAMPLES).run()
    assert retried.succeeded == 1
    assert retried.skipped == 2
    assert retry_engine.batch_sizes == [1]


def test_sample_filtered_run_preserves_manifest_history(tmp_path: Path) -> None:
    config = _config(tmp_path)
    GenerationRunner(config, _registry(SuccessfulEngine()), sample_loader=lambda _config: SAMPLES).run()
    manifest_path = tmp_path / "outputs" / "run" / "target" / "manifest.jsonl"
    assert len(read_manifest(manifest_path)) == 3

    smoke_engine = SuccessfulEngine()
    smoke = GenerationRunner(config, _registry(smoke_engine), sample_loader=lambda _config: SAMPLES).run(
        sample_ids={"sample-0"}
    )

    assert smoke.skipped == 1
    statuses = {record.sample_id: record.status for record in read_manifest(manifest_path)}
    assert statuses == {
        "sample-0": ManifestStatus.SKIPPED,
        "sample-1": ManifestStatus.SUCCEEDED,
        "sample-2": ManifestStatus.SUCCEEDED,
    }


def test_force_regenerates_successful_records(tmp_path: Path) -> None:
    config = _config(tmp_path)
    GenerationRunner(config, _registry(SuccessfulEngine()), sample_loader=lambda _config: SAMPLES).run()
    forced_engine = SuccessfulEngine()

    summary = GenerationRunner(config, _registry(forced_engine), sample_loader=lambda _config: SAMPLES).run(force=True)

    assert summary.succeeded == 3
    assert summary.skipped == 0
    assert forced_engine.batch_sizes == [2, 1]


def test_unknown_engine_fails_before_loading_dataset(tmp_path: Path) -> None:
    registry = EngineRegistry()
    runner = GenerationRunner(
        _config(tmp_path, engine="missing"),
        registry,
        sample_loader=lambda _config: pytest.fail("unknown engine loaded the dataset"),
    )

    with pytest.raises(ValueError, match="unknown configured engines"):
        runner.run(dry_run=True)


def test_reject_model_engine_mismatch_before_loading_dataset(tmp_path: Path) -> None:
    registry = EngineRegistry()
    registry.register("xtts", lambda: pytest.fail("mismatched target created an engine"))
    config = _config(tmp_path, engine="xtts")
    config = GenerationConfig(
        dataset=config.dataset,
        targets=(TargetConfig(name="target", engine="xtts", model_id="voxcpm2"),),
        output=config.output,
    )
    runner = GenerationRunner(
        config,
        registry,
        sample_loader=lambda _config: pytest.fail("mismatched target loaded the dataset"),
    )

    with pytest.raises(ValueError, match="requires engine 'voxcpm'"):
        runner.run(dry_run=True)


def test_filters_targets_and_samples(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = GenerationConfig(
        dataset=config.dataset,
        targets=(config.targets[0], TargetConfig(name="other", engine="test", model_id="other")),
        output=config.output,
    )
    runner = GenerationRunner(config, _registry(SuccessfulEngine()), sample_loader=lambda _config: SAMPLES)

    summary = runner.run(dry_run=True, target_names={"other"}, sample_ids={"sample-1"})

    assert summary.planned == 1
    records = read_manifest(tmp_path / "outputs" / "run" / "other" / "plan.jsonl")
    assert [record.sample_id for record in records] == ["sample-1"]


def test_rejects_unknown_target_or_sample_filter(tmp_path: Path) -> None:
    runner = GenerationRunner(_config(tmp_path), _registry(SuccessfulEngine()), sample_loader=lambda _config: SAMPLES)

    with pytest.raises(ValueError, match="unknown configured targets"):
        runner.run(dry_run=True, target_names={"missing"})
    with pytest.raises(ValueError, match="unknown selected sample IDs"):
        runner.run(dry_run=True, sample_ids={"missing"})
