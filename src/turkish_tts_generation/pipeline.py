"""Dataset-to-engine generation job runner."""

import hashlib
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from turkish_tts_generation.config import DatasetConfig, GenerationConfig, TargetConfig
from turkish_tts_generation.contracts import (
    EngineResult,
    GenerationItem,
    ManifestRecord,
    ManifestStatus,
    RunSummary,
    TextSample,
)
from turkish_tts_generation.dataset import batches, load_samples
from turkish_tts_generation.engine import EngineRegistry
from turkish_tts_generation.io import read_manifest, write_manifest

SampleLoader = Callable[[DatasetConfig], list[TextSample]]


class GenerationRunner:
    """Execute or plan a configured dataset/target generation matrix."""

    def __init__(
        self,
        config: GenerationConfig,
        registry: EngineRegistry,
        *,
        sample_loader: SampleLoader | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.sample_loader = sample_loader or load_samples

    def run(self, *, dry_run: bool = False, force: bool = False) -> RunSummary:
        """Run every configured target against the selected dataset rows."""
        self._validate_engines()
        samples = self.sample_loader(self.config.dataset)
        summary = RunSummary()
        for target in self.config.targets:
            summary += self._run_target(target, samples, dry_run=dry_run, force=force)
        return summary

    def _validate_engines(self) -> None:
        unknown = sorted({target.engine for target in self.config.targets if not self.registry.contains(target.engine)})
        if unknown:
            available = ", ".join(self.registry.names) or "none"
            msg = f"unknown configured engines: {', '.join(unknown)} (available: {available})"
            raise ValueError(msg)

    def _run_target(
        self,
        target: TargetConfig,
        samples: Sequence[TextSample],
        *,
        dry_run: bool,
        force: bool,
    ) -> RunSummary:
        target_dir = self.config.output.root / self.config.output.run_name / target.name
        audio_dir = target_dir / "audio"
        manifest_path = target_dir / "manifest.jsonl"
        output_manifest_path = target_dir / ("plan.jsonl" if dry_run else "manifest.jsonl")
        completed = self._completed_records(manifest_path) if self.config.output.resume and not force else {}

        records: list[ManifestRecord] = []
        pending: list[GenerationItem] = []
        for sample in samples:
            output_path = audio_dir / self._audio_filename(sample)
            previous = completed.get(sample.sample_id)
            if previous is not None and Path(previous.output_path).exists():
                records.append(self._record(sample, target, output_path, ManifestStatus.SKIPPED))
            else:
                pending.append(GenerationItem(sample=sample, output_path=output_path))

        if dry_run:
            records.extend(
                self._record(item.sample, target, item.output_path, ManifestStatus.PLANNED) for item in pending
            )
            write_manifest(output_manifest_path, records)
            return self._summarize(records)

        audio_dir.mkdir(parents=True, exist_ok=True)
        records.extend(self._generate_pending(target, pending))
        records.sort(key=lambda record: record.source_index)
        write_manifest(output_manifest_path, records)
        return self._summarize(records)

    @staticmethod
    def _completed_records(path: Path) -> dict[str, ManifestRecord]:
        return {
            record.sample_id: record
            for record in read_manifest(path)
            if record.status in {ManifestStatus.SUCCEEDED, ManifestStatus.SKIPPED}
        }

    def _generate_pending(self, target: TargetConfig, pending: Sequence[GenerationItem]) -> list[ManifestRecord]:
        if not pending:
            return []
        engine = self.registry.create(target.engine)
        try:
            engine.load(target)
        except Exception as error:  # noqa: BLE001
            return [self._failed_record(item, target, error) for item in pending]

        records: list[ManifestRecord] = []
        try:
            for batch in batches(pending, target.batch_size):
                records.extend(self._generate_batch(engine.generate, batch, target))
        finally:
            engine.unload()
        return records

    def _generate_batch(
        self,
        generate: Callable[[Sequence[GenerationItem]], Sequence[EngineResult]],
        batch: Sequence[GenerationItem],
        target: TargetConfig,
    ) -> list[ManifestRecord]:
        try:
            results = generate(batch)
            results_by_id = {result.sample_id: result for result in results}
            if len(results_by_id) != len(results):
                raise ValueError("engine returned duplicate sample IDs")
            expected_ids = {item.sample.sample_id for item in batch}
            if set(results_by_id) != expected_ids:
                raise ValueError("engine result IDs do not match the requested batch")
        except Exception as error:  # noqa: BLE001
            return [self._failed_record(item, target, error) for item in batch]

        records: list[ManifestRecord] = []
        for item in batch:
            result = results_by_id[item.sample.sample_id]
            if result.error is not None:
                records.append(self._failed_record(item, target, RuntimeError(result.error), result=result))
            elif not item.output_path.is_file():
                records.append(
                    self._failed_record(item, target, FileNotFoundError("engine wrote no audio file"), result=result)
                )
            else:
                records.append(
                    self._record(
                        item.sample,
                        target,
                        item.output_path,
                        ManifestStatus.SUCCEEDED,
                        result=result,
                    )
                )
        return records

    def _failed_record(
        self,
        item: GenerationItem,
        target: TargetConfig,
        error: Exception,
        *,
        result: EngineResult | None = None,
    ) -> ManifestRecord:
        return self._record(
            item.sample,
            target,
            item.output_path,
            ManifestStatus.FAILED,
            result=result,
            error=f"{type(error).__name__}: {error}",
        )

    def _record(
        self,
        sample: TextSample,
        target: TargetConfig,
        output_path: Path,
        status: ManifestStatus,
        *,
        result: EngineResult | None = None,
        error: str | None = None,
    ) -> ManifestRecord:
        dataset = self.config.dataset
        return ManifestRecord(
            run_name=self.config.output.run_name,
            dataset_id=dataset.path,
            dataset_subset=dataset.subset,
            dataset_revision=dataset.revision,
            dataset_split=dataset.split,
            source_index=sample.source_index,
            sample_id=sample.sample_id,
            text=sample.text,
            target_name=target.name,
            engine=target.engine,
            model_id=target.model_id,
            output_path=str(output_path),
            status=status,
            sample_rate=result.sample_rate if result else None,
            duration_seconds=result.duration_seconds if result else None,
            inference_seconds=result.inference_seconds if result else None,
            error=error,
        )

    def _audio_filename(self, sample: TextSample) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", sample.sample_id).strip("-.") or "sample"
        digest = hashlib.sha256(sample.sample_id.encode()).hexdigest()[:8]
        extension = self.config.output.audio_format
        return f"{sample.source_index:08d}-{slug[:80]}-{digest}.{extension}"

    @staticmethod
    def _summarize(records: Sequence[ManifestRecord]) -> RunSummary:
        counts = {status: 0 for status in ManifestStatus}
        for record in records:
            counts[record.status] += 1
        return RunSummary(
            planned=counts[ManifestStatus.PLANNED],
            succeeded=counts[ManifestStatus.SUCCEEDED],
            failed=counts[ManifestStatus.FAILED],
            skipped=counts[ManifestStatus.SKIPPED],
        )
