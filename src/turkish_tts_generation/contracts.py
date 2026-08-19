"""Shared data contracts for datasets, engines, and manifests."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TextSample:
    """One normalized text row from the source dataset."""

    sample_id: str
    text: str
    source_index: int
    reference_audio: str | None = None
    reference_text: str | None = None
    speaker_id: str | None = None


@dataclass(frozen=True, slots=True)
class GenerationItem:
    """One sample and its precomputed output destination."""

    sample: TextSample
    output_path: Path


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Per-sample result returned by an inference engine."""

    sample_id: str
    sample_rate: int | None = None
    duration_seconds: float | None = None
    inference_seconds: float | None = None
    error: str | None = None


class ManifestStatus(str, Enum):
    """Lifecycle status of a generation item."""

    PLANNED = "planned"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """Serializable provenance and status for one target/sample pair."""

    run_name: str
    dataset_id: str
    dataset_subset: str | None
    dataset_revision: str | None
    dataset_split: str
    source_index: int
    sample_id: str
    text: str
    target_name: str
    engine: str
    model_id: str
    output_path: str
    status: ManifestStatus
    sample_rate: int | None = None
    duration_seconds: float | None = None
    inference_seconds: float | None = None
    error: str | None = None
    prompt_bank_sha256: str | None = None
    checkpoint_revision: str | None = None
    runtime_lock_sha256: str | None = None
    generation_options: dict[str, object] | None = None
    seed: int | None = None
    raw_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Aggregate counts returned by a generation job."""

    planned: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0

    def __add__(self, other: "RunSummary") -> "RunSummary":
        return RunSummary(
            planned=self.planned + other.planned,
            succeeded=self.succeeded + other.succeeded,
            failed=self.failed + other.failed,
            skipped=self.skipped + other.skipped,
        )
