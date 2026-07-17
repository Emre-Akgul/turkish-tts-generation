"""Batch-first inference-engine protocol and registry."""

from collections.abc import Callable, Sequence
from typing import Protocol

from turkish_tts_generation.config import TargetConfig
from turkish_tts_generation.contracts import EngineResult, GenerationItem


class InferenceEngine(Protocol):
    """Interface implemented by inference backends, not individual models."""

    def load(self, target: TargetConfig) -> None:
        """Load resources for a configured target."""
        ...

    def generate(self, batch: Sequence[GenerationItem]) -> Sequence[EngineResult]:
        """Generate one result for each item in a batch."""
        ...

    def unload(self) -> None:
        """Release loaded resources."""
        ...


EngineFactory = Callable[[], InferenceEngine]


class EngineRegistry:
    """Resolve configured engine names to engine factories."""

    def __init__(self) -> None:
        self._factories: dict[str, EngineFactory] = {}

    def register(self, name: str, factory: EngineFactory) -> None:
        """Register a unique engine factory."""
        normalized_name = name.strip().lower()
        if not normalized_name:
            msg = "engine name must not be empty"
            raise ValueError(msg)
        if normalized_name in self._factories:
            msg = f"engine already registered: {normalized_name}"
            raise ValueError(msg)
        self._factories[normalized_name] = factory

    def create(self, name: str) -> InferenceEngine:
        """Create an engine by its configured name."""
        normalized_name = name.strip().lower()
        try:
            factory = self._factories[normalized_name]
        except KeyError as error:
            available = ", ".join(sorted(self._factories)) or "none"
            msg = f"unknown engine '{name}' (available: {available})"
            raise ValueError(msg) from error
        return factory()

    def contains(self, name: str) -> bool:
        """Return whether an engine name is registered."""
        return name.strip().lower() in self._factories

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered engine names in stable order."""
        return tuple(sorted(self._factories))


class NoopEngine:
    """Registry placeholder used to validate dry-run jobs."""

    def load(self, target: TargetConfig) -> None:
        """Reject real generation because this engine is planning-only."""
        msg = f"engine 'noop' cannot generate target '{target.name}'; use --dry-run"
        raise RuntimeError(msg)

    def generate(self, batch: Sequence[GenerationItem]) -> Sequence[EngineResult]:
        """Reject generation if called without loading."""
        raise RuntimeError("engine 'noop' does not generate audio")

    def unload(self) -> None:
        """No resources are held by this engine."""
        return None


def create_default_registry() -> EngineRegistry:
    """Create the built-in engine registry."""
    registry = EngineRegistry()
    registry.register("noop", NoopEngine)
    return registry
