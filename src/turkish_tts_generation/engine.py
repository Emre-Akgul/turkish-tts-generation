"""Batch-first inference-engine protocol, registry, and isolated runtimes."""

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from turkish_tts_generation.config import TargetConfig
from turkish_tts_generation.contracts import EngineResult, GenerationItem
from turkish_tts_generation.models import resolve_model


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


class SubprocessEngine:
    """Run a model family in its own dependency-isolated Python process."""

    def __init__(self, engine_name: str) -> None:
        self.engine_name = engine_name
        self.process: subprocess.Popen[str] | None = None

    def load(self, target: TargetConfig) -> None:
        model = resolve_model(target.model_id)
        if model.engine != self.engine_name:
            msg = f"model '{target.model_id}' requires engine '{model.engine}', not '{self.engine_name}'"
            raise ValueError(msg)

        root = Path(os.getenv("TTS_MODEL_ROOT", "models")).expanduser()
        model_path = Path(str(target.options.get("model_path", root / model.directory))).expanduser().resolve()
        if not model_path.is_dir():
            msg = f"model directory does not exist: {model_path}"
            raise FileNotFoundError(msg)
        companion_path = None
        if model.companion_directory:
            companion_path = (
                Path(str(target.options.get("companion_path", root / model.companion_directory))).expanduser().resolve()
            )

        python = self._python_executable()
        command = [
            str(python),
            "-m",
            "turkish_tts_generation.worker",
            "--engine",
            self.engine_name,
            "--model-path",
            str(model_path),
            "--device",
            target.device,
            "--dtype",
            target.dtype,
            "--options",
            json.dumps(target.options),
        ]
        if companion_path is not None:
            command.extend(("--companion-path", str(companion_path)))

        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, env.get("PYTHONPATH"))))
        self.process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )
        response = self._read_response()
        if not response.get("ready"):
            error = str(response.get("error", "worker failed during startup"))
            self.unload()
            raise RuntimeError(error)

    def generate(self, batch: Sequence[GenerationItem]) -> Sequence[EngineResult]:
        payload = {
            "command": "generate",
            "items": [
                {
                    "sample_id": item.sample.sample_id,
                    "text": item.sample.text,
                    "reference_audio": item.sample.reference_audio,
                    "reference_text": item.sample.reference_text,
                    "speaker_id": item.sample.speaker_id,
                    "output_path": str(item.output_path.resolve()),
                }
                for item in batch
            ],
        }
        self._write_request(payload)
        response = self._read_response()
        if "error" in response:
            raise RuntimeError(str(response["error"]))
        return [EngineResult(**result) for result in response["results"]]

    def unload(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin:
                    process.stdin.write('{"command":"shutdown"}\n')
                    process.stdin.flush()
                process.wait(timeout=15)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream:
                stream.close()

    def _python_executable(self) -> Path:
        variable = f"TTS_{self.engine_name.upper().replace('-', '_')}_PYTHON"
        configured = os.getenv(variable)
        if configured:
            return Path(configured).expanduser().absolute()
        project_root = Path(__file__).resolve().parents[2]
        runtime = project_root / ".runtimes" / self.engine_name / ".venv" / "bin" / "python"
        return runtime.absolute() if runtime.is_file() else Path(sys.executable)

    def _write_request(self, payload: Mapping[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("engine worker is not running")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    def _read_response(self) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("engine worker is not running")
        line = self.process.stdout.readline()
        if line:
            response = json.loads(line)
            if isinstance(response, dict):
                return response
        code = self.process.poll()
        raise RuntimeError(f"{self.engine_name} worker exited unexpectedly (exit code {code})")


def create_default_registry() -> EngineRegistry:
    """Create the built-in engine registry."""
    registry = EngineRegistry()
    registry.register("noop", NoopEngine)
    for name in ("voxcpm", "chatterbox", "f5-tts", "moss-tts", "supertonic", "xtts", "omnivoice", "freya"):
        registry.register(name, lambda name=name: SubprocessEngine(name))
    return registry
