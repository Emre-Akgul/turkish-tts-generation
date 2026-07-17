# Turkish TTS Generation

Generate comparable TTS artifacts from a Hugging Face dataset through inference-engine backends.

The template separates four concerns:

```text
Hugging Face Dataset → normalized text samples → inference engine → audio + manifest
```

Models are configuration values. Engines are reusable backends registered under a short name, so one engine can serve many compatible models.

## Setup

```bash
make setup
```

Set `HF_TOKEN` in your environment when the dataset is private or gated. Tokens are never stored in the YAML configuration.

## Configure a job

Edit [`configs/generation.yaml`](configs/generation.yaml):

```yaml
dataset:
  path: your-namespace/your-dataset
  subset: null
  revision: null
  split: train
  text_column: text
  id_column: null
  shuffle: true
  seed: 42
  limit: 10

targets:
  - name: baseline
    engine: noop
    model_id: your-model-id
    batch_size: 4
    device: auto
    dtype: auto
    options: {}

output:
  root: outputs
  run_name: example
  audio_format: wav
  resume: true
```

`text_column` is required. When `id_column` is omitted, deterministic IDs are generated from source row indices. Selection is deterministic for a fixed `shuffle`, `seed`, and `limit`.

## Plan a run

The built-in `noop` engine exists only for dry runs. It validates the complete dataset/target matrix without loading a model or writing audio:

```bash
uv run tts-generate --config configs/generation.yaml --dry-run
```

Each target receives a plan at:

```text
outputs/<run_name>/<target_name>/plan.jsonl
```

Real engines write audio and a resumable manifest under:

```text
outputs/<run_name>/<target_name>/audio/
outputs/<run_name>/<target_name>/manifest.jsonl
```

Successful artifacts are skipped on the next run. Failed or missing artifacts are retried. Pass `--force` to ignore resume state.

## Add an inference engine

Implement the batch-first `InferenceEngine` protocol and register its factory in `create_default_registry` so the CLI can resolve its configured name:

```python
registry.register("my-engine", MyEngine)
```

An engine receives a `TargetConfig` during `load`, then batches of `GenerationItem` values. It must write audio to each item's precomputed `output_path` and return one `EngineResult` per sample.

## Development

```bash
make format
make check
```
