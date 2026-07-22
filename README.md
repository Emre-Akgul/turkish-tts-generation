# Turkish TTS Generation

Generate comparable Turkish TTS samples from one Hugging Face dataset. Checkpoints that share an architecture share an engine; dependency-incompatible engines run in isolated, persistent subprocesses.

## Supported models

| Model | Engine | Conditioning |
| --- | --- | --- |
| Trendyol-TTS | `voxcpm` | text |
| Chatterbox Multilingual V3 | `chatterbox` | text, optional reference |
| VoxCPM2 | `voxcpm` | text |
| Orkhon-TTS | `f5-tts` | reference audio |
| MOSS-TTS-Nano-100M | `moss-tts` | reference audio |
| Supertonic 3 | `supertonic` | built-in voice |
| XTTS-v2 | `xtts` | reference audio or speaker ID |
| OmniVoice | `omnivoice` | text, optional reference |
| Freya-TTS | `freya` | deterministic seed voice |

Trendyol-TTS and VoxCPM2 intentionally use the same `voxcpm` engine. Model aliases and exact Hugging Face IDs are both accepted.

## 1. Install the job runner

```bash
make setup
```

## 2. Download checkpoints

Models are not stored in Git. Put them on a disk with at least 23 GiB free:

```bash
make download-models MODEL_ROOT=/media/$USER/your-disk/turkish-tts-models
export TTS_MODEL_ROOT=/media/$USER/your-disk/turkish-tts-models
```

The downloader is resumable and verifies required files. `HF_TOKEN` is read from the environment for gated repositories.

## 3. Install engine runtimes

Install only the families you plan to execute:

```bash
make setup-runtime ENGINE=voxcpm
make setup-runtime ENGINE=supertonic
```

Use `ENGINE=all` to install all eight environments. They are kept under `.runtimes/` because their Torch and Transformers requirements conflict. Override an interpreter with, for example, `TTS_VOXCPM_PYTHON=/path/to/python`.

## 4. Configure and run

Edit [`configs/generation.yaml`](configs/generation.yaml). Voice-cloning models obtain conditioning through these optional dataset mappings:

```yaml
dataset:
  text_column: text
  reference_audio_column: reference_audio
  reference_text_column: reference_text
  speaker_id_column: speaker_id
```

A target-wide reference can instead be placed in `options.reference_audio`. Relative paths are interpreted from the command's working directory.

Validate dataset selection, targets, and output paths without loading any model:

```bash
uv run tts-generate --config configs/generation.yaml --dry-run
```

Generate audio:

```bash
uv run tts-generate --config configs/generation.yaml
```

Outputs are written to:

```text
outputs/<run>/<target>/audio/
outputs/<run>/<target>/manifest.jsonl
```

Successful existing artifacts are resumed; failed or missing samples are retried. Pass `--force` to regenerate everything.

## Engine options

Each target accepts an `options` mapping. Common useful options include `model_path`, `reference_audio`, `reference_text`, `language`, `seed`, `steps`, and `speed`. Defaults are Turkish-oriented. The engine worker returns one result for every requested item, so a failure in one sample does not abort later samples or batches.

## Development

```bash
make format
make check
```
