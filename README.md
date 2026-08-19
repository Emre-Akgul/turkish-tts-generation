# Turkish TTS Generation

Generate comparable Turkish TTS samples from a local JSONL file or Hugging Face dataset. Checkpoints that share an architecture share an engine; dependency-incompatible engines run in isolated, persistent subprocesses.

## Supported models

| Model | Engine | Conditioning |
| --- | --- | --- |
| Trendyol-TTS | `voxcpm` | text |
| Chatterbox Multilingual V3 | `chatterbox` | text, optional reference |
| VoxCPM2 | `voxcpm` | text |
| Orkhon-TTS | `f5-tts` | optional reference |
| MOSS-TTS-Nano-100M | `moss-tts` | optional reference |
| Supertonic 3 | `supertonic` | built-in voice |
| XTTS-v2 | `xtts` | optional reference or speaker ID |
| OmniVoice | `omnivoice` | text, optional reference |
| Freya-TTS | `freya` | deterministic seed voice |
| Fish Audio S2 Pro | `fish-speech` | text, optional reference |

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

There are ten models and nine engine environments. Use `ENGINE=all` to install every environment. They are kept under `.runtimes/` because their Torch and Transformers requirements conflict. Set `TTS_RUNTIME_ROOT` to stage them elsewhere, or override one interpreter with, for example, `TTS_VOXCPM_PYTHON=/path/to/python`.

## 4. Configure and run

Edit [`configs/generation.yaml`](configs/generation.yaml). Voice-cloning models obtain conditioning through these optional dataset mappings:

```yaml
dataset:
  text_column: text
  reference_audio_column: reference_audio
  reference_text_column: reference_text
  speaker_id_column: speaker_id
```

All reference fields are optional. When no reference or speaker ID is supplied, Orkhon, MOSS, and XTTS use the Turkish sample bundled with the downloaded XTTS checkpoint as their default voice. A target-wide reference can instead be placed in `options.reference_audio`; it overrides this fallback. Relative paths are interpreted from the command's working directory.

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

Use repeatable `--target NAME` and `--sample-id ID` options to run one model or a smoke subset. Every worker receives a frozen random seed before model loading.

## Turkish TTS Arena v1

The public [`data/turkish_arena_v1.jsonl`](data/turkish_arena_v1.jsonl) bank contains 240 original Turkish prompts under CC0. It has exact category and length quotas and automated structural validation; it is intentionally marked `automated_only`, not human-reviewed.

Validate the prompt contract and plan the full ten-model matrix:

```bash
uv run tts-validate-prompts data/turkish_arena_v1.jsonl
uv run tts-generate --config configs/arena-v1.yaml --dry-run
```

The dry run must report `planned=2400`. Checkpoint commits and top-level runtime requirements are pinned. On first installation of an engine, its complete resolved package set is saved under `runtime-locks/`; later installations reuse that lock.

For limited disk space, stage one model at a time. Replace `/safe/staging` with an explicit directory on a disk that passes the built-in 8 GiB headroom checks:

```bash
export TTS_MODEL_ROOT=/safe/staging/models
export TTS_RUNTIME_ROOT=/safe/staging/runtimes

uv run tts-stage --config configs/arena-v1.yaml --target supertonic-3 \
  --model-root "$TTS_MODEL_ROOT" --runtime-root "$TTS_RUNTIME_ROOT"
```

The stage command installs the pinned runtime, downloads only the selected model and required companions, runs the cross-category smoke set, resumes the full 240 prompts, prepares arena audio, verifies hashes, discovers previously completed targets, and performs guarded cleanup. Its equivalent individual commands are:

```bash
uv run python scripts/setup_runtime.py supertonic --runtime-root "$TTS_RUNTIME_ROOT"
uv run python scripts/download_models.py --model-root "$TTS_MODEL_ROOT" --model supertonic-3
uv run tts-generate --config configs/arena-v1.yaml --target supertonic-3 \
  --sample-id tr-arena-v1-0001 --sample-id tr-arena-v1-0051 \
  --sample-id tr-arena-v1-0073 --sample-id tr-arena-v1-0105 \
  --sample-id tr-arena-v1-0121 --sample-id tr-arena-v1-0136 \
  --sample-id tr-arena-v1-0157 --sample-id tr-arena-v1-0170 \
  --sample-id tr-arena-v1-0181 --sample-id tr-arena-v1-0200 \
  --sample-id tr-arena-v1-0217 --sample-id tr-arena-v1-0232
uv run tts-generate --config configs/arena-v1.yaml --target supertonic-3
uv run tts-prepare-arena --config configs/arena-v1.yaml --target supertonic-3
```

After all 240 raw and standardized files for completed targets validate, remove recoverable staging assets with:

```bash
uv run tts-cleanup --config configs/arena-v1.yaml \
  --completed-target supertonic-3 \
  --model-root "$TTS_MODEL_ROOT" --runtime-root "$TTS_RUNTIME_ROOT"
```

Cleanup refuses missing or changed outputs, paths outside the explicit staging roots, and assets still required by an incomplete target. Shared XTTS reference audio and VoxCPM2 AudioVAE files are retained until their final consumers complete. Deleted checkpoints and runtimes can be reconstructed from the recorded revisions and lock files.

Arena-ready files are mono 48 kHz, 16-bit PCM WAV normalized with two-pass EBU R128 processing to -23 LUFS and at most -1 dBTP. Raw files are retained unchanged. Each target receives an `arena-manifest.jsonl` containing relative paths and hashes for both forms.

## Engine options

Each target accepts an `options` mapping. Common useful options include `model_path`, `reference_audio`, `reference_text`, `language`, `seed`, `steps`, and `speed`. Defaults are Turkish-oriented. The engine worker returns one result for every requested item, so a failure in one sample does not abort later samples or batches.

## Development

```bash
make format
make check
```
