#!/usr/bin/env bash
# Generate TTS samples for every model in the arena config, one at a time.
# Downloaded checkpoints and installed runtimes are left in place (models
# already present on TTS_MODEL_ROOT are reused, missing ones are downloaded).
#
# Required:
#   TTS_MODEL_ROOT    staging directory for model checkpoints
#   TTS_RUNTIME_ROOT   staging directory for engine runtimes
#
# Optional:
#   TTS_CONFIG         config file (default: configs/arena-v1.yaml)
#   STOP_ON_FAILURE     0 to keep going after a target fails (default: 1)
#   TTS_HEADROOM_GIB    free-space safety margin required before each
#                       download/runtime install, in GiB (default: 8)
#   TTS_CLEANUP         1 to delete each model/runtime after it verifiably
#                       completes, to save disk (default: 0)
#
# Usage:
#   export TTS_MODEL_ROOT=/media/$USER/your-disk/turkish-tts-models
#   export TTS_RUNTIME_ROOT=/media/$USER/your-disk/turkish-tts-runtimes
#   ./scripts/generate_all.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${TTS_CONFIG:-configs/arena-v1.yaml}"
: "${TTS_MODEL_ROOT:?Set TTS_MODEL_ROOT to a staging directory with enough free space}"
: "${TTS_RUNTIME_ROOT:?Set TTS_RUNTIME_ROOT to a staging directory with enough free space}"

mapfile -t TARGETS < <(uv run python -c "
import yaml
with open('${CONFIG}') as f:
    config = yaml.safe_load(f)
for target in config['targets']:
    print(target['name'])
")

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "error: no targets found in ${CONFIG}" >&2
  exit 1
fi

HEADROOM_GIB="${TTS_HEADROOM_GIB:-8}"
CLEANUP_FLAG=()
if [[ "${TTS_CLEANUP:-0}" != "1" ]]; then
  CLEANUP_FLAG=(--skip-cleanup)
fi

echo "Staging ${#TARGETS[@]} targets from ${CONFIG}"
echo "Model root:   ${TTS_MODEL_ROOT}"
echo "Runtime root: ${TTS_RUNTIME_ROOT}"
echo "Headroom:     ${HEADROOM_GIB} GiB"
echo "Cleanup:      ${TTS_CLEANUP:-0}"

FAILED=()
for target in "${TARGETS[@]}"; do
  echo
  echo "==> Staging target: ${target}"
  if uv run tts-stage \
      --config "${CONFIG}" \
      --target "${target}" \
      --model-root "${TTS_MODEL_ROOT}" \
      --runtime-root "${TTS_RUNTIME_ROOT}" \
      --headroom-gib "${HEADROOM_GIB}" \
      "${CLEANUP_FLAG[@]}"; then
    echo "==> Done: ${target}"
  else
    echo "==> FAILED: ${target} (left on disk for inspection, not cleaned up)" >&2
    FAILED+=("${target}")
    if [[ "${STOP_ON_FAILURE:-1}" == "1" ]]; then
      echo "Stopping (set STOP_ON_FAILURE=0 to continue with remaining targets)." >&2
      break
    fi
  fi
done

if ((${#FAILED[@]})); then
  echo
  echo "Targets that failed: ${FAILED[*]}" >&2
  exit 1
fi

echo
echo "All targets staged successfully."
