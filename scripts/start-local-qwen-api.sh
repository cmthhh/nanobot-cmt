#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT/models/qwen/Qwen2___5-7B-Instruct"
DEVICE="${NANOBOT_LOCAL_DEVICE:-mps}"
DTYPE="${NANOBOT_LOCAL_DTYPE:-auto}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export NANOBOT_LOCAL_ENABLE_TOOLS="${NANOBOT_LOCAL_ENABLE_TOOLS:-0}"

exec python "$ROOT/scripts/local_qwen_mps_server.py" \
  --model-dir "$MODEL_DIR" \
  --served-model-name qwen2.5-7b \
  --port 8000 \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --max-input-tokens 8192 \
  --max-new-tokens 1024 \
  "$@"
