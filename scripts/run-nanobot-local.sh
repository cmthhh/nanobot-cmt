#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/config.local.qwen.json"

export NANOBOT_STREAM_IDLE_TIMEOUT_S="${NANOBOT_STREAM_IDLE_TIMEOUT_S:-180}"
export NANOBOT_LLM_TIMEOUT_S="${NANOBOT_LLM_TIMEOUT_S:-300}"

exec nanobot agent --config "$CONFIG" "$@"
