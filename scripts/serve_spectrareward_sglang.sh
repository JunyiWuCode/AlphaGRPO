#!/usr/bin/env bash
set -euo pipefail

# Serve Qwen3-VL for SpectraReward through SGLang's native /generate endpoint.
# Usage: bash scripts/serve_spectrareward_sglang.sh HOST PORT [MODEL] [DP] [TP]

HOST=${1:-0.0.0.0}
PORT=${2:-18090}
MODEL=${3:-${SPECTRAREWARD_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}}
DP=${4:-${SPECTRAREWARD_DP:-8}}
TP=${5:-${SPECTRAREWARD_TP:-1}}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}

echo "[spectrareward-sglang] model=${MODEL} host=${HOST} port=${PORT} dp=${DP} tp=${TP}"

exec python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dp "${DP}" \
  --tp "${TP}" \
  --trust-remote-code \
  --enable-multimodal \
  --mem-fraction-static "${SPECTRAREWARD_MEM_FRACTION:-0.85}" \
  --chunked-prefill-size "${SPECTRAREWARD_CHUNKED_PREFILL_SIZE:-2048}" \
  --max-prefill-tokens "${SPECTRAREWARD_MAX_PREFILL_TOKENS:-32768}" \
  --context-length "${SPECTRAREWARD_CONTEXT_LENGTH:-4096}" \
  --max-running-requests "${SPECTRAREWARD_MAX_RUNNING_REQUESTS:-256}" \
  --max-queued-requests "${SPECTRAREWARD_MAX_QUEUED_REQUESTS:-4096}" \
  --schedule-policy lpm \
  --log-requests-level 0
