#!/usr/bin/env bash
# Serve the SpectraReward reward MLLM over HTTP (single port, server-side
# data-parallel). The training side talks to it via SPECTRAREWARD_URL.
#
# Usage:
#   bash scripts/serve_spectrareward.sh <model_id> [host] [port] [num_gpus]
#
# Example (8-way DP on one node, port 18090):
#   bash scripts/serve_spectrareward.sh Qwen/Qwen3-VL-30B-A3B-Instruct 0.0.0.0 18090 8
#
# Then on the training side:
#   export SPECTRAREWARD_URL=http://<server_ip>:18090
#
# Requires (in the active environment): torch, transformers, accelerate,
# pillow, flask. Override the first GPU index with START_GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_ID="${1:?usage: $0 <model_id> [host] [port] [num_gpus]}"
HOST="${2:-0.0.0.0}"
PORT="${3:-18090}"
NUM_GPUS="${4:-8}"
START_GPU="${START_GPU:-0}"

echo "[serve-spectrareward] model=${MODEL_ID} ${HOST}:${PORT} num_gpus=${NUM_GPUS} start_gpu=${START_GPU}"

exec python "${SCRIPT_DIR}/mllm_server.py" \
    --model-id "$MODEL_ID" \
    --host "$HOST" \
    --port "$PORT" \
    --num-gpus "$NUM_GPUS" \
    --start-gpu "$START_GPU"
