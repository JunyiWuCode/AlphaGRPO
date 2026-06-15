#!/usr/bin/env bash
# Deploy HPSv3 reward model server using uv virtual environment.
#
# Usage:
#   bash scripts/serve_hpsv3.sh [host] [port] [device] [checkpoint]
#
# Arguments (all optional):
#   host       — bind address        (default: 0.0.0.0)
#   port       — listen port         (default: 18087)
#   device     — cuda / cuda:0 / cpu (default: cuda)
#   checkpoint — path to HPSv3.safetensors; auto-downloads from HF if omitted

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${HPSV3_VENV_DIR:-.venvs/hpsv3}"

HOST="${1:-0.0.0.0}"
PORT="${2:-18087}"
DEVICE="${3:-cuda}"
CHECKPOINT="${4:-}"

# ── 1. Ensure uv is available ──────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[hpsv3] uv not found — installing ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── 2. Create venv (skip if exists) ────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[hpsv3] Creating venv at ${VENV_DIR} ..."
    uv venv "$VENV_DIR" --python 3.10
else
    echo "[hpsv3] Venv already exists at ${VENV_DIR}, skipping creation."
fi

# ── 3. Install / sync dependencies ─────────────────────────────────
echo "[hpsv3] Syncing dependencies ..."
source "$VENV_DIR/bin/activate"
uv pip install hpsv3 flask

# ── 4. Ensure HPSv3 weights are available ──────────────────────────
WEIGHTS_DIR="${HPSV3_WEIGHTS_DIR:-./HPSv3}"

if [ -z "$CHECKPOINT" ] && [ ! -f "$WEIGHTS_DIR/HPSv3.safetensors" ]; then
    echo "[hpsv3] Weights not found at ${WEIGHTS_DIR}, downloading from HuggingFace ..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('MizzenAI/HPSv3', local_dir='${WEIGHTS_DIR}')
"
fi

CHECKPOINT="${CHECKPOINT:-$WEIGHTS_DIR/HPSv3.safetensors}"

# ── 5. Launch server ────────────────────────────────────────────────
echo "[hpsv3] Starting server  host=${HOST}  port=${PORT}  device=${DEVICE}"

exec python "${SCRIPT_DIR}/hpsv3_server.py" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE" \
    --checkpoint "$CHECKPOINT"
