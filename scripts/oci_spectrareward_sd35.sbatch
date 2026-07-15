#!/usr/bin/env bash
#SBATCH --job-name=sr-sd35
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@600
#SBATCH --output=experiments/spectrareward_sd35/slurm-%j.out
#SBATCH --error=experiments/spectrareward_sd35/slurm-%j.err

set -euo pipefail

ROOT=${ALPHAGRPO_ROOT:-/home/hcai/workspace/code/junyiwu/AlphaGRPO}
RUN_ROOT=${RUN_ROOT:-${ROOT}/experiments/spectrareward_sd35}
RUN_DIR=${RUN_ROOT}/${SLURM_JOB_ID}
LOG_DIR=${LOG_DIR:-${RUN_ROOT}/training}
PORT=${SPECTRAREWARD_PORT:-18090}
MODEL=${SPECTRAREWARD_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}
ENV_PREFIX=${ALPHAGRPO_ENV:-/home/hcai/workspace/anaconda3/envs/alpha_grpo}
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#NODES[@]} -ne 2 ]]; then
  echo "Expected exactly two nodes, got: ${NODES[*]}" >&2
  exit 1
fi
POLICY_NODE=${NODES[0]}
REWARD_NODE=${NODES[1]}

export ALPHAGRPO_ROOT=${ROOT}
export LOG_DIR
export PYTHONPATH=${ROOT}/alpha_grpo:${ROOT}/third_party/advantage_weighted_matching:${PYTHONPATH:-}
export SPECTRAREWARD_MODEL_ID=${MODEL}
export SPECTRAREWARD_BACKEND=sglang
export SPECTRAREWARD_URL=http://${REWARD_NODE}:${PORT}
export SPECTRAREWARD_MAX_CONCURRENT=${SPECTRAREWARD_MAX_CONCURRENT:-8}
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHON_BIN=${ENV_PREFIX}/bin/python
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

echo "job=${SLURM_JOB_ID} policy_node=${POLICY_NODE} reward_node=${REWARD_NODE}"
echo "root=${ROOT} log_dir=${LOG_DIR} reward_url=${SPECTRAREWARD_URL}"

if [[ -z ${RESUME_FROM:-} ]] && compgen -G "${LOG_DIR}/checkpoints/checkpoint-*" > /dev/null; then
  export RESUME_FROM=${LOG_DIR}/checkpoints
  echo "Auto-resuming from ${RESUME_FROM}"
fi

srun --nodes=1 --ntasks=1 --nodelist="${REWARD_NODE}" --gpus-per-node=8 --exclusive \
  bash -lc "cd '${ROOT}' && bash scripts/serve_spectrareward_sglang.sh 0.0.0.0 '${PORT}' '${MODEL}' 8 1" \
  >"${RUN_DIR}/reward_server.log" 2>&1 &
REWARD_STEP_PID=$!

cleanup() {
  kill "${REWARD_STEP_PID}" 2>/dev/null || true
  wait "${REWARD_STEP_PID}" 2>/dev/null || true
  tail -n 100 "${RUN_DIR}/reward_server.log" || true
}
trap cleanup EXIT

for attempt in $(seq 1 120); do
  if curl --noproxy '*' --fail --silent "${SPECTRAREWARD_URL}/health" >/dev/null; then
    echo "SpectraReward server is healthy after ${attempt} checks"
    break
  fi
  if ! kill -0 "${REWARD_STEP_PID}" 2>/dev/null; then
    echo "Reward server exited during startup" >&2
    exit 1
  fi
  if [[ ${attempt} -eq 120 ]]; then
    echo "Timed out waiting for reward server" >&2
    exit 1
  fi
  sleep 10
done

srun --nodes=1 --ntasks=1 --nodelist="${POLICY_NODE}" --gpus-per-node=8 --exclusive \
  bash -lc "cd '${ROOT}' && '${ENV_PREFIX}/bin/accelerate' launch --multi_gpu --num_machines 1 --num_processes 8 --mixed_precision bf16 third_party/advantage_weighted_matching/scripts/train_sd3_awm.py --config third_party/advantage_weighted_matching/config/spectrareward_sd35.py" \
  2>&1 | tee "${RUN_DIR}/train.log"
