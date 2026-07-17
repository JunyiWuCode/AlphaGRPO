#!/usr/bin/env bash
#SBATCH --job-name=sr-sd35-pair
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=experiments/spectrareward_sd35_benchmarks/slurm-pair-%j.out
#SBATCH --error=experiments/spectrareward_sd35_benchmarks/slurm-pair-%j.err

set -euo pipefail

ROOT=${ALPHAGRPO_ROOT:-/home/hcai/workspace/code/junyiwu/AlphaGRPO}
ENV_PREFIX=${ALPHAGRPO_ENV:-/home/hcai/workspace/anaconda3/envs/alpha_grpo}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/experiments/spectrareward_sd35_benchmarks}
LORA_PATH=${LORA_PATH:-${ROOT}/experiments/spectrareward_sd35/training/checkpoints/checkpoint-380/lora_ema}
BENCHMARKS=${BENCHMARKS:-geneval,tiif,dpg,geneval2,wise}
BATCH_SIZE=${BATCH_SIZE:-48}
MAX_TASKS=${MAX_TASKS:-0}
RUN_DIR=${OUTPUT_ROOT}/jobs/${SLURM_JOB_ID}

mkdir -p "${RUN_DIR}"
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#NODES[@]} -ne 2 ]]; then
  printf 'Expected two nodes, got: %s\n' "${NODES[*]}" >&2
  exit 1
fi

BASELINE_NODE=${NODES[0]}
RL_NODE=${NODES[1]}
export PYTHONPATH=${ROOT}/alpha_grpo:${ROOT}/third_party/advantage_weighted_matching:${PYTHONPATH:-}
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

printf 'job=%s baseline_node=%s rl_node=%s batch_size=%s\n' \
  "${SLURM_JOB_ID}" "${BASELINE_NODE}" "${RL_NODE}" "${BATCH_SIZE}"
printf 'benchmarks=%s output_root=%s lora_path=%s\n' \
  "${BENCHMARKS}" "${OUTPUT_ROOT}" "${LORA_PATH}"

srun --nodes=1 --ntasks=1 --nodelist="${BASELINE_NODE}" --gpus-per-node=8 --exclusive \
  bash -lc "cd '${ROOT}' && '${ENV_PREFIX}/bin/torchrun' --standalone --nnodes=1 --nproc-per-node=8 scripts/eval_sd35_benchmarks.py --data-root Bagel/eval/gen --output-root '${OUTPUT_ROOT}' --variant baseline --benchmarks '${BENCHMARKS}' --resolution 512 --num-steps 16 --guidance-scale 4.0 --batch-size '${BATCH_SIZE}' --max-tasks '${MAX_TASKS}' --seed 0" \
  >"${RUN_DIR}/baseline.log" 2>&1 &
BASELINE_PID=$!

srun --nodes=1 --ntasks=1 --nodelist="${RL_NODE}" --gpus-per-node=8 --exclusive \
  bash -lc "cd '${ROOT}' && '${ENV_PREFIX}/bin/torchrun' --standalone --nnodes=1 --nproc-per-node=8 scripts/eval_sd35_benchmarks.py --data-root Bagel/eval/gen --output-root '${OUTPUT_ROOT}' --variant spectrareward_step380 --lora-path '${LORA_PATH}' --benchmarks '${BENCHMARKS}' --resolution 512 --num-steps 16 --guidance-scale 4.0 --batch-size '${BATCH_SIZE}' --max-tasks '${MAX_TASKS}' --seed 0" \
  >"${RUN_DIR}/spectrareward_step380.log" 2>&1 &
RL_PID=$!

cleanup() {
  kill "${BASELINE_PID}" "${RL_PID}" 2>/dev/null || true
}
trap cleanup EXIT

status=0
wait "${BASELINE_PID}" || status=$?
wait "${RL_PID}" || status=$?
printf 'paired_evaluation_exit_status=%s\n' "${status}"
exit "${status}"
