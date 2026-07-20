#!/usr/bin/env bash
#SBATCH --job-name=sr-tiif-qwen
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=experiments/spectrareward_sd35_benchmarks/score-tiif-qwen-%j.out
#SBATCH --error=experiments/spectrareward_sd35_benchmarks/score-tiif-qwen-%j.err

set -euo pipefail

ROOT=${ALPHAGRPO_ROOT:-/home/hcai/workspace/code/junyiwu/AlphaGRPO}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/experiments/spectrareward_sd35_benchmarks}
RUN_DIR=${OUTPUT_ROOT}/score_jobs/${SLURM_JOB_ID}
mkdir -p "${RUN_DIR}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#NODES[@]} -ne 2 ]]; then
  printf 'Expected two nodes, got: %s\n' "${NODES[*]}" >&2
  exit 1
fi

score_variant() {
  local node=$1
  local variant=$2
  local port=$3
  srun --nodes=1 --ntasks=1 --nodelist="${node}" --gpus-per-node=8 --exclusive \
    bash -lc "cd '${ROOT}' && bash scripts/score_tiif_sglang_variant.sh '${variant}' '${port}' '${RUN_DIR}' '${OUTPUT_ROOT}'" \
    >"${RUN_DIR}/${variant}.log" 2>&1 &
}

printf 'job=%s baseline_node=%s rl_node=%s judge=%s\n' \
  "${SLURM_JOB_ID}" "${NODES[0]}" "${NODES[1]}" "${TIIF_JUDGE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
score_variant "${NODES[0]}" baseline 18110
BASELINE_PID=$!
score_variant "${NODES[1]}" spectrareward_step380 18110
RL_PID=$!

cleanup() {
  kill "${BASELINE_PID}" "${RL_PID}" 2>/dev/null || true
}
trap cleanup EXIT

status=0
wait "${BASELINE_PID}" || status=$?
wait "${RL_PID}" || status=$?
printf 'paired_tiif_qwen_exit_status=%s\n' "${status}"
exit "${status}"
