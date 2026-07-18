#!/usr/bin/env bash
#SBATCH --job-name=sr-dpg
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=experiments/spectrareward_sd35_benchmarks/score-dpg-%j.out
#SBATCH --error=experiments/spectrareward_sd35_benchmarks/score-dpg-%j.err

set -euo pipefail

ROOT=${ALPHAGRPO_ROOT:-/home/hcai/workspace/code/junyiwu/AlphaGRPO}
ENV_PREFIX=${ALPHAGRPO_ENV:-/home/hcai/workspace/anaconda3/envs/alpha_grpo}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/experiments/spectrareward_sd35_benchmarks}
DPG_CSV=${ROOT}/Bagel/eval/gen/dpg/dpg_bench.csv
DPG_SCRIPT=${ROOT}/Bagel/eval/gen/dpg/compute_dpg_bench.py
RUN_DIR=${OUTPUT_ROOT}/score_jobs/${SLURM_JOB_ID}

mkdir -p "${RUN_DIR}"
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#NODES[@]} -ne 2 ]]; then
  printf 'Expected two nodes, got: %s\n' "${NODES[*]}" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

score_variant() {
  local node=$1
  local variant=$2
  local port=$3
  local image_dir=${OUTPUT_ROOT}/${variant}/dpg/images
  local result_dir=${OUTPUT_ROOT}/${variant}/dpg/results
  local result_file=${result_dir}/dpg_results.txt
  mkdir -p "${result_dir}"
  srun --nodes=1 --ntasks=1 --nodelist="${node}" --gpus-per-node=8 --exclusive \
    bash -lc "cd '${ROOT}' && '${ENV_PREFIX}/bin/accelerate' launch --multi_gpu --num_machines 1 --num_processes 8 --main_process_port '${port}' '${DPG_SCRIPT}' --image-root-path '${image_dir}' --resolution 512 --csv '${DPG_CSV}' --res-path '${result_file}' --pic-num 1 --vqa-model mplug && '${ENV_PREFIX}/bin/python' scripts/summarize_dpg_result.py '${result_file}' '${result_dir}/summary.json' --expected-images 1065" \
    >"${RUN_DIR}/${variant}.log" 2>&1 &
}

printf 'job=%s baseline_node=%s rl_node=%s\n' "${SLURM_JOB_ID}" "${NODES[0]}" "${NODES[1]}"
score_variant "${NODES[0]}" baseline 29531
BASELINE_PID=$!
score_variant "${NODES[1]}" spectrareward_step380 29532
RL_PID=$!

cleanup() {
  kill "${BASELINE_PID}" "${RL_PID}" 2>/dev/null || true
}
trap cleanup EXIT

status=0
wait "${BASELINE_PID}" || status=$?
wait "${RL_PID}" || status=$?
printf 'paired_dpg_exit_status=%s\n' "${status}"
exit "${status}"
