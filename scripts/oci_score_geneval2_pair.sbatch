#!/usr/bin/env bash
#SBATCH --job-name=sr-geneval2
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=experiments/spectrareward_sd35_benchmarks/score-geneval2-%j.out
#SBATCH --error=experiments/spectrareward_sd35_benchmarks/score-geneval2-%j.err

set -euo pipefail

ROOT=${ALPHAGRPO_ROOT:-/home/hcai/workspace/code/junyiwu/AlphaGRPO}
ENV_PREFIX=${ALPHAGRPO_ENV:-/home/hcai/workspace/anaconda3/envs/alpha_grpo}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/experiments/spectrareward_sd35_benchmarks}
BENCHMARK_DATA=${ROOT}/Bagel/eval/gen/geneval2/geneval2_data.jsonl
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
  local image_map=${OUTPUT_ROOT}/${variant}/geneval2/images/geneval_image_map.json
  local result_dir=${OUTPUT_ROOT}/${variant}/geneval2/results
  srun --nodes=1 --ntasks=1 --nodelist="${node}" --gpus-per-node=8 --exclusive \
    bash -lc "cd '${ROOT}' && '${ENV_PREFIX}/bin/torchrun' --standalone --nnodes=1 --nproc-per-node=8 scripts/eval_geneval2_distributed.py --benchmark-data '${BENCHMARK_DATA}' --image-map '${image_map}' --output-dir '${result_dir}'" \
    >"${RUN_DIR}/${variant}.log" 2>&1 &
}

printf 'job=%s baseline_node=%s rl_node=%s\n' "${SLURM_JOB_ID}" "${NODES[0]}" "${NODES[1]}"
score_variant "${NODES[0]}" baseline
BASELINE_PID=$!
score_variant "${NODES[1]}" spectrareward_step380
RL_PID=$!

cleanup() {
  kill "${BASELINE_PID}" "${RL_PID}" 2>/dev/null || true
}
trap cleanup EXIT

status=0
wait "${BASELINE_PID}" || status=$?
wait "${RL_PID}" || status=$?
printf 'paired_geneval2_exit_status=%s\n' "${status}"
exit "${status}"
