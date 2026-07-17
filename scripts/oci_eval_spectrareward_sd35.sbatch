#!/usr/bin/env bash
#SBATCH --job-name=sr-sd35-eval
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=experiments/spectrareward_sd35_benchmarks/slurm-%j.out
#SBATCH --error=experiments/spectrareward_sd35_benchmarks/slurm-%j.err

set -euo pipefail

ROOT=${ALPHAGRPO_ROOT:-/home/hcai/workspace/code/junyiwu/AlphaGRPO}
ENV_PREFIX=${ALPHAGRPO_ENV:-/home/hcai/workspace/anaconda3/envs/alpha_grpo}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/experiments/spectrareward_sd35_benchmarks}
VARIANT=${VARIANT:?Set VARIANT to baseline or spectrareward_step380}
LORA_PATH=${LORA_PATH:-}
BENCHMARKS=${BENCHMARKS:-geneval,tiif,dpg,geneval2,wise}
MAX_TASKS=${MAX_TASKS:-0}
BATCH_SIZE=${BATCH_SIZE:-48}

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH=${ROOT}/alpha_grpo:${ROOT}/third_party/advantage_weighted_matching:${PYTHONPATH:-}
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

printf 'job=%s host=%s variant=%s benchmarks=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "${VARIANT}" "${BENCHMARKS}"
printf 'output_root=%s lora_path=%s\n' "${OUTPUT_ROOT}" "${LORA_PATH:-<baseline>}"

cd "${ROOT}"
"${ENV_PREFIX}/bin/torchrun" \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  scripts/eval_sd35_benchmarks.py \
  --data-root Bagel/eval/gen \
  --output-root "${OUTPUT_ROOT}" \
  --variant "${VARIANT}" \
  --lora-path "${LORA_PATH}" \
  --benchmarks "${BENCHMARKS}" \
  --resolution 512 \
  --num-steps 16 \
  --guidance-scale 4.0 \
  --batch-size "${BATCH_SIZE}" \
  --max-tasks "${MAX_TASKS}" \
  --seed 0
