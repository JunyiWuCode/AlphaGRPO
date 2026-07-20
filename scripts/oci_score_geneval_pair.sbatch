#!/usr/bin/env bash
#SBATCH --job-name=sr-geneval
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=experiments/spectrareward_sd35_benchmarks/score-geneval-%j.out
#SBATCH --error=experiments/spectrareward_sd35_benchmarks/score-geneval-%j.err

set -euo pipefail

ROOT=${ALPHAGRPO_ROOT:-/home/hcai/workspace/code/junyiwu/AlphaGRPO}
ENV_PREFIX=${GENEVAL_ENV:-/home/hcai/workspace/anaconda3/envs/flow_opd_geneval_reward}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/experiments/spectrareward_sd35_benchmarks}
MMD_ROOT=${MMD_ROOT:-/home/hcai/workspace/code/junyiwu/FAR-RL/third_party/reward-server/mmdetection}
MODEL_ROOT=${GENEVAL_MODEL_ROOT:-/home/hcai/workspace/code/junyiwu/FAR-RL/third_party/reward-server/model/mask2former2}
MODEL_CONFIG=${MMD_ROOT}/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py
EVAL_SCRIPT=${ROOT}/Bagel/eval/gen/geneval/evaluation/evaluate_images_mp.py
RUN_DIR=${OUTPUT_ROOT}/score_jobs/${SLURM_JOB_ID}

mkdir -p "${RUN_DIR}"
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#NODES[@]} -ne 2 ]]; then
  printf 'Expected two nodes, got: %s\n' "${NODES[*]}" >&2
  exit 1
fi
for path in "${ENV_PREFIX}/bin/torchrun" "${MODEL_CONFIG}" \
  "${MODEL_ROOT}/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"; do
  [[ -e ${path} ]] || { printf 'Missing required path: %s\n' "${path}" >&2; exit 1; }
done

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

score_variant() {
  local node=$1
  local variant=$2
  local port=$3
  local image_dir=${OUTPUT_ROOT}/${variant}/geneval/images
  local result_dir=${OUTPUT_ROOT}/${variant}/geneval/results
  mkdir -p "${result_dir}"
  srun --nodes=1 --ntasks=1 --nodelist="${node}" --gpus-per-node=8 --exclusive \
    bash -lc "cd '${ROOT}' && export PYTHONPATH='${MMD_ROOT}':\${PYTHONPATH:-} && '${ENV_PREFIX}/bin/torchrun' --standalone --nnodes=1 --nproc-per-node=8 --master-port '${port}' '${EVAL_SCRIPT}' '${image_dir}' --outfile '${result_dir}/results.jsonl' --model-path '${MODEL_ROOT}' --model-config '${MODEL_CONFIG}' && '${ENV_PREFIX}/bin/python' scripts/summarize_geneval_result.py '${result_dir}/results.jsonl' '${result_dir}/summary.json' --expected-images 2212 --expected-prompts 553" \
    >"${RUN_DIR}/${variant}.log" 2>&1 &
}

printf 'job=%s baseline_node=%s rl_node=%s\n' "${SLURM_JOB_ID}" "${NODES[0]}" "${NODES[1]}"
score_variant "${NODES[0]}" baseline 29631
BASELINE_PID=$!
score_variant "${NODES[1]}" spectrareward_step380 29632
RL_PID=$!

cleanup() {
  kill "${BASELINE_PID}" "${RL_PID}" 2>/dev/null || true
}
trap cleanup EXIT

status=0
wait "${BASELINE_PID}" || status=$?
wait "${RL_PID}" || status=$?
printf 'paired_geneval_exit_status=%s\n' "${status}"
exit "${status}"
