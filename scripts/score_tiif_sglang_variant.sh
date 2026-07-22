#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 VARIANT PORT RUN_DIR OUTPUT_ROOT" >&2
  exit 2
fi

VARIANT=$1
PORT=$2
RUN_DIR=$3
OUTPUT_ROOT=$4
ROOT=${ALPHAGRPO_ROOT:-/home/hcai/workspace/code/junyiwu/AlphaGRPO}
ENV_PREFIX=${ALPHAGRPO_ENV:-/home/hcai/workspace/anaconda3/envs/alpha_grpo}
MODEL=${TIIF_JUDGE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
RESULT_DIR=${OUTPUT_ROOT}/${VARIANT}/tiif/results_qwen3_vl_8b
SERVER_LOG=${RUN_DIR}/${VARIANT}_server.log

mkdir -p "${RESULT_DIR}"
export PYTHON_BIN=${ENV_PREFIX}/bin/python
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NO_PROXY=127.0.0.1,localhost
export no_proxy=${NO_PROXY}

bash "${ROOT}/scripts/serve_spectrareward_sglang.sh" 0.0.0.0 "${PORT}" "${MODEL}" 8 1 \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for attempt in $(seq 1 120); do
  if curl --noproxy '*' --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "TIIF judge is healthy after ${attempt} checks"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "TIIF judge exited during startup" >&2
    tail -n 100 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if [[ ${attempt} -eq 120 ]]; then
    echo "Timed out waiting for TIIF judge" >&2
    tail -n 100 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 10
done

"${ENV_PREFIX}/bin/python" "${ROOT}/Bagel/eval/gen/tiif/eval_with_vlm.py" \
  --jsonl_dir "${ROOT}/Bagel/eval/gen/tiif/testmini_eval_prompts" \
  --generation_jsonl_dir "${ROOT}/Bagel/eval/gen/tiif/testmini_prompts" \
  --manifest_file "${OUTPUT_ROOT}/${VARIANT}/manifest.jsonl" \
  --image_dir "${OUTPUT_ROOT}/${VARIANT}/tiif/images" \
  --eval_model sd35 \
  --output_dir "${RESULT_DIR}/raw" \
  --api_key local-sglang \
  --base_url "http://127.0.0.1:${PORT}/v1" \
  --model "${MODEL}" \
  --max_workers "${TIIF_MAX_WORKERS:-32}" \
  --temperature 0 \
  --max_tokens_per_question 24 \
  --allow_extra_answers \
  --seed 0 \
  --max_retries 8

"${ENV_PREFIX}/bin/python" "${ROOT}/scripts/summarize_tiif_results.py" \
  "${RESULT_DIR}/raw" \
  "${RESULT_DIR}/summary.json" \
  --judge-model "${MODEL}" \
  --expected-files-per-length 277 \
  --expected-questions-per-length 1446
