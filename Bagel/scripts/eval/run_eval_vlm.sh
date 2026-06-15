# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

set -e

# Set proxy and API key
# export OPENAI_API_KEY=$openai_api_key
model_path=ByteDance-Seed/BAGEL-7B-MoT

export OPENAI_API_KEY=""  # set up your OpenAI API key
export API_URL=""  # set up the OpenAI-compatible API endpoint url

DATASETS=("mme" "mmbench-dev-en" "mmvet" "mmmu-val" "mathvista-testmini" "mmvp")

DATASETS_STR="${DATASETS[*]}"
export DATASETS_STR


# Download benchmark datasets from HuggingFace if not present
if [[ "$DATASETS_STR" == *'mmmu-val'* ]] && [ ! -d ./data/MMMU/ ]; then
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='MMMU/MMMU', repo_type='dataset', local_dir='./data/MMMU/')"
fi

if [[ "$DATASETS_STR" == *'mmbench-dev-en'* ]] && [ ! -d ./data/mmbench/ ]; then
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='opencompass/MMBench', repo_type='dataset', local_dir='./data/mmbench/')"
fi

if [[ "$DATASETS_STR" == *'mathvista-testmini'* ]] && [ ! -d ./data/MathVista/ ]; then
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='AI4Math/MathVista', repo_type='dataset', local_dir='./data/MathVista/')"
fi


output_path=./output/

if [ ! -d $mnt_output_path ]; then
    mkdir -p $mnt_output_path
fi

bash scripts/eval/eval_vlm.sh \
    $output_path \
    --model-path $model_path
