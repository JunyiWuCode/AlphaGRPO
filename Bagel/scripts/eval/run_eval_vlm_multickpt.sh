 # Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

set -e

# Set proxy and API key
# export OPENAI_API_KEY=$openai_api_key

model_path=ByteDance-Seed/BAGEL-7B-MoT

export OPENAI_API_KEY=""  # set up your OpenAI API key
export OPENAI_API_URL=""  # set up the OpenAI API base url
export API_URL=""  # set up the OpenAI-compatible API endpoint url
export AZURE_OPENAI_ENDPOINT=""  # set up the Azure OpenAI endpoint url

export GPUS=8

# write down the config name here.
jobname=
checkpoint_list=(379)

# set up the benchmarks names
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


if [[ "$DATASETS_STR" == *'mmmu-val'* ]] && [ ! -d ./data/MMMU/ ]; then
    cp -r ./data/ ./data/MMMU/
fi

if [[ "$DATASETS_STR" == *'mmbench-dev-en'* ]] && [ ! -d ./data/mmbench/ ]; then
    cp -r ./data/ ./data/mmbench/
fi

if [[ "$DATASETS_STR" == *'mmmu-val'* ]] && [ ! -d ./data/MMMU/ ]; then
    cp -r ./data/ ./data/MMMU/
fi

if [[ "$DATASETS_STR" == *'mathvista-testmini'* ]] && [ ! -d ./data/MathVista/ ]; then
    cp -r ./data/ ./data/MathVista/
fi

output_root=./output
output_root=./output/${jobname}

for checkpoint_num in "${checkpoint_list[@]}"
# for checkpoint_num in {99,199,399}
do
    output_path=${output_root}/checkpoint_${checkpoint_num}/ 
    checkpoint_dir=./output${jobname}/
    mnt_output_path=${checkpoint_dir}eval_vlm_output
    if [ ! -d $mnt_output_path ]; then
        mkdir -p $mnt_output_path
    fi

    export BAGEL_LORA_PATH=${checkpoint_dir}/checkpoint-${checkpoint_num}/hf_model/


    # results_file=$mnt_output_path/checkpoint_${checkpoint_num}/mathvista-testmini/results.json
    # out_dir=$mnt_output_path/checkpoint_${checkpoint_num}/mathvista-testmini/

    results_file=$output_path/mathvista-testmini/results.json
    out_dir=$output_path/mathvista-testmini/
    # python eval/vlm/eval/mathvista/extract_answer_mp.py --output_file ${results_file} --output_dir ${out_dir} 
    python eval/vlm/eval/mathvista/calculate_score.py --output_file ${results_file} --output_dir ${out_dir} --score_file score.json

    exit 0 
    bash scripts/eval/eval_vlm.sh \
        $output_path \
        --model-path $model_path

    target_path="${mnt_output_path}$(basename ${output_path})"

    # if [ -d "$target_path" ]; then
    #     # 如果存在，复制内容到目标路径
    #     echo "Target directory exists, copying contents to ${target_path}"
    #     cp -r "${output_path}"* "${target_path}/"
    #     # 复制完成后删除原路径
    #     rm -rf "${output_path}"
    # else
        # 如果不存在，直接移动
    #     echo "Target directory does not exist, moving ${output_path} to ${mnt_output_path}"
    #     mv "${output_path}" "${mnt_output_path}"
    # # fi
    # echo move the results to ${target_path}
done