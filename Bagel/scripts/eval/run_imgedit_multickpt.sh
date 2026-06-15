# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -e
pip install sympy httpx==0.23.3

export NCCL_TIMEOUT=36000000

GPUS=8
model_path=ByteDance-Seed/BAGEL-7B-MoT

export OPENAI_API_URL=  # set up the OpenAI-compatible API endpoint url
export OPENAI_API_KEY=""  # set up your OpenAI API key
export AZURE_OPENAI_ENDPOINT=  # set up the Azure OpenAI endpoint url
export MODEL_NAME=gpt-4.1-2025-04-14

if [ ! -e $(pwd)/eval/gen/imgedit/Benchmark/ ]; then
    ln -s ./data/ $(pwd)/eval/gen/imgedit/Benchmark
fi

use_think=False
jobname=''
checkpoint_list=(**)

output_root=./output/${jobname}
checkpoint_dir=./output${jobname}
echo $(dirname $checkpoint_dir)


for checkpoint_num in "${checkpoint_list[@]}"
do
    task='imgedit'

    output_path=${output_root}/checkpoint-${checkpoint_num}
    if [ "$use_think" = True ]; then
        mnt_output_path=${checkpoint_dir}/eval_${task}_think_output
    else
        mnt_output_path=${checkpoint_dir}/eval_${task}_output
    fi

    if [ ! -d $mnt_output_path ]; then
        mkdir -p $mnt_output_path
    fi

    export BAGEL_LORA_PATH=${checkpoint_dir}/checkpoint-${checkpoint_num}/hf_model/

    # generate images
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=$GPUS \
        --master_addr=127.0.0.1 \
        --master_port=12345 \
        ./eval/gen/gen_images_mp_imgedit.py \
        --output_dir $output_path/bagel \
        --metadata_file ./eval/gen/imgedit/Benchmark/singleturn/singleturn.json \
        --max_latent_size 64 \
        --model-path $model_path

    pip install httpx==0.23.0 openai==1.87.0

    # calculate score
    python ./eval/gen/imgedit/basic_bench.py \
        --result_img_folder $output_path/bagel \
        --edit_json ./eval/gen/imgedit/Benchmark/singleturn/singleturn.json \
        --origin_img_root ./eval/gen/imgedit/Benchmark/singleturn \
        --num_processes 4 \
        --prompts_json ./eval/gen/imgedit/Benchmark/singleturn/judge_prompt.json


    # summarize score
    python ./eval/gen/imgedit/step1_get_avgscore.py \
        --result_json $output_path/bagel/result.json \
        --average_score_json $output_path/bagel/average_score.json

    python ./eval/gen/imgedit/step2_typescore.py \
        --average_score_json  $output_path/bagel/average_score.json \
        --edit_json ./eval/gen/imgedit/Benchmark/singleturn/singleturn.json \
        --typescore_json $output_path/bagel/typescore.json

    echo "Moving ${output_path} to ${mnt_output_path}"
    mv "${output_path}" "${mnt_output_path}"
done