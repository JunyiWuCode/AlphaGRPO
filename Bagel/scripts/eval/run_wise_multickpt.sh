# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -e
pip install sympy httpx==0.23.3
pip install openai -U 

GPUS=8
model_path=ByteDance-Seed/BAGEL-7B-MoT

export OPENAI_API_URL=  # set up the OpenAI-compatible API endpoint url
export OPENAI_API_KEY=""  # set up your OpenAI API key
export API_URL=$OPENAI_API_URL
export AZURE_OPENAI_ENDPOINT=  # set up the Azure OpenAI endpoint url

resolution=1024
use_think=True


jobname=
resolution=1024
checkpoint_list=(379)


for checkpoint_num in "${checkpoint_list[@]}"
do
    checkpoint_dir=./output${jobname}
    if [ "$use_think" = True ]; then
        output_root=./output${resolution}_think_output/${jobname}
        mnt_output_path=${checkpoint_dir}/eval_wise_${resolution}_think_output
    else
        output_root=./output${resolution}_output/${jobname}
        mnt_output_path=${checkpoint_dir}/eval_wise_${resolution}_output
    fi

    if [ "$resolution" = 512 ]; then
        mnt_output_path=${mnt_output_path}_512px
    fi

    echo $(dirname $checkpoint_dir)

    output_path=${output_root}/checkpoint-${checkpoint_num}
    image_dir=$output_path/images

    if [ ! -d $mnt_output_path ]; then
        mkdir -p $mnt_output_path
    fi

    export BAGEL_LORA_PATH=${checkpoint_dir}/checkpoint-${checkpoint_num}/hf_model/

    # generate images
    if [ "$use_think" = True ]; then
        torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=$GPUS \
        --master_addr=127.0.0.1 \
        --master_port=12345 \
        ./eval/gen/gen_images_mp_wise.py \
        --output_dir $output_path/images \
        --metadata_file ./eval/gen/wise/final_data.json \
        --resolution $resolution \
        --max_latent_size 64 \
        --model-path $model_path \
        --think
    else
        torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=$GPUS \
        --master_addr=127.0.0.1 \
        --master_port=12345 \
        ./eval/gen/gen_images_mp_wise.py \
        --output_dir $output_path/images \
        --metadata_file ./eval/gen/wise/final_data.json \
        --resolution $resolution \
        --max_latent_size 64 \
        --model-path $model_path
    fi

    # output_path=${mnt_output_path}/checkpoint-${checkpoint_num}
    # image_dir=$output_path/images

    # calculate score
    echo gpt evaluating.
    python3 eval/gen/wise/gpt_eval_mp.py \
            --json_path eval/gen/wise/data/cultural_common_sense.json \
            --image_dir $image_dir \
            --output_dir $output_path

    python3 eval/gen/wise/gpt_eval_mp.py \
            --json_path eval/gen/wise/data/spatio-temporal_reasoning.json \
            --image_dir $image_dir \
            --output_dir $output_path

    python3 eval/gen/wise/gpt_eval_mp.py \
            --json_path eval/gen/wise/data/natural_science.json \
            --image_dir $image_dir \
            --output_dir $output_path

    python3 eval/gen/wise/cal_score.py \
            --output_dir $output_path

    echo "Moving ${output_path} to ${mnt_output_path}"
    mv "${output_path}" "${mnt_output_path}"
done

