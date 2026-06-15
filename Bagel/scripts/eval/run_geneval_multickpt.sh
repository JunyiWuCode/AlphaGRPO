# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -e
export NCCL_TIMEOUT=36000000

GPUS=8
model_path=ByteDance-Seed/BAGEL-7B-MoT

export OPENAI_API_KEY=""  # set up your OpenAI API key
export API_URL=""  # set up the OpenAI-compatible API endpoint url

use_think=False
jobname=
checkpoint_list=(379)
prompt_set='long'
resolution=1024  # or 512

output_root=./output/${jobname}
checkpoint_dir=./output${jobname}
echo $(dirname $checkpoint_dir)

for checkpoint_num in "${checkpoint_list[@]}"
# for checkpoint_num in 259
do
    task='geneval'
    output_path=${output_root}/checkpoint-${checkpoint_num}

    # rm  -rf $output_path 
    if [ "$use_think" = True ]; then
        mnt_output_path=${checkpoint_dir}/eval_geneval_${prompt_set}_think_output
    else
        mnt_output_path=${checkpoint_dir}/eval_geneval_${prompt_set}_output
    fi

    if [ "$resolution" = "512" ]; then
        mnt_output_path=${mnt_output_path}_512px
    fi

    if [ ! -d $mnt_output_path ]; then
        mkdir -p $mnt_output_path
    fi

    # mv "${output_path}" "${mnt_output_path}"

    export BAGEL_LORA_PATH=${checkpoint_dir}/checkpoint-${checkpoint_num}/hf_model/

    # generate images
    if [ "$prompt_set" = "long" ]; then
        metadata_file=./eval/gen/geneval/prompts/evaluation_metadata_long.jsonl
    else
        metadata_file=./eval/gen/geneval/prompts/evaluation_metadata.jsonl
    fi

    if [ "$use_think" = True ]; then
        torchrun \
            --nnodes=1 \
            --node_rank=0 \
            --nproc_per_node=$GPUS \
            --master_addr=127.0.0.1 \
            --master_port=12345 \
            ./eval/gen/gen_images_mp.py \
            --output_dir $output_path/images \
            --batch_size 1 \
            --num_images 4 \
            --resolution $resolution \
            --max_latent_size 64 \
            --model-path $model_path \
            --metadata_file $metadata_file \
            --think
    else
        torchrun \
            --nnodes=1 \
            --node_rank=0 \
            --nproc_per_node=$GPUS \
            --master_addr=127.0.0.1 \
            --master_port=12345 \
            ./eval/gen/gen_images_mp.py \
            --output_dir $output_path/images \
            --batch_size 1 \
            --num_images 4 \
            --resolution $resolution \
            --max_latent_size 64 \
            --model-path $model_path \
            --metadata_file $metadata_file 
    fi

    # output_path=$mnt_output_path/checkpoint-${checkpoint_num}
    # # calculate score
    echo calculating score. Save resutls in $output_path/results.jsonl
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=$GPUS \
        --master_addr=127.0.0.1 \
        --master_port=12345 \
        ./eval/gen/geneval/evaluation/evaluate_images_mp.py \
        $output_path/images \
        --outfile $output_path/results.jsonl \
        --model-path ./eval/gen/geneval/model \
        --model-config ./mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py

    # summarize score
    echo summarize score on $output_path
    python ./eval/gen/geneval/evaluation/summary_scores.py $output_path/results.jsonl
    echo "Moving ${output_path} to ${mnt_output_path}"
    mv "${output_path}" "${mnt_output_path}"
done

