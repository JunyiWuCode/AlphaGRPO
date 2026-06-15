# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -e

GPUS=4
model_path=ByteDance-Seed/BAGEL-7B-MoT

jobname=baseline

output_root=./output/${jobname}
output_path=${output_root}

resolution=1024
use_think=True

mnt_output_path=./output
if [ "$resolution" = "512" ]; then
    mnt_output_path=${mnt_output_path}_512px
fi
if [ "$use_think" = True ]; then
    mnt_output_path=${mnt_output_path}_think
fi


if [ "$use_think" = True ]; then
    # generate images
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=$GPUS \
        --master_addr=127.0.0.1 \
        --master_port=12346 \
        ./eval/gen/gen_images_mp_geneval2.py \
        --output_dir $output_path/images \
        --metadata_file ./eval/gen/geneval2/geneval2_data.jsonl \
        --batch_size 1 \
        --num_images 4 \
        --resolution $resolution \
        --max_latent_size 64 \
        --model-path $model_path \
        --think
else
    # generate images
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=$GPUS \
        --master_addr=127.0.0.1 \
        --master_port=12346 \
        ./eval/gen/gen_images_mp_geneval2.py \
        --output_dir $output_path/images \
        --metadata_file ./eval/gen/geneval2/geneval2_data.jsonl \
        --batch_size 1 \
        --num_images 4 \
        --resolution $resolution \
        --max_latent_size 64 \
        --model-path $model_path 
fi


python ./eval/gen/geneval2/evaluation.py \
    --benchmark_data ./eval/gen/geneval2/geneval2_data.jsonl \
    --image_filepath_data  $output_path/images/geneval_image_map.json \
    --method soft_tifa_gm \
    --output_file $output_path/score_lists.json


python ./eval/gen/geneval2/soft_tifa_analysis.py \
    --benchmark_data ./eval/gen/geneval2/geneval2_data.jsonl \
    --score_data $output_path/score_lists.json

echo move $output_paht to $mnt_output_path
mv $output_path $mnt_output_path

 