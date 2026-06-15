# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -e

GPUS=8
resolution=512
model_path=ByteDance-Seed/BAGEL-7B-MoT

jobname=baseline

output_root=./output

if [ "$resolution" = "512" ]; then
    output_root=${output_root}_512px
fi
output_path=${output_root}

# generate images
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_mp.py \
    --output_dir $output_path/images \
    --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata_long.jsonl \
    --batch_size 1 \
    --num_images 4 \
    --resolution $resolution \
    --max_latent_size 64 \
    --model-path $model_path 


# calculate score
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
python ./eval/gen/geneval/evaluation/summary_scores.py $output_path/results.jsonl


if [ "$resolution" = "512" ]; then
    mv $output_path ./output
else 
    mv $output_path ./output
fi

