# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -e

GPUS=8
model_path=ByteDance-Seed/BAGEL-7B-MoT

jobname=baseline

output_root=./output
output_path=${output_root}

resolution=1024
mnt_output_path=./output

if [ "$resolution" = "512" ]; then
    mnt_output_path=${mnt_output_path}_512px
fi

source_output_path=${mnt_output_path}
mnt_output_path=${mnt_output_path}_reflect

# generate images
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12346 \
    ./eval/gen/gen_images_mp_reflect.py \
    --output_dir $output_path/images \
    --source_output_dir $source_output_path/images \
    --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata_long.jsonl \
    --batch_size 1 \
    --num_images 4 \
    --max_latent_size 64 \
    --model-path $model_path \
    --only_wrong

    # --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata.jsonl \

echo calculating score. Save resutls in $output_path/results.jsonl
# calculate score
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12346 \
    ./eval/gen/geneval/evaluation/evaluate_images_mp.py \
    $output_path/images \
    --outfile $output_path/results.jsonl \
    --model-path ./eval/gen/geneval/model \
    --model-config ./mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py

echo summarize score on $output_path

# summarize score
python ./eval/gen/geneval/evaluation/summary_scores.py $output_path/results.jsonl

echo "Moving ${output_path} to ${mnt_output_path}"
mv "${output_path}" "${mnt_output_path}"