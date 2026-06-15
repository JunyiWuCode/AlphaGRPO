# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -e

pip install httpx==0.23.0
pip install openai -U

model_path=ByteDance-Seed/BAGEL-7B-MoT

export OPENAI_API_URL=  # set up the OpenAI-compatible API endpoint url
export OPENAI_API_KEY=""  # set up your OpenAI API key
export API_URL=$OPENAI_API_URL

use_think=True

GPUS=8
resolution=512

# generate images
if [ "$use_think" = True ]; then
        output_path=./output/baseline/
        image_dir=$output_path/images

        torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=$GPUS \
        --master_addr=127.0.0.1 \
        --master_port=12345 \
        ./eval/gen/gen_images_mp_wise.py \
        --output_dir $output_path/images \
        --metadata_file ./eval/gen/wise/final_data.json \
        --resolution resolution \
        --max_latent_size 64 \
        --model-path $model_path \
        --think
else
        output_path=./output/baseline/
        image_dir=$output_path/images

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

# calculate score
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
