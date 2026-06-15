# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -e

model_path=ByteDance-Seed/BAGEL-7B-MoT

export API_URL=  # set up the OpenAI-compatible API endpoint url
openai_api_key=""  # set up your OpenAI API key

output_path=./output/baseline/
mnt_output_path=./output

export OPENAI_API_KEY=$openai_api_key

GPUS=8
pip install xlsxwriter

# generate images
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_mp_rise.py \
    --output_dir $output_path/bagel \
    --metadata_file ./data/.json \
    --max_latent_size 64 \
    --model-path $model_path \
    --think --image-path ./data/


# calculate score
python ./eval/gen/rise/gpt_eval.py \
    --data ./data/.json \
    --input ./data/ \
    --output $output_path/bagel


mv $output_path $mnt_output_path

