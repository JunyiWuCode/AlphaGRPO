# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -e

GPUS=8
model_path=ByteDance-Seed/BAGEL-7B-MoT

export OPENAI_API_URL=  # set up the OpenAI-compatible API endpoint url
export OPENAI_API_KEY=""  # set up your OpenAI API key

jobname=baseline
resolution=1024

output_root=./output/${jobname}
checkpoint_dir=./output${jobname}
echo $(dirname $checkpoint_dir)
# rm -rf $output_root
output_path=$output_root

mnt_output_path=./output
if [ "$resolution" = "512" ]; then
    mnt_output_path=${mnt_output_path}_512px
fi

# generate images
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_mp_tiif.py \
    --output_dir $output_path/images/ \
    --metadata_file_root ./eval/gen/tiif/testmini_prompts/ \
    --batch_size 1 \
    --resolution $resolution \
    --max_latent_size 64 \
    --model-path $model_path

JSONL_DIR=./eval/gen/tiif/testmini_eval_prompts/
IMAGE_DIR=$output_path/images/
IMAGE_DIR=$mnt_output_path/images/
MODEL_NAME=bagel
OUTPUT_DIR=$output_path/eval_results_1024_4ov2
API_KEY=${OPENAI_API_KEY}
BASE_URL=${OPENAI_API_URL}
MODEL="gpt-4o-2024-08-06"
# MODEL="gpt-4.1-2025-04-14"

echo $IMAGE_DIR
# calculate score
python ./eval/gen/tiif/eval_with_vlm.py \
    --jsonl_dir $JSONL_DIR \
    --image_dir $IMAGE_DIR \
    --eval_model $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --api_key $API_KEY \
    --base_url $BASE_URL \
    --model "$MODEL"

# summarize score
python ./eval/gen/tiif/summary_results.py --input_dir $OUTPUT_DIR
python ./eval/gen/tiif/summary_dimension_results.py --input_excel $OUTPUT_DIR/result_summary.xlsx --output_txt $OUTPUT_DIR/result_summary_dimension.txt

# mv $output_path $mnt_output_path
