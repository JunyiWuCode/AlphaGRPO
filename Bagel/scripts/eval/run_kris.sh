# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -e


model_path=ByteDance-Seed/BAGEL-7B-MoT

export API_URL=  # set up the OpenAI-compatible API endpoint url
openai_api_key=""  # set up your OpenAI API key
# Download KRIS_Bench from HuggingFace if not present
bench_dir=./eval/gen/kris/KRIS_Bench
if [ ! -d "$bench_dir" ]; then
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Liang0223/KRIS_Bench', repo_type='dataset', local_dir='$bench_dir')"
fi
image_path=$bench_dir

output_path=./output/baseline/
mnt_output_path=./output


export OPENAI_API_KEY=$openai_api_key
GPUS=8


# # generate images
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_mp_kris.py \
    --output_dir $output_path/bagel \
    --metadata_file ./eval/gen/kris/final_data.json \
    --max_latent_size 64 \
    --model-path $model_path \
    --image-path $image_path \
    --think

# calculate score
python ./eval/gen/kris/metrics_common.py \
    --results_dir $output_path \
    --bench_dir $bench_dir \
    --max_workers 4

python ./eval/gen/kris/metrics_knowledge.py \
    --results_dir $output_path \
    --bench_dir $bench_dir \
    --max_workers 4

python ./eval/gen/kris/metrics_multi_element.py \
    --results_dir $output_path \
    --bench_dir $bench_dir \
    --max_workers 4

python ./eval/gen/kris/metrics_temporal_prediction.py \
    --results_dir $output_path \
    --bench_dir $bench_dir \
    --max_workers 4

python ./eval/gen/kris/metrics_view_change.py \
    --results_dir $output_path \
    --bench_dir $bench_dir \
    --max_workers 4


# summarize score
python ./eval/gen/kris/summarize.py \
    --results_dir $output_path/bagel \


mv $output_path $mnt_output_path
