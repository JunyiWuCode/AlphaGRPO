# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -e

GPUS=8
model_path=ByteDance-Seed/BAGEL-7B-MoT

jobname=baseline
resolution=1024

output_root=./output/${jobname}
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
    ./eval/gen/gen_images_mp_dpg.py \
    --output_dir $output_path \
    --metadata_file ./eval/gen/dpg/metadata.json \
    --batch_size 1 \
    --resolution $resolution \
    --max_latent_size 64 \
    --model-path $model_path

pip install uv
export UV_LINK_MODE=copy
pwd_bench="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd ./
uv venv --python 3.9
source .venv/bin/activate
uv pip install -r ${pwd_bench}/eval/gen/dpg/requirements-for-dpg_bench.txt
uv pip install addict simplejson sortedcontainers datasets==2.21.0 oss2
cd $pwd_bench
echo $pwd_bench

python -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('damo/mplug_visual-question-answering_coco_large_en')"


IMAGE_DIR=$output_path/images/
RES_PATH=$output_path/dpg-bench_results.txt

# IMAGE_DIR=$mnt_output_path/images/
# RES_PATH=$mnt_output_path/dpg-bench_results.txt

echo $IMAGE_DIR
# calculate score
accelerate launch --num_machines 1 --num_processes $GPUS --multi_gpu --mixed_precision "fp16" --main_process_port 53343 \
  ./eval/gen/dpg/compute_dpg_bench.py \
  --image-root-path $IMAGE_DIR \
  --resolution $resolution \
  --res-path $RES_PATH \
  --vqa-model mplug

echo $output_path move to $mnt_output_path
mv $output_path $mnt_output_path






