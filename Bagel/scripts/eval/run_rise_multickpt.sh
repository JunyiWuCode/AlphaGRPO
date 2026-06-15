

# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -e

model_path=ByteDance-Seed/BAGEL-7B-MoT

export API_URL=  # set up the OpenAI-compatible API endpoint url
openai_api_key=""  # set up your OpenAI API key

export OPENAI_API_KEY=$openai_api_key

GPUS=8

# Download RISEBench from HuggingFace
bench_dir=./eval/gen/rise/RISEBench
if [ ! -d "$bench_dir" ]; then
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='PhoenixZ/RISEBench', repo_type='dataset', local_dir='$bench_dir')"
fi
jobname=''
checkpoint_list=(379)

output_root=./output/${jobname}
checkpoint_dir=./output${jobname}


pip install xlsxwriter

echo $(dirname $checkpoint_dir)

for checkpoint_num in "${checkpoint_list[@]}"
do
    output_path=${output_root}/checkpoint-${checkpoint_num}
    mnt_output_path=${checkpoint_dir}/eval_tiif_${resolution}_output

    if [ ! -d $mnt_output_path ]; then
        mkdir -p $mnt_output_path
    fi

    export BAGEL_LORA_PATH=${checkpoint_dir}/checkpoint-${checkpoint_num}/hf_model/

    # generate images
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=$GPUS \
        --master_addr=127.0.0.1 \
        --master_port=12345 \
        ./eval/gen/gen_images_mp_rise.py \
        --output_dir $output_path/bagel \
        --metadata_file $bench_dir/datav2_total_w_subtask.json \
        --max_latent_size 64 \
        --model-path $model_path \
        --think --image-path $bench_dir/data


    # calculate score
    python ./eval/gen/rise/gpt_eval.py \
        --data $bench_dir/datav2_total_w_subtask.json \
        --input $bench_dir/data \
        --output $output_path/bagel

    echo "Moving ${output_path} to ${mnt_output_path}"
    mv $output_path $mnt_output_path

done
