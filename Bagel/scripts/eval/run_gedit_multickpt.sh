# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# run this script at the root of the project folder
pip install httpx==0.23.0
pip install openai==1.87.0
pip install datasets
pip install megfile


N_GPU=8  # Number of GPU used in for the evaluation
MODEL_PATH=ByteDance-Seed/BAGEL-7B-MoT

AZURE_ENDPOINT=  # set up the azure openai endpoint url
AZURE_OPENAI_KEY=""  # set up the azure openai key

N_GPT_PARALLEL=5


mkdir -p "$OUTPUT_DIR"
mkdir -p "$GEN_DIR"
mkdir -p "$LOG_DIR"


# # ----------------------------
# #    Download GEdit Dataset
# # ----------------------------
python -c "from datasets import load_dataset; dataset = load_dataset('stepfun-ai/GEdit-Bench')"
echo "Dataset Downloaded"

jobname=''
checkpoint_list=(**)
resolution=512


output_root=./output/${jobname}
checkpoint_dir=./output${jobname}
echo $(dirname $checkpoint_dir)

for checkpoint_num in "${checkpoint_list[@]}"
do
    task='gedit'

    output_path=${output_root}/checkpoint-${checkpoint_num}
    if [ "$use_think" = True ]; then
        mnt_output_path=${checkpoint_dir}/eval_${task}_think_output
    else
        mnt_output_path=${checkpoint_dir}/eval_${task}_output
    fi

    if [ ! -d $mnt_output_path ]; then
        mkdir -p $mnt_output_path
    fi

    export BAGEL_LORA_PATH=${checkpoint_dir}/checkpoint-${checkpoint_num}/hf_model/

    OUTPUT_DIR=$output_path
    GEN_DIR="$OUTPUT_DIR/gen_image"
    LOG_DIR="$OUTPUT_DIR/logs"

    # # ---------------------
    # #    Generate Images
    # # ---------------------
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=8 \
        --master_addr=127.0.0.1 \
        --master_port=12345 \
        eval/gen/gedit/gen_images_gedit.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$GEN_DIR"

    echo "Image Generation Done"


    # # ---------------------
    # #    GPT Evaluation
    # # ---------------------
    cd eval/gen/gedit
    python test_gedit_score.py --save_path "$OUTPUT_DIR" --azure_endpoint "$AZURE_ENDPOINT" --gpt_keys "$AZURE_OPENAI_KEY"  --max_workers "$N_GPT_PARALLEL"
    echo "Evaluation Done"


    # # --------------------
    # #    Print Results
    # # --------------------
    python calculate_statistics.py --save_path "$OUTPUT_DIR"  --language en

    echo "Moving ${output_path} to ${mnt_output_path}"
    mv "${output_path}" "${mnt_output_path}"
done