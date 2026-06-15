
pip install -U torch==2.8.0 "sglang[all]==0.5.5.post3" sympy httpx==0.23.3 ml_dtypes flashinfer-python==0.5.2 -U --user  # "protobuf==3.20.3" 

pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
# pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.2+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"

# pip install -U flash-attn --no-build-isolation --use-pep517 --user
# Install Qwen-VL utility library (recommended for offline inference)
pip install qwen-vl-utils==0.0.14 transformers==4.57.1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

model=Qwen3-VL-30B-A3B-Instruct
export OPENAI_API_URL=http://[$1]:$2/v1
echo $OPENAI_API_URL

DP_NUM=1
TP_NUM=1
echo DP_NUM $DP_NUM TP_NUM $TP_NUM

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY


# for i in {0..11}; do  
    # echo start sglang serve $i.

    python -m sglang.launch_server \
    --model-path /path/to/models/${model} \
    --tp $TP_NUM --dp $DP_NUM \
    --host $1 \
    --port $2 \
    --trust-remote-code \
    --mem-fraction-static 0.9 \
    --enable-mixed-chunk \
    --log-requests-level 0 \
    --chunked-prefill-size 2048 \
    --max-prefill-tokens 32768 \
    --enable-torch-compile \
    --context-length 4096 \
    --max-running-requests 4096 \
    --schedule-policy lpm \
    --moe-runner-backend triton \
    --enable-multimodal \
    --max-queued-requests 20000 \
    --watchdog-timeout 1000000 \
    --kv-cache-dtype auto 

    # echo abort sglang serve $i.
# done

# --enable-expert-parallel \    


# --disable-custom-all-reduce

