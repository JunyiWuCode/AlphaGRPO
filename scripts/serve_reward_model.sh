export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

model=Qwen/Qwen3-VL-30B-A3B-Instruct
export OPENAI_API_URL=http://[$1]:$2/v1
echo reward model address: $OPENAI_API_URL

DP_NUM=1
TP_NUM=1
echo DP_NUM $DP_NUM TP_NUM $TP_NUM

python -m sglang.launch_server \
    --model-path ${model} \
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
