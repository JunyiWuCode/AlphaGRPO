
set -e

# torch 2.8
# Install vLLM >=0.11.0
pip install -U vllm==0.11.0 sympy httpx==0.23.3  --user
pip install flashinfer-python ml_dtypes -U --user

# torch 2.9
# pip install uv
# uv venv vllm-env
# source vllm-env/bin/activate
# uv pip install -U vllm "triton-kernels @ git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels" --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

pip install -U flash-attn==2.8.2 --no-build-isolation --use-pep517 --user

# Install Qwen-VL utility library (recommended for offline inference)
pip install qwen-vl-utils==0.0.14

pip install "numpy<2.0"


# model=Qwen3-235B-A22B-Instruct-2507
model=Qwen3-VL-235B-A22B-Instruct
# model=Qwen3-VL-30B-A3B-Instruct

# Public serving variables:
# - SERVE_HOST: host/IP to bind. Falls back to MY_HOST_IPV6, then 0.0.0.0.
# - SERVE_PORT: port to bind. Falls back to PORT3, then 8000.
serve_host=${SERVE_HOST:-${MY_HOST_IPV6:-0.0.0.0}}
serve_port=${SERVE_PORT:-${PORT3:-8000}}

export OPENAI_API_URL=http://[${serve_host}]:${serve_port}/v1/chat/completions
export OPENAI_API_URL=http://[${serve_host}]:${serve_port}/v1
echo $OPENAI_API_URL

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export NCCL_P2P_DISABLE=1

unset HTTP_PROXY
unset http_proxy
unset https_proxy
unset no_proxy

 vllm serve /path/to/models/${model} \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --limit-mm-per-prompt.video 0 \
  --enable-expert-parallel \
  --mm-encoder-tp-mode data \
  --disable-custom-all-reduce \
  --max-model-len 32768 \
  --max_num_batched_tokens 32768 \
  --max-num-seqs 4096 \
  --async-scheduling \
  --gpu-memory-utilization  0.95 \
  --dtype bfloat16 \
  --host $serve_host \
  --port $serve_port



# Call example.
# import time
# from openai import OpenAI

# client = OpenAI(
#     api_key="EMPTY",
#     base_url="http://localhost:8000/v1",
#     timeout=3600
# )

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
#                 }
#             },
#             {
#                 "type": "text",
#                 "text": "Read all the text in the image."
#             }
#         ]
#     }
# ]

# start = time.time()
# response = client.chat.completions.create(
#     model="Qwen/Qwen3-VL-235B-A22B-Instruct",
#     messages=messages,
#     max_tokens=2048
# )
# print(f"Response costs: {time.time() - start:.2f}s")
# print(f"Generated text: {response.choices[0].message.content}")