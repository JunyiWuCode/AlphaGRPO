# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Generic MLLM reward server.

Hosts one or more reward scorers (the SpectraReward scorer:
mean image-conditioned prompt log-likelihood) and scores (image, prompt)
pairs over HTTP, exactly as the in-process reward does.

Protocol (pickle over HTTP):
  Request:  pickle.dumps({"images": [jpeg_bytes, ...], "prompts": [str, ...]})
  Response: pickle.dumps({"outputs": [float, ...]})

Single GPU:
  python scripts/mllm_server.py --model-id Qwen/Qwen3-VL-30B-A3B-Instruct --port 18090

Data-parallel (single port, N GPUs internally):
  python scripts/mllm_server.py --model-id Qwen/Qwen3-VL-30B-A3B-Instruct \
      --port 18090 --num-gpus 8

In DP mode the server loads N scorers pinned to cuda:START..START+N-1 and
dispatches each request to whichever scorer is free (blocking queue), so the
client only ever sees a single URL.
"""

import argparse
import os
import pickle
import queue
import sys
import threading
import traceback
from io import BytesIO

import torch
from flask import Flask, Response, request
from PIL import Image

# Make `from rewards.spectrareward import SpectraRewardScorer` importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "alpha_grpo"))

app = Flask(__name__)

# Pool of (scorer, gpu_index). Requests block on .get() when all scorers are
# busy; a worker returns its scorer when done.
_scorer_pool: "queue.Queue" = queue.Queue()
_pool_total = 0


@app.route("/", methods=["POST"])
def score():
    try:
        data = pickle.loads(request.data)
        images = [Image.open(BytesIO(b)).convert("RGB") for b in data["images"]]
        prompts = list(data["prompts"])

        scorer, gpu_idx = _scorer_pool.get()  # blocks until a scorer is free
        try:
            rewards = scorer(images, prompts)  # Tensor[B]
        finally:
            _scorer_pool.put((scorer, gpu_idx))

        return Response(
            pickle.dumps({"outputs": rewards.tolist()}),
            mimetype="application/octet-stream",
        )
    except Exception:
        traceback.print_exc()
        return Response(
            pickle.dumps({"error": traceback.format_exc()}),
            status=500,
            mimetype="application/octet-stream",
        )


@app.route("/health", methods=["GET"])
def health():
    return f"ok; {_scorer_pool.qsize()}/{_pool_total} scorers idle"


def _build_scorer(model_id, gpu_idx, dtype, user_instruction,
                  prompt_prefix, prompt_suffix, exclude_eos):
    from rewards.spectrareward import SpectraRewardScorer
    return SpectraRewardScorer(
        model_id=model_id,
        device=f"cuda:{gpu_idx}",
        dtype=dtype,
        user_instruction=user_instruction,
        prompt_prefix=prompt_prefix,
        prompt_suffix=prompt_suffix,
        exclude_eos=exclude_eos,
        lazy_gpu=False,  # server keeps the reward model resident on GPU
    )


def main():
    global _pool_total
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=18090)
    p.add_argument("--num-gpus", type=int, default=1,
                   help="Load N scorers on cuda:START..START+N-1 and dispatch across them.")
    p.add_argument("--start-gpu", type=int, default=0,
                   help="First GPU index (default 0).")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--user-instruction", default="")
    p.add_argument("--prompt-prefix", default="")
    p.add_argument("--prompt-suffix", default="")
    p.add_argument("--no-exclude-eos", action="store_true")
    args = p.parse_args()

    if args.num_gpus < 1:
        raise ValueError(f"--num-gpus must be >= 1 (got {args.num_gpus})")

    dtype = getattr(torch, args.dtype)
    exclude_eos = not args.no_exclude_eos

    print(f"[mllm-server] Loading {args.num_gpus} scorer(s) of {args.model_id} "
          f"on cuda:{args.start_gpu}..cuda:{args.start_gpu + args.num_gpus - 1} ...")

    scorers = [None] * args.num_gpus
    errors = [None] * args.num_gpus

    def _load(i):
        gpu = args.start_gpu + i
        try:
            scorers[i] = _build_scorer(
                args.model_id, gpu, dtype,
                args.user_instruction, args.prompt_prefix, args.prompt_suffix,
                exclude_eos,
            )
            print(f"[mllm-server]   scorer #{i} ready on cuda:{gpu}")
        except Exception as e:
            errors[i] = e
            traceback.print_exc()

    threads = [threading.Thread(target=_load, args=(i,), daemon=True)
               for i in range(args.num_gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if any(errors):
        raise RuntimeError(f"Failed to load one or more scorers: {errors}")

    for i, s in enumerate(scorers):
        _scorer_pool.put((s, args.start_gpu + i))
    _pool_total = args.num_gpus

    print(f"[mllm-server] Ready — listening on {args.host}:{args.port} (DP={args.num_gpus})")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
