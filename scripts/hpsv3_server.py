# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""HPSv3 reward model server.

Receives pickle-serialized requests with JPEG images and prompts,
returns pickle-serialized scores.

Protocol (matches deqa/geneval pattern):
  Request:  pickle.dumps({"images": [jpeg_bytes, ...], "prompts": [str, ...]})
  Response: pickle.dumps({"outputs": [float, ...]})

Usage:
  python hpsv3_server.py --host 0.0.0.0 --port 18087 --device cuda
  # Or launch via: bash serve_hpsv3.sh [host] [port] [device] [checkpoint]
"""

import argparse
import pickle
import sys
from io import BytesIO

import torch
from flask import Flask, Response, request
from PIL import Image

app = Flask(__name__)
inferencer = None


@app.route("/", methods=["POST"])
def score():
    data = pickle.loads(request.data)

    images = [
        Image.open(BytesIO(img_bytes)).convert("RGB")
        for img_bytes in data["images"]
    ]
    prompts = list(data["prompts"])

    rewards = inferencer.reward(images, prompts)
    scores = [r[0].item() for r in rewards]  # mu

    return Response(
        pickle.dumps({"outputs": scores}),
        mimetype="application/octet-stream",
    )


@app.route("/health", methods=["GET"])
def health():
    return "ok"


def main():
    global inferencer

    parser = argparse.ArgumentParser(description="HPSv3 reward server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18087)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to HPSv3.safetensors (auto-downloads from HF if omitted)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from hpsv3 import HPSv3RewardInferencer

    print(f"[hpsv3] Loading model on {args.device} ...")
    inferencer = HPSv3RewardInferencer(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    print(f"[hpsv3] Ready — listening on {args.host}:{args.port}")

    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()