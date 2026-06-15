# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Quick test for HPSv3 reward server.

Usage:
  python scripts/test_hpsv3.py [image_path] [prompt]
  python scripts/test_hpsv3.py                          # uses a generated test image
  python scripts/test_hpsv3.py my_image.png "a cat"
"""

import argparse
import pickle
import time
from io import BytesIO

from PIL import Image


def make_test_image():
    img = Image.new("RGB", (512, 512))
    pixels = img.load()
    for y in range(512):
        for x in range(512):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=None)
    parser.add_argument("prompt", nargs="?", default="a beautiful sunset over the ocean")
    parser.add_argument("--url", default="http://127.0.0.1:18087")
    args = parser.parse_args()

    if args.image:
        image = Image.open(args.image).convert("RGB")
    else:
        print("No image provided, using generated test image.")
        image = make_test_image()

    buf = BytesIO()
    image.save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()

    data = pickle.dumps({
        "images": [jpeg_bytes],
        "prompts": [args.prompt],
    })

    import requests
    print(f"Server: {args.url}")
    print(f"Prompt: {args.prompt}")
    print("-" * 60)

    t0 = time.time()
    resp = requests.post(args.url, data=data, timeout=120)
    elapsed = time.time() - t0

    result = pickle.loads(resp.content)
    print(f"Score (mu): {result['outputs']}")
    print(f"Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()