# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Quick test for UnifiedReward sglang server.

Usage:
  python scripts/test_unifiedreward.py [image_path] [prompt]
  python scripts/test_unifiedreward.py                          # uses a generated test image
  python scripts/test_unifiedreward.py my_image.png "a cat"
"""

import argparse
import base64
import re
import time
from io import BytesIO

from openai import OpenAI
from PIL import Image


def make_test_image():
    """Generate a simple gradient test image."""
    img = Image.new("RGB", (512, 512))
    pixels = img.load()
    for y in range(512):
        for x in range(512):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    return img


def pil_to_base64(image):
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return f"data:image;base64,{base64.b64encode(buf.getvalue()).decode()}"


def score(client, model, image, prompt):
    question = (
        f"<image>\nYou are given a text caption and a generated image based on that caption. "
        f"Your task is to evaluate this image based on two key criteria:\n"
        f"1. Alignment with the Caption: Assess how well this image aligns with the provided caption. "
        f"Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n"
        f"2. Overall Image Quality: Examine the visual quality of this image, including clarity, "
        f"detail preservation, color accuracy, and overall aesthetic appeal.\n"
        f"Based on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\n"
        f"Your task is provided as follows:\nText Caption: [{prompt}]"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": pil_to_base64(image)}},
                {"type": "text", "text": question},
            ],
        }],
        temperature=0,
        max_tokens=512,
    )
    text = response.choices[0].message.content
    match = re.search(r"Final Score:\s*([1-5](?:\.\d+)?)", text)
    return float(match.group(1)) if match else 0.0, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=None)
    parser.add_argument("prompt", nargs="?", default="a beautiful sunset over the ocean")
    parser.add_argument("--url", default="http://127.0.0.1:17140/v1")
    parser.add_argument("--model", default="UnifiedReward-7b-v1.5")
    args = parser.parse_args()

    if args.image:
        image = Image.open(args.image).convert("RGB").resize((512, 512))
    else:
        print("No image provided, using generated test image.")
        image = make_test_image()

    client = OpenAI(base_url=args.url, api_key="flowgrpo")

    print(f"Server:  {args.url}")
    print(f"Model:   {args.model}")
    print(f"Prompt:  {args.prompt}")
    print("-" * 60)

    t0 = time.time()
    final_score, raw_output = score(client, args.model, image, args.prompt)
    elapsed = time.time() - t0

    print(f"Raw output:\n{raw_output}")
    print("-" * 60)
    print(f"Final Score: {final_score}")
    print(f"Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()