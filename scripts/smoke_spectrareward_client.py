#!/usr/bin/env python
"""Exercise the SGLang SpectraReward path with one deterministic image."""

import argparse

from PIL import Image

from rewards.sglang_spectrareward import SpectraRewardSGLangClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--prompt",
        default="A plain white square centered on a black background.",
    )
    args = parser.parse_args()

    image = Image.new("RGB", (512, 512), color="black")
    for x in range(160, 352):
        for y in range(160, 352):
            image.putpixel((x, y), (255, 255, 255))

    scorer = SpectraRewardSGLangClient(args.model, args.url, max_concurrent=1)
    score = scorer([image], [args.prompt])[0]
    print(f"spectrareward={score:.8f}")


if __name__ == "__main__":
    main()
