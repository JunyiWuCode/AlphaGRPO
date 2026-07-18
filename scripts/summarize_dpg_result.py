#!/usr/bin/env python3
"""Convert the bundled DPG-Bench text output into a machine-readable summary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SCORE_RE = re.compile(r"^DPG-Bench score:\s*([0-9.]+)\s*$")
CATEGORY_RE = re.compile(r"^\s*([^:]+):\s*([0-9.]+)(?:\s*\(n=(\d+)\))?\s*$")


def parse_dpg_result(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        summary_start = next(i for i, line in enumerate(lines) if line.startswith("Model:"))
    except StopIteration as error:
        raise ValueError(f"Missing DPG summary in {path}") from error

    image_count = sum(bool(line.strip()) for line in lines[:summary_start])
    summary: dict = {
        "method": "dpg_bench_mplug",
        "model": lines[summary_start].split(":", 1)[1].strip(),
        "image_count": image_count,
        "l1_category_scores": {},
        "l2_category_scores": {},
    }
    category_level: str | None = None
    for line in lines[summary_start + 1 :]:
        stripped = line.strip()
        if stripped == "L1 category scores:":
            category_level = "l1_category_scores"
            continue
        if stripped == "L2 category scores:":
            category_level = "l2_category_scores"
            continue
        score_match = SCORE_RE.match(stripped)
        if score_match:
            summary["score"] = float(score_match.group(1))
            category_level = None
            continue
        if stripped.startswith(("Image path:", "Save results to:")):
            category_level = None
            continue
        category_match = CATEGORY_RE.match(line)
        if category_level and category_match:
            entry = {"score": float(category_match.group(2))}
            if category_match.group(3) is not None:
                entry["count"] = int(category_match.group(3))
            summary[category_level][category_match.group(1).strip()] = entry

    if "score" not in summary:
        raise ValueError(f"Missing DPG-Bench score in {path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-images", type=int)
    args = parser.parse_args()

    summary = parse_dpg_result(args.result)
    if args.expected_images is not None and summary["image_count"] != args.expected_images:
        raise ValueError(
            f"Expected {args.expected_images} images, found {summary['image_count']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
