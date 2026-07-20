#!/usr/bin/env python3
"""Convert GenEval JSONL output into a machine-readable summary."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize_geneval(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"No GenEval rows found in {path}")

    by_task: dict[str, list[bool]] = defaultdict(list)
    prompts: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_task[row["tag"]].append(bool(row["correct"]))
        prompts[row["metadata"]].append(bool(row["correct"]))

    task_scores = {
        task: {
            "score": 100.0 * sum(values) / len(values),
            "correct": sum(values),
            "count": len(values),
        }
        for task, values in by_task.items()
    }
    score = sum(item["score"] for item in task_scores.values()) / len(task_scores)
    return {
        "method": "geneval_mask2former_clip",
        "score": score,
        "image_accuracy": 100.0 * sum(bool(row["correct"]) for row in rows) / len(rows),
        "prompt_accuracy_at_4": 100.0 * sum(any(values) for values in prompts.values()) / len(prompts),
        "image_count": len(rows),
        "prompt_count": len(prompts),
        "task_scores": task_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--expected-prompts", type=int)
    args = parser.parse_args()

    summary = summarize_geneval(args.result)
    for key, expected in (
        ("image_count", args.expected_images),
        ("prompt_count", args.expected_prompts),
    ):
        if expected is not None and summary[key] != expected:
            raise ValueError(f"Expected {expected} for {key}, found {summary[key]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
