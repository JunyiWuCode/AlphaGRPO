#!/usr/bin/env python3
"""Summarize TIIF short/long accuracy without the optional Excel dependency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GROUPS = {
    "basic_attribute": ["shape+color", "color+texture", "texture+color", "shape+texture"],
    "basic_relation": ["2d_spatial_relation", "3d_spatial_relation", "action+2d", "action+3d"],
    "basic_reasoning": ["numeracy", "negation", "differentiation", "comparison"],
    "advanced_attribute_relation": [
        "action+color", "action+texture", "color+2d", "color+3d", "shape+2d",
        "shape+3d", "texture+2d", "texture+3d",
    ],
    "advanced_attribute_reasoning": [
        "numeracy+color", "numeracy+texture", "comparison+color", "comparison+texture",
        "differentiation+color", "differentiation+texture", "negation+color", "negation+texture",
    ],
    "advanced_relation_reasoning": [
        "numeracy+2d", "numeracy+3d", "comparison+2d", "comparison+3d",
        "differentiation+2d", "differentiation+3d", "negation+2d", "negation+3d",
    ],
    "text_generation": ["text"],
    "style_control": ["style"],
    "real_complex": ["real_world"],
}


def summarize_tiif(root: Path) -> dict:
    model_dirs = [path for path in root.iterdir() if path.is_dir()]
    if len(model_dirs) != 1:
        raise ValueError(f"Expected one eval_model directory under {root}, found {model_dirs}")

    category_scores: dict[str, dict[str, dict]] = {"short": {}, "long": {}}
    file_counts = {"short": 0, "long": 0}
    question_counts = {"short": 0, "long": 0}
    for attribute_dir in model_dirs[0].iterdir():
        if not attribute_dir.is_dir():
            continue
        for length in ("short", "long"):
            result_dir = attribute_dir / length
            files = sorted(result_dir.glob("*.json")) if result_dir.is_dir() else []
            correct = 0
            total = 0
            for path in files:
                row = json.loads(path.read_text(encoding="utf-8"))
                answers = list(zip(row.get("gt_answers", []), row.get("model_pred", [])))
                total += len(answers)
                correct += sum(str(gt).strip().lower() == str(pred).strip().lower() for gt, pred in answers)
            if files and not total:
                raise ValueError(f"No scored questions found in {result_dir}")
            if files:
                category_scores[length][attribute_dir.name] = {
                    "score": 100.0 * correct / total,
                    "correct": correct,
                    "questions": total,
                    "files": len(files),
                }
                file_counts[length] += len(files)
                question_counts[length] += total

    group_scores: dict[str, dict[str, float]] = {"short": {}, "long": {}}
    scores = {}
    for length in ("short", "long"):
        missing = sorted({item for items in GROUPS.values() for item in items} - category_scores[length].keys())
        if missing:
            raise ValueError(f"Missing {length} TIIF categories: {missing}")
        for group, attributes in GROUPS.items():
            group_scores[length][group] = sum(
                category_scores[length][attribute]["score"] for attribute in attributes
            ) / len(attributes)
        scores[length] = sum(group_scores[length].values()) / len(group_scores[length])

    return {
        "method": "tiif_vlm_yes_no",
        "score_short": scores["short"],
        "score_long": scores["long"],
        "file_counts": file_counts,
        "question_counts": question_counts,
        "group_scores": group_scores,
        "category_scores": category_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--expected-files-per-length", type=int)
    parser.add_argument("--expected-questions-per-length", type=int)
    args = parser.parse_args()

    summary = summarize_tiif(args.result_dir)
    summary["judge_model"] = args.judge_model
    for length in ("short", "long"):
        if args.expected_files_per_length is not None and summary["file_counts"][length] != args.expected_files_per_length:
            raise ValueError(
                f"Expected {args.expected_files_per_length} {length} files, "
                f"found {summary['file_counts'][length]}"
            )
        if args.expected_questions_per_length is not None and summary["question_counts"][length] != args.expected_questions_per_length:
            raise ValueError(
                f"Expected {args.expected_questions_per_length} {length} questions, "
                f"found {summary['question_counts'][length]}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("judge_model", "score_short", "score_long")}), flush=True)


if __name__ == "__main__":
    main()
