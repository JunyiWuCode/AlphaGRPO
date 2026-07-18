#!/usr/bin/env python3
"""Distributed implementation of the bundled GenEval2 soft-TIFA scorer."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


YES_ANSWERS = ("Yes", "yes", " yes", " Yes")
NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def answer_candidates(question: str, answer: str) -> tuple[str, ...]:
    if not question.startswith("How many"):
        return YES_ANSWERS
    numeric = NUMBER_WORDS.get(answer, "other")
    return (
        answer,
        answer.capitalize(),
        f" {answer}",
        f" {answer.capitalize()}",
        numeric,
        f" {numeric}",
    )


def geometric_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute a geometric mean of an empty list")
    if any(value <= 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-data", type=Path, required=True)
    parser.add_argument("--image-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    return parser.parse_args()


def main() -> None:
    import torch
    import torch.distributed as dist
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    rows = read_jsonl(args.benchmark_data)
    image_map = json.loads(args.image_map.read_text(encoding="utf-8"))
    missing = [row["prompt"] for row in rows if row["prompt"] not in image_map]
    if missing:
        raise ValueError(f"Image map is missing {len(missing)} prompts")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    token_id_cache: dict[tuple[str, ...], list[int]] = {}
    local_scores: dict[int, list[float]] = {}
    local_indices = list(range(rank, len(rows), world_size))
    for completed, index in enumerate(local_indices, start=1):
        row = rows[index]
        score_list: list[float] = []
        for question, answer in row["vqa_list"]:
            candidates = answer_candidates(question, answer)
            if candidates not in token_id_cache:
                token_id_cache[candidates] = [
                    processor.tokenizer.encode(candidate)[0] for candidate in candidates
                ]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_map[row["prompt"]]},
                        {"type": "text", "text": f"{question} Answer in one word."},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            probabilities = torch.softmax(outputs.scores[0].float(), dim=-1)
            answer_probability = sum(
                probabilities[0, token_id].item()
                for token_id in token_id_cache[candidates]
            )
            score_list.append(answer_probability)
        local_scores[index] = score_list
        if completed % 10 == 0 or completed == len(local_indices):
            print(f"rank={rank} completed={completed}/{len(local_indices)}", flush=True)

    if world_size > 1:
        gathered: list[dict[int, list[float]] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_scores)
    else:
        gathered = [local_scores]

    if rank == 0:
        merged: dict[int, list[float]] = {}
        for shard in gathered:
            if shard is not None:
                merged.update(shard)
        if len(merged) != len(rows):
            raise RuntimeError(f"Expected {len(rows)} scores, received {len(merged)}")
        score_lists = [merged[index] for index in range(len(rows))]
        per_prompt_scores = [geometric_mean(scores) for scores in score_lists]
        total_score = 100.0 * sum(per_prompt_scores) / len(per_prompt_scores)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "score_lists.json").write_text(
            json.dumps(score_lists) + "\n", encoding="utf-8"
        )
        summary = {
            "method": "soft_tifa_gm",
            "model_id": args.model_id,
            "score": total_score,
            "prompt_count": len(rows),
            "question_count": sum(len(scores) for scores in score_lists),
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary), flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
