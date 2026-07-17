#!/usr/bin/env python3
"""Generate SD3.5 images in the layouts expected by AlphaGRPO benchmarks."""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_BENCHMARKS = ("geneval", "tiif", "dpg", "geneval2", "wise")


@dataclass(frozen=True)
class GenerationTask:
    benchmark: str
    prompt: str
    output_path: str
    seed: int


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _geneval_tasks(data_root: Path, output_root: Path, seed: int) -> list[GenerationTask]:
    rows = _read_jsonl(data_root / "geneval/prompts/evaluation_metadata_long.jsonl")
    tasks: list[GenerationTask] = []
    for prompt_index, row in enumerate(rows):
        prompt_dir = output_root / "geneval/images" / f"{prompt_index:05d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        sample_dir = prompt_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for sample_index in range(4):
            tasks.append(
                GenerationTask(
                    benchmark="geneval",
                    prompt=row["prompt"],
                    output_path=str(sample_dir / f"{sample_index:05d}.png"),
                    seed=seed + prompt_index * 4 + sample_index,
                )
            )
    return tasks


def _tiif_tasks(data_root: Path, output_root: Path, seed: int) -> list[GenerationTask]:
    # Keep glob ordering consistent with the bundled TIIF evaluator.
    files = glob.glob(str(data_root / "tiif/testmini_prompts/*.jsonl"))
    rows: list[dict] = []
    for path in files:
        rows.extend(_read_jsonl(Path(path)))

    tasks: list[GenerationTask] = []
    for index, row in enumerate(rows):
        for description_offset, description in enumerate(("short_description", "long_description")):
            output_path = (
                output_root
                / "tiif/images"
                / row["type"]
                / "sd35"
                / description
                / f"{index}.png"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append(
                GenerationTask(
                    benchmark="tiif",
                    prompt=row[description],
                    output_path=str(output_path),
                    seed=seed + index * 2 + description_offset,
                )
            )
    return tasks


def _dpg_tasks(data_root: Path, output_root: Path, seed: int) -> list[GenerationTask]:
    rows = json.loads((data_root / "dpg/metadata.json").read_text(encoding="utf-8"))
    image_dir = output_root / "dpg/images"
    image_dir.mkdir(parents=True, exist_ok=True)
    return [
        GenerationTask(
            benchmark="dpg",
            prompt=row["prompt"],
            output_path=str(image_dir / row["filename"].replace(".txt", ".jpg")),
            seed=seed + index,
        )
        for index, row in enumerate(rows)
    ]


def _geneval2_tasks(data_root: Path, output_root: Path, seed: int) -> list[GenerationTask]:
    rows = _read_jsonl(data_root / "geneval2/geneval2_data.jsonl")
    image_dir = output_root / "geneval2/images"
    image_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        row["prompt"]: str((image_dir / f"{index:05d}.png").resolve())
        for index, row in enumerate(rows)
    }
    return [
        GenerationTask(
            benchmark="geneval2",
            prompt=row["prompt"],
            output_path=mapping[row["prompt"]],
            seed=seed + index,
        )
        for index, row in enumerate(rows)
    ]


def _wise_tasks(data_root: Path, output_root: Path, seed: int) -> list[GenerationTask]:
    rows = json.loads((data_root / "wise/final_data.json").read_text(encoding="utf-8"))
    image_dir = output_root / "wise/images"
    image_dir.mkdir(parents=True, exist_ok=True)
    return [
        GenerationTask(
            benchmark="wise",
            prompt=row["Prompt"],
            output_path=str(image_dir / f"{row['prompt_id']}.png"),
            seed=seed + index,
        )
        for index, row in enumerate(rows)
    ]


def build_tasks(
    data_root: Path,
    output_root: Path,
    benchmarks: Iterable[str],
    seed: int,
) -> list[GenerationTask]:
    builders = {
        "geneval": _geneval_tasks,
        "tiif": _tiif_tasks,
        "dpg": _dpg_tasks,
        "geneval2": _geneval2_tasks,
        "wise": _wise_tasks,
    }
    tasks: list[GenerationTask] = []
    seed_offset = 0
    for benchmark in benchmarks:
        benchmark_tasks = builders[benchmark](data_root, output_root, seed + seed_offset)
        tasks.extend(benchmark_tasks)
        seed_offset += len(benchmark_tasks)
    return tasks


def write_benchmark_sidecars(
    data_root: Path, output_root: Path, tasks: Iterable[GenerationTask]
) -> None:
    task_list = list(tasks)
    if any(task.benchmark == "geneval" for task in task_list):
        geneval_rows = _read_jsonl(data_root / "geneval/prompts/evaluation_metadata_long.jsonl")
        for prompt_index, row in enumerate(geneval_rows):
            metadata_path = output_root / "geneval/images" / f"{prompt_index:05d}" / "metadata.jsonl"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    geneval2_mapping = {
        task.prompt: str(Path(task.output_path).resolve())
        for task in task_list
        if task.benchmark == "geneval2"
    }
    if geneval2_mapping:
        mapping_path = output_root / "geneval2/images/geneval_image_map.json"
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_path.write_text(
            json.dumps(geneval2_mapping, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _distributed_context() -> tuple[int, int, int]:
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _load_pipeline(model_id: str, lora_path: str | None, device: Any):
    import torch
    from diffusers import StableDiffusion3Pipeline
    from peft import PeftModel

    pipe = StableDiffusion3Pipeline.from_pretrained(model_id)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    pipe.vae.to(device, dtype=torch.float32)
    pipe.text_encoder.to(device, dtype=torch.bfloat16)
    pipe.text_encoder_2.to(device, dtype=torch.bfloat16)
    pipe.text_encoder_3.to(device, dtype=torch.bfloat16)
    pipe.transformer.to(device)

    if lora_path:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path)
        pipe.transformer.set_adapter("default")
        pipe.transformer.to(device)

    for component in (
        pipe.vae,
        pipe.text_encoder,
        pipe.text_encoder_2,
        pipe.text_encoder_3,
        pipe.transformer,
    ):
        component.eval()
    return pipe


def _batched(values: list[GenerationTask], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def limit_tasks(tasks: list[GenerationTask], max_tasks: int) -> list[GenerationTask]:
    return tasks[:max_tasks] if max_tasks > 0 else tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--benchmarks", default=",".join(SUPPORTED_BENCHMARKS))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    import torch
    import torch.distributed as dist

    from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob

    args = parse_args()
    benchmarks = tuple(item.strip() for item in args.benchmarks.split(",") if item.strip())
    unknown = sorted(set(benchmarks) - set(SUPPORTED_BENCHMARKS))
    if unknown:
        raise ValueError(f"Unsupported benchmarks: {unknown}")

    rank, local_rank, world_size = _distributed_context()
    variant_root = (args.output_root / args.variant).resolve()
    if rank == 0:
        variant_root.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    tasks = build_tasks(args.data_root.resolve(), variant_root, benchmarks, args.seed)
    tasks = limit_tasks(tasks, args.max_tasks)
    if rank == 0:
        write_benchmark_sidecars(args.data_root.resolve(), variant_root, tasks)
        manifest_path = variant_root / "manifest.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(asdict(task), ensure_ascii=False) + "\n" for task in tasks),
            encoding="utf-8",
        )
        print(f"variant={args.variant} tasks={len(tasks)} manifest={manifest_path}", flush=True)
    if world_size > 1:
        dist.barrier()

    local_tasks = tasks[rank::world_size]
    pending = [task for task in local_tasks if not Path(task.output_path).is_file()]
    print(
        f"rank={rank}/{world_size} local_tasks={len(local_tasks)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    device = torch.device("cuda", local_rank)
    pipe = _load_pipeline(args.model_id, args.lora_path or None, device)
    completed = 0
    for batch in _batched(pending, args.batch_size):
        generators = [torch.Generator(device=device).manual_seed(task.seed) for task in batch]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result, _ = pipeline_with_logprob(
                pipe,
                prompt=[task.prompt for task in batch],
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=generators,
                output_type="pil",
                return_dict=True,
                noise_level=0.0,
                sde_frac=0.0,
                use_sa_solver=True,
            )
        for task, image in zip(batch, result.images, strict=True):
            path = Path(task.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
        completed += len(batch)
        if completed % 20 == 0 or completed == len(pending):
            print(f"rank={rank} completed={completed}/{len(pending)}", flush=True)

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        missing = sum(not Path(task.output_path).is_file() for task in tasks)
        print(f"generation_complete tasks={len(tasks)} missing={missing}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
