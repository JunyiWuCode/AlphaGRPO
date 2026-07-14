# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Self-SpectraReward utilities for BAGEL-style unified MLLMs."""

import contextlib
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _as_pil_images(images: Sequence[Image.Image] | torch.Tensor | np.ndarray) -> list[Image.Image]:
    if isinstance(images, torch.Tensor):
        if images.ndim != 4:
            raise ValueError(f"Expected 4D image tensor, got shape {tuple(images.shape)}")
        if images.shape[1] in (1, 3, 4):
            images = images.permute(0, 2, 3, 1)
        images = images.detach().float().cpu().clamp(0, 1).numpy()
        return [
            Image.fromarray((image * 255).round().astype(np.uint8)).convert("RGB")
            for image in images
        ]

    if isinstance(images, np.ndarray):
        if images.ndim != 4:
            raise ValueError(f"Expected 4D image array, got shape {images.shape}")
        if images.shape[1] in (1, 3, 4):
            images = images.transpose(0, 2, 3, 1)
        if images.dtype != np.uint8:
            images = np.clip(images, 0, 1)
            images = (images * 255).round().astype(np.uint8)
        return [Image.fromarray(image).convert("RGB") for image in images]

    return [
        image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
        for image in images
    ]


def _autocast_for(device):
    if torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def compute_self_spectrareward(
    inferencer,
    model,
    images,
    prompts,
    device,
    *,
    use_vae: bool = True,
    prompt_prefix: str = "",
    prompt_suffix: str = "",
    exclude_eos: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, list[float]]]:
    """Compute BAGEL's own image-conditioned prompt log-likelihood.

    This is the released Self-SpectraReward path: condition the current model on
    generated images, teacher-force the original prompts, and reward higher
    prompt likelihood. Ablation modes from the research branch intentionally stay
    out of the public release.
    """

    pil_images = _as_pil_images(images)
    scored_prompts = [f"{prompt_prefix}{prompt}{prompt_suffix}" for prompt in prompts]
    batch_size = len(pil_images)

    with _autocast_for(device):
        gen_context = inferencer.init_gen_context(batch_size)
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        past_key_values = gen_context["past_key_values"]

        if use_vae:
            vae_input, kv_lens, ropes = model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=pil_images,
                transforms=inferencer.vae_transform,
                new_token_ids=inferencer.new_token_ids,
            )
            vae_input = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in vae_input.items()
            }
            past_key_values = model.forward_cache_update_vae(
                inferencer.vae_model,
                past_key_values,
                **vae_input,
            )

        vit_input, kv_lens, ropes = model.prepare_vit_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=pil_images,
            transforms=inferencer.vit_transform,
            new_token_ids=inferencer.new_token_ids,
        )
        vit_input = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in vit_input.items()
        }
        past_key_values = model.forward_cache_update_vit(past_key_values, **vit_input)

        text_input, _, _ = model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=scored_prompts,
            tokenizer=inferencer.tokenizer,
            new_token_ids=inferencer.new_token_ids,
        )
        text_input = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in text_input.items()
        }
        _, logits = model.forward_cache_update_text(
            past_key_values,
            **text_input,
            return_logits=True,
        )

    text_token_lens = text_input["text_token_lens"].tolist()
    packed_text_ids = text_input["packed_text_ids"]
    eos_id = inferencer.new_token_ids["eos_token_id"] if exclude_eos else None

    mean_ces = []
    offset = 0
    for length in text_token_lens:
        sample_logits = logits[offset:offset + length]
        sample_ids = packed_text_ids[offset:offset + length]
        targets = sample_ids[1:]
        token_ce = F.cross_entropy(sample_logits[:-1], targets, reduction="none")
        if eos_id is not None:
            token_ce = token_ce[targets != eos_id]
        mean_ces.append(token_ce.float().mean())
        offset += length

    rewards = torch.stack([-ce for ce in mean_ces]).to(device)
    metadata = {"ce": [ce.item() for ce in mean_ces]}
    return {"self_spectrareward": rewards, "avg": rewards}, metadata
