# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""SpectraReward: image-conditioned prompt likelihood from a frozen MLLM.

Given generated images and their prompts, SpectraReward performs a single
teacher-forced multimodal forward pass and returns the mean prompt-token
log-likelihood. Higher values indicate that the image makes the prompt easier
for the MLLM to read back.
"""

from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class SpectraRewardScorer(torch.nn.Module):
    """Compute `mean log p(prompt | image)` with an off-the-shelf MLLM."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prompt_prefix: str = "",
        prompt_suffix: str = "",
        user_instruction: str = "",
        exclude_eos: bool = True,
        lazy_gpu: bool = True,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.user_instruction = user_instruction
        self.exclude_eos = exclude_eos
        self.lazy_gpu = lazy_gpu and torch.device(device).type == "cuda"

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self._tokenizer.padding_side = "right"

        target_device = "cpu" if lazy_gpu else device
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            device_map={"": "cpu"},
            low_cpu_mem_usage=False,
        ).to(target_device)
        self.model.requires_grad_(False)
        self.model.eval()

        self._end_ids = self._resolve_end_token_ids()
        if lazy_gpu:
            self._pin_cpu_tensors()

    def _resolve_end_token_ids(self) -> set[int]:
        """Collect end-of-turn token ids across common MLLM chat templates."""

        tok = self._tokenizer
        added = getattr(tok, "added_tokens_decoder", {}) or {}
        end_ids = set()
        if tok.eos_token_id is not None:
            end_ids.add(tok.eos_token_id)
        for name in ("<|im_end|>", "<end_of_turn>", "<|endoftext|>", "</s>"):
            tid = tok.convert_tokens_to_ids(name)
            if tid is None or tid == tok.unk_token_id:
                continue
            if tid not in added:
                continue
            end_ids.add(tid)
        return end_ids

    def _pin_cpu_tensors(self):
        if not torch.cuda.is_available():
            return
        for param in self.model.parameters():
            param.data = param.data.pin_memory()
        for buffer in self.model.buffers():
            if buffer.is_floating_point() or buffer.is_complex():
                buffer.data = buffer.data.pin_memory()

    def _offload_to_gpu(self):
        self.model.to(self.device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _offload_to_cpu(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.model.to("cpu", non_blocking=True)
        self._pin_cpu_tensors()

    @torch.no_grad()
    def __call__(self, images: List[Image.Image], prompts: List[str]) -> torch.Tensor:
        if self.lazy_gpu:
            self._offload_to_gpu()
        try:
            full_prompts = [
                f"{self.prompt_prefix}{prompt}{self.prompt_suffix}"
                for prompt in prompts
            ]
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": self.user_instruction},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": full_prompt}],
                    },
                ]
                for image, full_prompt in zip(images, full_prompts)
            ]

            prompt_starts = torch.tensor([
                self.processor.apply_chat_template(
                    message[:1],
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )["input_ids"].shape[1]
                for message in messages
            ])

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            logits = self.model(**inputs).logits
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device)
            prompt_mask = (
                positions.unsqueeze(0) >= prompt_starts.to(input_ids.device).unsqueeze(-1)
            ) & attention_mask.bool()

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = prompt_mask[:, 1:]

            if self.exclude_eos:
                for token_id in self._end_ids:
                    shift_mask &= shift_labels != token_id

            token_ce = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
            ).view(batch_size, -1)

            mean_ce = (token_ce * shift_mask).sum(-1) / shift_mask.sum(-1).clamp(min=1)
            return (-mean_ce).cpu()
        finally:
            if self.lazy_gpu:
                self._offload_to_cpu()
