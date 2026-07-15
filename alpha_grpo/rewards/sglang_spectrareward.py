# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""SpectraReward client backed by SGLang prefill log probabilities."""

from __future__ import annotations

import base64
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Sequence

import requests
from PIL import Image
from requests.adapters import HTTPAdapter, Retry


def _input_ids(processor_output) -> list[int]:
    input_ids = processor_output["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("SpectraReward request preparation expects one example")
        input_ids = input_ids[0]
    return list(input_ids)


def mean_target_logprob(
    entries: Sequence[Sequence],
    expected_token_ids: Sequence[int],
    excluded_token_ids: Iterable[int] = (),
) -> float:
    """Locate the assistant target in SGLang input logprobs and average it."""

    returned_ids = [int(entry[1]) for entry in entries]
    expected = list(expected_token_ids)
    if not expected:
        raise ValueError("SpectraReward target has no tokens")

    match_start = None
    for start in range(len(returned_ids) - len(expected), -1, -1):
        if returned_ids[start : start + len(expected)] == expected:
            match_start = start
            break
    if match_start is None:
        raise ValueError(
            "SGLang input_token_logprobs do not contain the expected assistant "
            f"target: expected {len(expected)} tokens, received {len(returned_ids)}"
        )

    excluded = set(excluded_token_ids)
    values = []
    for offset, token_id in enumerate(expected):
        logprob = entries[match_start + offset][0]
        if token_id not in excluded and logprob is not None:
            values.append(float(logprob))
    if not values:
        raise ValueError("No scored prompt tokens remain after excluding end tokens")
    return sum(values) / len(values)


class SpectraRewardSGLangClient:
    """Compute mean log p(prompt | image) using SGLang's `/generate` API."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
        prompt_prefix: str = "",
        prompt_suffix: str = "",
        user_instruction: str = "",
        exclude_eos: bool = True,
        max_concurrent: int = 8,
        timeout: float = 600,
        processor=None,
        session=None,
    ):
        self.model_id = model_id
        self.generate_url = f"{base_url.rstrip('/').removesuffix('/v1')}/generate"
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.user_instruction = user_instruction
        self.exclude_eos = exclude_eos
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        if processor is None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.processor = processor
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.end_token_ids = self._resolve_end_token_ids() if exclude_eos else set()
        self.session = session or self._make_session(max_concurrent)

    @staticmethod
    def _make_session(max_concurrent: int):
        session = requests.Session()
        session.trust_env = False
        retry = Retry(
            total=20,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=False,
        )
        pool_size = max(1, max_concurrent)
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            pool_block=True,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _resolve_end_token_ids(self) -> set[int]:
        end_ids = set()
        if self.tokenizer.eos_token_id is not None:
            end_ids.add(int(self.tokenizer.eos_token_id))
        for token in ("<|im_end|>", "<end_of_turn>", "<|endoftext|>", "</s>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                end_ids.add(int(token_id))
        return end_ids

    @staticmethod
    def _image_data_uri(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _prepare_request(self, image: Image.Image, prompt: str):
        target = f"{self.prompt_prefix}{prompt}{self.prompt_suffix}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.user_instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target}],
            },
        ]

        prefix_ids = _input_ids(
            self.processor.apply_chat_template(
                messages[:1],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        )
        full_ids = _input_ids(
            self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=False,
            )
        )
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "The reward model chat template does not share a stable assistant prefix"
            )
        expected_ids = full_ids[len(prefix_ids) :]
        if not expected_ids:
            raise ValueError("The SpectraReward prompt tokenized to an empty target")

        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        payload = {
            "text": rendered,
            "image_data": self._image_data_uri(image),
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "return_logprob": True,
            # Include one boundary token so every target token has a next-token score.
            "logprob_start_len": max(len(prefix_ids) - 1, 0),
            "top_logprobs_num": 0,
            "return_text_in_logprobs": False,
        }
        return payload, expected_ids

    def score_one(self, image: Image.Image, prompt: str) -> float:
        payload, expected_ids = self._prepare_request(image, prompt)
        response = self.session.post(
            self.generate_url,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        entries = body.get("meta_info", {}).get("input_token_logprobs")
        if entries is None:
            raise RuntimeError(
                "SGLang did not return meta_info.input_token_logprobs; verify "
                "return_logprob support and use sglang==0.5.5.post3"
            )
        return mean_target_logprob(entries, expected_ids, self.end_token_ids)

    def __call__(self, images: Sequence[Image.Image], prompts: Sequence[str]):
        if len(images) != len(prompts):
            raise ValueError(
                f"SpectraReward received {len(images)} images and {len(prompts)} prompts"
            )
        workers = max(1, min(self.max_concurrent, len(images)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self.score_one, images, prompts))
