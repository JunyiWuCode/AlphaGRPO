# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""ReMax algorithm."""

import torch
from algorithms import register, BaseAlgorithm


@register('remax')
class ReMaxAlgorithm(BaseAlgorithm):

    @classmethod
    def compute_text_loss(cls, config, sample, accelerator, info,
                          gen_step=None,
                          text_per_token_log_probs=None, text_entropies=None, text_token_lens=None,
                          **kwargs):
        advantages_clip = torch.clamp(
            sample["text_advantages"][:, 0], -config.train.adv_clip_max, config.train.adv_clip_max
        )
        normalized_advantages_clip = (advantages_clip / config.train.adv_clip_max) / 2.0

        text_advantages = torch.repeat_interleave(
            normalized_advantages_clip.to(accelerator.device),
            text_token_lens[:sample["advantages"].shape[0]].to(accelerator.device)
        )

        step_loss = -(text_advantages * text_per_token_log_probs).mean()

        info["text_loss"].append(step_loss.detach())

        return info, step_loss

    @classmethod
    def compute_image_loss(cls, config, sample, accelerator, info,
                           gen_step=None, j=0,
                           image_log_probs=None, **kwargs):
        advantages_clip = torch.clamp(
            sample["advantages"][:, j], -config.train.adv_clip_max, config.train.adv_clip_max
        )
        normalized_advantages_clip = (advantages_clip / config.train.adv_clip_max) / 2.0
        step_loss = -(image_log_probs.to(accelerator.device) * normalized_advantages_clip).mean()
        info["image_loss"].append(step_loss.detach())
        return info, step_loss