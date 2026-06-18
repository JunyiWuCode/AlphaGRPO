# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""NFT (Noise-Free Training) algorithm for flow-matching diffusion models."""

import torch

from algorithms import BaseAlgorithm, register


@register('nft')
class NFTAlgorithm(BaseAlgorithm):
    """Reward-weighted flow-matching with positive/negative predictions."""

    @classmethod
    def compute_text_loss(cls, config, sample, accelerator, info, **kwargs):
        raise NotImplementedError(
            "NFT does not support text loss. Set config.train.algorithm to "
            "'grpo' or 'remax' for text steps, and config.train.image_algorithm "
            "to 'nft' for image steps only."
        )

    @classmethod
    def compute_image_loss(cls, config, sample, accelerator, info,
                           gen_step=None, j=0,
                           model_output=None, **kwargs):
        v_current = model_output
        v_old = gen_step.ema_model_output[:, j].detach()
        x0 = gen_step.next_latents[:, -1].detach()
        sigma = gen_step.fm_sigma[j]
        noise_j = gen_step.fm_noise[j]
        target = noise_j - x0

        beta = getattr(config, 'nft_beta', getattr(config, 'beta', 1.0))
        v_pos = beta * v_current + (1 - beta) * v_old
        v_neg = (1 + beta) * v_old - beta * v_current

        spatial_dims = tuple(range(1, v_current.ndim))

        pos_error = v_pos - target
        with torch.no_grad():
            pos_weight = pos_error.detach().float().abs().mean(
                dim=spatial_dims, keepdim=True).clamp(min=1e-5)
        positive_loss = (pos_error ** 2 / pos_weight).mean(dim=spatial_dims) * sigma

        neg_error = v_neg - target
        with torch.no_grad():
            neg_weight = neg_error.detach().float().abs().mean(
                dim=spatial_dims, keepdim=True).clamp(min=1e-5)
        negative_loss = (neg_error ** 2 / neg_weight).mean(dim=spatial_dims) * sigma

        advantages = torch.clamp(
            sample["advantages"][:, j], -config.train.adv_clip_max, config.train.adv_clip_max
        )
        reward_weight = (advantages / config.train.adv_clip_max) / 2.0 + 0.5
        reward_weight = reward_weight.clamp(0, 1)

        patches_per_sample = model_output.shape[0] // reward_weight.shape[0]
        reward_weight = reward_weight.repeat_interleave(patches_per_sample)

        policy_loss = (
            reward_weight * positive_loss / beta
            + (1.0 - reward_weight) * negative_loss / beta
        )
        policy_loss = (policy_loss * config.train.adv_clip_max).mean()
        step_loss = policy_loss

        if config.train.image_beta > 0 and gen_step.ref_model_output is not None:
            v_ref = gen_step.ref_model_output[:, j].detach()
            kl_loss = ((v_current - v_ref) ** 2).mean(dim=spatial_dims).mean()
            step_loss = step_loss + config.train.image_beta * kl_loss
            info["image_kl_loss"].append(kl_loss.detach())

        info["image_policy_loss"].append(policy_loss.detach())
        with torch.no_grad():
            info["old_deviate"].append(((v_current - v_old) ** 2).mean())

        return info, step_loss
