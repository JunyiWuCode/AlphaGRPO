# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""AWM (Advantage-Weighted Model) algorithm."""

import torch
from algorithms import register, BaseAlgorithm


@register('awm')
class AWMAlgorithm(BaseAlgorithm):

    @classmethod
    def compute_text_loss(cls, config, sample, accelerator, info, **kwargs):
        raise NotImplementedError(
            "AWM does not support text loss. Set config.train.algorithm to "
            "'grpo' or 'remax' for text steps, and config.train.image_algorithm "
            "to 'awm' for image steps only."
        )

    @classmethod
    def compute_image_loss(cls, config, sample, accelerator, info,
                           gen_step=None, j=0,
                           image_log_probs=None, model_output=None, std_dev_t=None,
                           **kwargs):
        # Extract rollout/ref data from gen_step
        old_image_lp_j = gen_step.image_log_probs[:, j]

        advantages = torch.clamp(
            sample["advantages"][:, j], -config.train.adv_clip_max, config.train.adv_clip_max
        )
        if config.train.advantage_max is not None:
            advantages = advantages / config.train.adv_clip_max * config.train.advantage_max

        if config.train.true_ratio:
            image_ratio = torch.exp(image_log_probs - old_image_lp_j.detach())
        else:
            image_ratio = torch.exp(image_log_probs - image_log_probs.detach())
        image_ratio = image_ratio.to('cuda')

        image_unclipped_loss = -advantages * image_ratio
        image_clipped_loss = -advantages * torch.clamp(
            image_ratio, 1.0 - config.train.image_clip_range, 1.0 + config.train.image_clip_range
        )
        image_policy_loss = torch.maximum(image_unclipped_loss, image_clipped_loss)

        if config.train.rollout_correction:
            weight = torch.exp(image_log_probs.detach() - old_image_lp_j).clamp(
                max=config.train.correction_truncated_threshold)
            weight = weight.to(accelerator.device)
            if accelerator.is_main_process:
                print('correction image weight', torch.exp(image_log_probs.detach() - old_image_lp_j))
            image_policy_loss = weight * image_policy_loss

        image_policy_loss = torch.mean(image_policy_loss)

        step_loss = image_policy_loss

        if config.train.image_beta > 0:
            ref_mo_j = gen_step.ref_model_output[:, j]
            image_kl_loss = cls._kl_mse(model_output, ref_mo_j, std_dev_t, config.train.kl_weight)
            step_loss = step_loss + config.train.image_beta * image_kl_loss

        if config.train.ema_beta > 0 and gen_step.ema_model_output is not None:
            ema_mo_j = gen_step.ema_model_output[:, j]
            ema_kl_loss = cls._kl_mse(model_output, ema_mo_j, std_dev_t, config.train.kl_ema_weight)
            step_loss = step_loss + config.train.ema_beta * ema_kl_loss

        info["image_approx_kl"].append(
            0.5 * torch.mean((image_log_probs.detach() - old_image_lp_j) ** 2)
        )
        info["image_clipfrac"].append(
            torch.mean((torch.abs(image_ratio - 1.0) > config.train.image_clip_range).float())
        )
        info["image_policy_loss"].append(image_policy_loss.detach())
        info["image_ratio"].append(image_ratio.mean().detach())
        if config.train.image_beta > 0:
            info["image_kl_loss"].append(image_kl_loss.detach())
        if config.train.ema_beta > 0 and gen_step.ema_model_output is not None:
            info["image_ema_kl_loss"].append(ema_kl_loss.detach())

        return info, step_loss

    @staticmethod
    def _kl_mse(model_output, ref_output, std_dev_t, kl_weight):
        """MSE-based KL with optional ELBO weighting (/ 2*std_dev_t^2)."""
        spatial_dims = tuple(range(1, model_output.ndim))
        mse = ((model_output - ref_output) ** 2).mean(dim=spatial_dims)
        if kl_weight == 'ELBO':
            mse = mse / (2 * std_dev_t ** 2)
        elif kl_weight != 'Uniform':
            raise ValueError(f"Unknown kl_weight: {kl_weight}")
        return mse.mean()