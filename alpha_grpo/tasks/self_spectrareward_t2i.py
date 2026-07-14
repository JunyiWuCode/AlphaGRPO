# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Text-to-image training with Self-SpectraReward."""

import contextlib
from functools import partial

import torch
import wandb
from accelerate.utils import gather
from trl.models import unwrap_model_for_generation

from common import AdvantageContext, GenerationStep
from rewards.self_spectrareward import compute_self_spectrareward
from tasks import register
from tasks.t2i import T2ITask
from utils import create_generator, log_images_to_tracker, sample_sde_window


@register("self_spectrareward_t2i")
class SelfSpectraRewardT2ITask(T2ITask):
    """T2I rollout where BAGEL itself supplies the prompt-likelihood reward."""

    def rollout(self, config, tokenizer, batch_data,
                global_step, stat_tracker, **kwargs):
        autocast = contextlib.nullcontext if config.use_lora else self.accelerator.autocast
        sde_window = sample_sde_window(config)

        if len(batch_data) != 2:
            raise RuntimeError("self_spectrareward_t2i expects (prompts, metadata) batches")
        prompts, prompt_metadata = batch_data
        input_lists = [[prompt] for prompt in prompts]

        generators = create_generator(prompts, global_step) if config.train.init_same_noise else None

        with autocast():
            with torch.no_grad():
                images, latents, image_log_probs, think_texts, text_per_token_log_probs, contexts = self.generate_image(
                    input_lists,
                    config,
                    think=config.sample.think,
                    sde_window=sde_window,
                    generators=generators,
                    deterministic=False,
                    repeat_think_text=config.sample.repeat_think_text,
                    prefix_think_token_ids=tokenizer(config.sample.prefix_think_text).input_ids if config.sample.prefix_think_text is not None else None,
                    use_flowcps=config.sample.use_flowcps,
                    task="t2i",
                    return_log_probs=True,
                )

        prompt_ids = tokenizer(
            prompts,
            padding="max_length",
            max_length=config.model.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.accelerator.device, dtype=torch.long)

        latents = torch.stack(latents, dim=1)
        if len(image_log_probs):
            image_log_probs = torch.stack(image_log_probs, dim=1)

        model = self.inferencer.model
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            self.inferencer.model = unwrapped_model
            rewards, reward_metadata = self._compute_self_spectrareward(unwrapped_model, images, prompts, config)
            self.inferencer.model = model

        sample = {
            "prompt_ids": prompt_ids,
            "prompts": prompts,
            "prompt_metadata": prompt_metadata,
            "contexts": contexts,
            "is_postprocessed": False,
            "input_images": None,
            "think_texts": think_texts,
            "images": images,
            "task": "self_spectrareward_t2i",
            "rewards": rewards,
            "reward_metadata": reward_metadata,
        }

        steps = []
        if think_texts is not None:
            steps.append(GenerationStep(
                step_type="think_text",
                texts=think_texts,
                text_per_token_log_probs=text_per_token_log_probs,
            ))
        steps.append(GenerationStep(
            step_type="image",
            latents=latents[:, :-1],
            next_latents=latents[:, 1:],
            image_log_probs=image_log_probs,
            sde_window=sde_window,
            task_type="t2i",
            images=images,
        ))
        sample["generation_steps"] = steps

        ctx = AdvantageContext(
            accelerator=self.accelerator,
            tokenizer=tokenizer,
            global_step=global_step,
            prompt_ids=prompt_ids,
        )
        sample["reward_postprocess_fn"] = partial(
            self.reward_postprocess,
            ctx=ctx,
        )

        if not config.train.async_reward_fn:
            sample, should_skip = sample["reward_postprocess_fn"](
                config, sample, stat_tracker=stat_tracker)
        else:
            should_skip = False

        return sample, should_skip

    def _compute_self_spectrareward(self, model, images, prompts, config):
        return compute_self_spectrareward(
            inferencer=self.inferencer,
            model=model,
            images=images,
            prompts=prompts,
            device=self.accelerator.device,
            use_vae=getattr(config, "self_spectrareward_use_vae", True),
            prompt_prefix=getattr(config, "self_spectrareward_prompt_prefix", ""),
            prompt_suffix=getattr(config, "self_spectrareward_prompt_suffix", ""),
            exclude_eos=getattr(config, "self_spectrareward_exclude_eos", True),
        )

    def eval(self, test_dataloader, config, global_step, autocast, **kwargs):
        all_rewards = []
        all_prompts = []
        all_images = []

        accelerator = self.accelerator
        inferencer = self.inferencer

        if accelerator.is_main_process:
            print("Start evaluation.")

        model = inferencer.model
        with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
            inferencer.model = unwrapped_model
            inferencer.model.eval()

            for prompts, _metadatas in test_dataloader:
                input_lists = [[prompt] for prompt in prompts]
                images, _ = inferencer.batch_generate_images(
                    input_lists=input_lists,
                    image_shape=(config.resolution, config.resolution),
                    cfg_text_scale=config.sample.cfg_text_scale,
                    cfg_img_scale=1.0,
                    num_timesteps=config.sample.eval_num_steps,
                    cfg_interval=config.sample.cfg_interval,
                    timestep_shift=config.sample.timestep_shift,
                    cfg_renorm_min=config.sample.cfg_renorm_min,
                    cfg_renorm_type=config.sample.cfg_renorm_type,
                    deterministic=True,
                    think=config.sample.think,
                    text_temperature=config.sample.eval_think_text_temperature,
                    text_top_p=config.sample.eval_think_text_top_p,
                    text_top_k=config.sample.eval_think_text_top_k,
                )
                rewards, _ = self._compute_self_spectrareward(
                    unwrapped_model, images, prompts, config)
                all_rewards.append(rewards)
                all_prompts.append(prompts)
                all_images.append(images)

        log_images_to_tracker(
            images=all_images[-1],
            prompts=all_prompts[-1],
            rewards=all_rewards[-1],
            accelerator=accelerator,
            global_step=global_step,
            config=config,
            max_images=len(all_prompts[-1]),
            prefix="eval",
        )

        aggregated_rewards = {
            key: gather(torch.cat([reward[key] for reward in all_rewards])).cpu().numpy()
            for key in all_rewards[0]
        }

        if accelerator.is_main_process:
            metrics = {
                f"eval/{key}": value.mean().item()
                for key, value in aggregated_rewards.items()
            }
            if config.logger_type == "wandb":
                wandb.log(metrics, step=global_step)
            else:
                accelerator.log(metrics, step=global_step)
            print("Evaluation results: self_spectrareward_t2i", {
                key: value.mean().item() for key, value in aggregated_rewards.items()
            })
            print("End of evaluation.")

        inferencer.model = model
        inferencer.model.train()
        return aggregated_rewards
