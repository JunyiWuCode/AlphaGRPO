# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Text+Image to Text (ti2t) task."""

import contextlib
from functools import partial

import torch

from tasks import register, BagelBaseTask
from tasks.t2i import T2ITask
from common import GenerationStep, AdvantageContext


@register(('ti2t', 'ti2t_think'))
class TI2TTask(BagelBaseTask):

    def rollout(self, config, tokenizer, batch_data,
                global_step, stat_tracker, **kwargs):
        task = config.train.task
        should_skip = False
        autocast = contextlib.nullcontext if config.use_lora else self.accelerator.autocast

        input_images = None
        if len(batch_data) == 2:
            prompts, prompt_metadata = batch_data
            input_lists = [[prompt] for prompt in prompts]
        elif len(batch_data) == 3:
            prompts, prompt_metadata, input_images = batch_data
            input_lists = [[prompt, img] for prompt, img in zip(prompts, input_images)]
        else:
            raise RuntimeError('Unrecognized batch_data format.')

        with autocast():
            with torch.no_grad():
                output_texts, output_texts_per_token_log_probs, think_texts, think_texts_per_token_log_probs, contexts = self.generate_text(
                    input_lists,
                    config,
                    think=config.sample.think,
                    prefix_think_token_ids=tokenizer(config.sample.prefix_think_text).input_ids if config.sample.prefix_think_text is not None else None,
                    task=task,
                    return_log_probs=True,
                )

        prompt_ids = tokenizer(
            prompts,
            padding="max_length",
            max_length=config.model.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.accelerator.device)

        prompt_ids = prompt_ids.to(dtype=torch.long)

        # Compute rewards
        rewards_future = self.compute_rewards(
            dict(images=input_images, prompts=prompts, metadata=prompt_metadata),
            return_future=True)

        # Compute text format rewards (async, through same pipeline)
        text_rewards_future = None
        if config.sample.think and config.train.use_think_text_format_reward and self.text_reward_fn is not None and think_texts:
            text_rewards_future = self._compute_text_rewards(
                dict(prompts=think_texts), return_future=True)

        # Prepare sample
        sample = {
            "input_images": input_images,
            "prompt_ids": prompt_ids,
            "prompts": prompts,
            "prompt_metadata": prompt_metadata,
            "think_texts": think_texts,
            "contexts": contexts,
            "is_postprocessed": False,
            "task": task,
        }

        # Build generation steps
        steps = []
        if think_texts is not None:
            steps.append(GenerationStep(
                step_type='think_text',
                texts=think_texts,
                text_per_token_log_probs=think_texts_per_token_log_probs,
            ))
        if output_texts is not None:
            steps.append(GenerationStep(
                step_type='text',
                texts=output_texts,
                text_per_token_log_probs=output_texts_per_token_log_probs,
            ))
        sample['generation_steps'] = steps

        # Bind reward_postprocess (reuse T2ITask's)
        ctx = AdvantageContext(
            accelerator=self.accelerator, tokenizer=tokenizer,
            global_step=global_step, prompt_ids=prompt_ids
        )
        sample['reward_postprocess_fn'] = partial(
            T2ITask.reward_postprocess, ctx=ctx,
        )

        if text_rewards_future is not None:
            sample['text_rewards_future'] = text_rewards_future

        if config.train.async_reward_fn:
            sample['rewards_future'] = rewards_future
            should_skip = False
        else:
            sample['rewards_future'] = rewards_future
            sample, should_skip = sample['reward_postprocess_fn'](config, sample, stat_tracker=stat_tracker)

        return sample, should_skip

    def _compute_text_rewards(self, input_data, return_future=False):
        """Compute text rewards using text_reward_fn."""
        rewards_future = self.executor.submit(self.text_reward_fn, input_data)
        if return_future:
            return rewards_future
        rewards, reward_metadata = rewards_future.result()
        rewards = {
            key: torch.as_tensor(value, device=self.accelerator.device).float()
            for key, value in rewards.items()
        }
        return rewards, reward_metadata