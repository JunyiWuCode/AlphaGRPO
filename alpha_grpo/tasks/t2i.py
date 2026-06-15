# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Text-to-Image (t2i) and Image Editing (ti2i) task."""

import contextlib
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import wandb
from accelerate.utils import gather_object
from trl.models import unwrap_model_for_generation

from tasks import register, BagelBaseTask
from common import GenerationStep, AdvantageContext, compute_advantages
from utils import create_generator, sample_sde_window, log_images_to_tracker, concat_images_list


@register(('t2i', 'ti2i'))
class T2ITask(BagelBaseTask):

    def rollout(self, config, tokenizer, batch_data,
                global_step, stat_tracker, **kwargs):
        task = config.train.task
        should_skip = False
        autocast = contextlib.nullcontext if config.use_lora else self.accelerator.autocast
        sde_window = sample_sde_window(config)

        input_images = None
        if len(batch_data) == 2:
            prompts, prompt_metadata = batch_data
            input_lists = [[prompt] for prompt in prompts]
        elif len(batch_data) == 3:
            prompts, prompt_metadata, input_images = batch_data
            input_lists = [[prompt, img] for prompt, img in zip(prompts, input_images)]
        else:
            raise RuntimeError('Unrecognized batch data format')

        if config.train.init_same_noise:
            generators = create_generator(prompts, global_step)
        else:
            generators = None

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
                    task='t2i' if task == 't2i' else 'edit',
                    return_log_probs=True,
                )

        prompt_ids = tokenizer(
            prompts,
            padding="max_length",
            max_length=config.model.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.accelerator.device)

        latents = torch.stack(latents, dim=1)
        if len(image_log_probs):
            image_log_probs = torch.stack(image_log_probs, dim=1)

        prompt_ids = prompt_ids.to(dtype=torch.long)

        # Compute rewards
        rewards_future = self.compute_rewards(
            dict(images=images, prompts=prompts, metadata=prompt_metadata), return_future=True)

        # Compute text format rewards (async, through same pipeline)
        text_rewards_future = None
        if config.sample.think and config.train.use_think_text_format_reward and self.text_reward_fn is not None and think_texts:
            text_rewards_future = self._compute_text_rewards(
                dict(prompts=think_texts), return_future=True)

        # Prepare sample
        sample = {
            "prompt_ids": prompt_ids,
            "prompts": prompts,
            "prompt_metadata": prompt_metadata,
            "contexts": contexts,
            "is_postprocessed": False,
            "input_images": input_images,
            "think_texts": think_texts,
            "images": images,
            "task": task
        }

        # Build generation steps
        steps = []
        if think_texts is not None:
            steps.append(GenerationStep(
                step_type='think_text',
                texts=think_texts,
                text_per_token_log_probs=text_per_token_log_probs,
            ))
        steps.append(GenerationStep(
            step_type='image',
            latents=latents[:, :-1],
            next_latents=latents[:, 1:],
            image_log_probs=image_log_probs,
            sde_window=sde_window,
            task_type=task,
            images=images,
        ))
        sample['generation_steps'] = steps

        # Bind reward_postprocess
        ctx = AdvantageContext(
            accelerator=self.accelerator, tokenizer=tokenizer,
            global_step=global_step, prompt_ids=prompt_ids
        )
        sample['reward_postprocess_fn'] = partial(
            self.reward_postprocess, ctx=ctx,
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

    def log_training_step(self, train_samples, accelerator, global_step, config, prefix=""):
        """Log T2I training: generated images grid + rewards."""
        sample = train_samples[0]
        images = sample['images']
        if sample.get('task') == 'ti2i':
            images = concat_images_list(sample.get('input_images', []), images)
        log_prefix = f"train_{prefix}" if prefix else "train"
        log_images_to_tracker(
            images=images,
            prompts=sample['prompts'],
            rewards=sample['rewards'],
            accelerator=accelerator,
            global_step=global_step,
            config=config,
            max_images=config.max_images_per_log,
            prefix=log_prefix,
        )

    @staticmethod
    def get_log_images(sample):
        if sample.get('task') == 'ti2i':
            return concat_images_list(sample['input_images'], sample['images'])
        return sample.get('images', [])

    @classmethod
    def reward_postprocess(cls, config, sample, stat_tracker, ctx=None):
        """T2I reward postprocess: resolve -> think_text_reward -> compute_advantages."""
        if sample["is_postprocessed"]:
            return sample, False

        # Resolve async rewards
        if "rewards_future" in sample:
            rewards_future = sample.pop('rewards_future')
            rewards, reward_metadata = rewards_future.result()
            rewards = {k: torch.as_tensor(v, device=ctx.accelerator.device).float() for k, v in rewards.items()}
            sample["rewards"] = rewards
            sample["reward_metadata"] = reward_metadata

        sample["rewards"]["ori_avg"] = deepcopy(sample["rewards"]["avg"])

        # Resolve text format rewards (submitted async in rollout)
        if 'text_rewards_future' in sample:
            text_rewards_future = sample.pop('text_rewards_future')
            text_rewards, _ = text_rewards_future.result()
            text_rewards = {k: torch.as_tensor(v, device=ctx.accelerator.device).float() for k, v in text_rewards.items()}
            think_text_reward = text_rewards['avg']
            sample['rewards']['avg'] += think_text_reward * config.train.think_text_weight
            sample['rewards']['think_text_reward'] = think_text_reward

        return compute_advantages(config, sample, stat_tracker, ctx)

    def eval(self, test_dataloader, config, global_step, autocast, **kwargs):
        """T2I evaluation."""

        task = config.train.task
        eval_task = 'edit' if task == 'ti2i' else 't2i'

        all_rewards = []
        all_text_rewards = []
        all_think_text_length = []
        all_prompts = []
        all_images = []
        all_think_texts = []

        accelerator = self.accelerator
        inferencer = self.inferencer

        if accelerator.is_main_process:
            print('Start evaluation.')
        model = inferencer.model
        with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
            inferencer.model = unwrapped_model
            inferencer.model.eval()

            for batch in test_dataloader:
                if len(batch) == 2:
                    prompts, metadatas = batch
                    input_lists = [[p] for p in prompts]
                else:
                    prompts, metadatas, input_images = batch
                    input_lists = [[p, img] for p, img in zip(prompts, input_images)]

                images, think_texts = inferencer.batch_generate_images(
                    input_lists=input_lists,
                    image_shape=(config.resolution, config.resolution),
                    cfg_text_scale=config.sample.cfg_text_scale,
                    cfg_img_scale=config.sample.cfg_image_scale if eval_task == 'edit' else 1.0,
                    num_timesteps=config.sample.eval_num_steps,
                    cfg_interval=config.sample.cfg_interval_edit if eval_task == 'edit' else config.sample.cfg_interval,
                    timestep_shift=config.sample.timestep_shift,
                    cfg_renorm_min=config.sample.cfg_renorm_min,
                    cfg_renorm_type=config.sample.cfg_renorm_type_edit if eval_task == 'edit' else config.sample.cfg_renorm_type,
                    deterministic=True,
                    think=config.sample.think,
                    text_temperature=config.sample.eval_think_text_temperature,
                    text_top_p=config.sample.eval_think_text_top_p,
                    text_top_k=config.sample.eval_think_text_top_k,
                )

                rewards = self.compute_rewards(
                    dict(images=images, prompts=prompts, metadata=metadatas),
                    return_future=config.train.async_reward_fn)
                if not config.train.async_reward_fn:
                    rewards, _ = rewards

                if think_texts is not None:
                    all_think_texts.append(think_texts)
                    if self.text_reward_fn is not None:
                        text_rewards = self._compute_text_rewards(
                            dict(prompts=think_texts),
                            return_future=config.train.async_reward_fn)
                        if not config.train.async_reward_fn:
                            text_rewards, _ = text_rewards
                        all_text_rewards.append(text_rewards)

                all_rewards.append(rewards)
                all_prompts.append(prompts)
                all_images.append(images)

        if config.train.async_reward_fn:
            all_rewards_future = all_rewards
            all_rewards = []
            for rewards_future in all_rewards_future:
                rewards, _ = rewards_future.result()
                rewards = {k: torch.as_tensor(v, device=accelerator.device).float() for k, v in rewards.items()}
                all_rewards.append(rewards)

            all_text_rewards_future = all_text_rewards
            all_text_rewards = []
            for text_rewards_future in all_text_rewards_future:
                text_rewards, _ = text_rewards_future.result()
                text_rewards = {k: torch.as_tensor(v, device=accelerator.device).float() for k, v in text_rewards.items()}
                all_text_rewards.append(text_rewards)

        for rewards, text_rewards in zip(all_rewards, all_text_rewards):
            rewards["think_text_format_reward"] = text_rewards['avg']

        log_images_to_tracker(
            images=images, prompts=prompts, rewards=rewards,
            accelerator=accelerator, global_step=global_step,
            config=config, max_images=len(prompts), prefix="eval"
        )

        aggregated_rewards = {k: accelerator.gather(torch.cat([r[k] for r in all_rewards])).cpu().numpy()
                              for k in all_rewards[0]}
        all_think_text_length = [np.mean([len(t.split(' ')) for t in think_texts]) for think_texts in all_think_texts]
        aggregated_think_text_length = accelerator.gather(torch.as_tensor(np.mean(all_think_text_length)).cuda()).mean()

        if accelerator.is_main_process:
            if config.logger_type == 'wandb':
                wandb.log({f"eval/{key}": value.mean().item() for key, value in aggregated_rewards.items()}, step=global_step)
                if config.sample.think:
                    wandb.log({"eval/think_text_length": aggregated_think_text_length}, step=global_step)

                columns = ["Prompt", "Generated image", "Think text", "Reward"]
                all_think_texts = all_think_texts if len(all_think_texts) else [[''] * len(all_images[-1]) for _ in range(len(all_images))]
                my_table = wandb.Table(columns=columns)
                for idx in range(2):
                    for img, prompt, think_text, reward in zip(all_images[-idx], all_prompts[-idx], all_think_texts[-idx], all_rewards[-idx]['avg']):
                        my_table.add_data(prompt, wandb.Image(img), think_text, reward)
                wandb.log({f"eval/step_{global_step}": my_table}, step=global_step)
            else:
                accelerator.log({f"eval/{key}": value.mean().item() for key, value in aggregated_rewards.items()}, step=global_step)
                if config.sample.think:
                    accelerator.log({"eval/think_text_length": aggregated_think_text_length}, step=global_step)

        inferencer.model = model
        inferencer.model.train()

        if accelerator.is_main_process:
            print('Evaluation results: t2i', {k: v.mean().item() for k, v in aggregated_rewards.items()})
            if config.sample.think:
                print('think_text_length', aggregated_think_text_length)
            print('End of evaluation.')
        return aggregated_rewards