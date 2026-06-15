# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Reflection task (v2): two-stage ODE preselection -> SDE with think-text refinement."""

import contextlib
import random
from concurrent import futures
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import wandb
from PIL import Image
from accelerate.utils import gather, gather_object
from trl.models import unwrap_model_for_generation

from tasks import register, BagelBaseTask, get_task
from common import GenerationStep, AdvantageContext, compute_advantages
from utils import (
    scatter_data, reduction_prompt_num, create_generator, sample_sde_window,
    log_images_to_tracker, concat_images_list,
)
from constants import (
    reflection_and_regenerate_prompt_with_caption,
)


def pil2tensor(pil_image):
    return torch.tensor(np.array(pil_image))


def tensor2pil(tensor):
    return Image.fromarray(tensor.numpy().astype(np.uint8))


# Shared queue for two-stage accumulation (persists across calls)
_t2i_result_queue = []


@register('reflect')
class ReflectTask(BagelBaseTask):

    def rollout(self, config, tokenizer, batch_data,
                global_step, stat_tracker, **kwargs):
        global _t2i_result_queue
        should_skip = False
        collect_samples = []
        autocast = contextlib.nullcontext if config.use_lora else self.accelerator.autocast
        accelerator = self.accelerator

        def should_rollout_reflect():
            return len(_t2i_result_queue) % (config.train.gradient_accumulation_steps * config.sample.num_batches_per_ppo_epoch) == 0

        # Reflection two-stage sampling
        if len(batch_data) == 2:
            images = None
            prompts, prompt_metadata = batch_data

            t2i_sample = None
            if config.train.mix_t2i_task:
                # Stage 1: SDE generation (reuse T2I rollout)
                T2ITaskClass = get_task('t2i')
                t2i_task = T2ITaskClass(**self._init_kwargs())
                t2i_sample, should_skip = t2i_task.rollout(
                    config=config, tokenizer=tokenizer,
                    batch_data=batch_data,
                    global_step=global_step, stat_tracker=stat_tracker,
                )
                if should_skip:
                    return t2i_sample, True

                _t2i_result_queue.append(t2i_sample)
                print('Add to T2I queue:', len(_t2i_result_queue))

                if not should_rollout_reflect():
                    return t2i_sample, False
                else:
                    collect_samples.append(t2i_sample)
            else:
                # Stage 1 ODE deterministic preselection
                all_stage1_prompts, all_stage1_prompt_metadata = reduction_prompt_num(
                    prompts, prompt_metadata, getattr(config.sample, 'stage1_reduction_ratio', 0.5))

                indices = list(range(len(all_stage1_prompts)))
                random.shuffle(indices)
                select_indices = scatter_data(indices)
                stage1_prompts = [all_stage1_prompts[i] for i in select_indices]
                stage1_prompt_metadata = [all_stage1_prompt_metadata[i] for i in select_indices]

                with autocast():
                    with torch.no_grad():
                        input_lists_stage1 = [[p] for p in stage1_prompts]
                        stage1_images = self.generate_image(
                            input_lists_stage1, config,
                            num_timesteps=config.sample.eval_num_steps,
                            think=False, deterministic=True,
                        )

                        rewards_stage1_future = self.compute_rewards(
                            dict(images=stage1_images, prompts=stage1_prompts, metadata=stage1_prompt_metadata),
                            return_future=True
                        )

                        def fake_reward_postprocess_fn(config, sample, stat_tracker=None):
                            rewards_future = sample.pop('rewards_future')
                            rewards, reward_metadata = rewards_future.result()
                            rewards = {k: torch.as_tensor(v, device=accelerator.device).float() for k, v in rewards.items()}
                            sample['rewards'] = rewards
                            sample['rewards_metadata'] = reward_metadata
                            return sample, False

                        fake_sample = dict(
                            prompts=stage1_prompts,
                            prompt_metadata=stage1_prompt_metadata,
                            reward_postprocess_fn=fake_reward_postprocess_fn,
                            rewards_future=rewards_stage1_future,
                            images=stage1_images
                        )

                        _t2i_result_queue.append(fake_sample)

                        if not should_rollout_reflect():
                            return [], False

        else:
            # Direct read stage1 prompts + init images from batch_data
            prompts, prompt_metadata, images = batch_data
            images = [img.resize((config.resolution, config.resolution)) for img in images]

            with autocast():
                with torch.no_grad():
                    init_reward_futures = self.compute_rewards(
                        dict(images=images, prompts=prompts, metadata=prompt_metadata),
                        return_future=True
                    )

            def fake_reward_postprocess_fn(config, sample, stat_tracker=None):
                rewards_future = sample.pop('rewards_future')
                rewards, reward_metadata = rewards_future.result()
                rewards = {k: torch.as_tensor(v, device=accelerator.device).float() for k, v in rewards.items()}
                sample['rewards'] = rewards
                sample['rewards_metadata'] = reward_metadata
                return sample, False

            fake_sample = dict(
                prompts=prompts,
                prompt_metadata=prompt_metadata,
                reward_postprocess_fn=fake_reward_postprocess_fn,
                rewards_future=init_reward_futures,
                images=images
            )

            _t2i_result_queue.append(fake_sample)
            if not should_rollout_reflect():
                return [], False

        # Process all queued stage1 results
        for t2i_sample in _t2i_result_queue:
            t2i_sample, should_skip = t2i_sample["reward_postprocess_fn"](config, t2i_sample, stat_tracker=stat_tracker)
            stage1_prompts = t2i_sample['prompts']
            stage1_prompt_metadata = t2i_sample['prompt_metadata']
            stage1_avg = t2i_sample['rewards']['avg']
            stage1_images = t2i_sample['images']

            sde_window = sample_sde_window(config)

            # Gather data across processes
            gathered_prompts = np.asarray(gather_object(stage1_prompts))
            gathered_prompt_metadata = np.asarray(gather_object(stage1_prompt_metadata))
            gathered_images = accelerator.gather(torch.stack([pil2tensor(img) for img in stage1_images]).cuda()).cpu()
            gathered_stage1_avg = accelerator.gather(stage1_avg).cpu().numpy()

            # Select lowest reward data for each unique prompt
            selected_data = []
            for prompt in np.unique(gathered_prompts):
                mask = gathered_prompts == prompt
                if config.train.sample_stage1_reward:
                    lowest_idx = torch.multinomial(torch.tensor(
                        1 - gathered_stage1_avg[mask]).softmax(dim=0), num_samples=1).item()
                else:
                    lowest_idx = np.argmin(gathered_stage1_avg[mask])

                selected_data.append({
                    'prompt': prompt,
                    'metadata': gathered_prompt_metadata[mask][lowest_idx],
                    'image': gathered_images[mask][lowest_idx],
                    'reward': gathered_stage1_avg[mask][lowest_idx]
                })

            prompts_mean = {item['prompt']: item['reward'] for item in selected_data}

            # Expand as group_size for GRPO
            expanded_data = selected_data * config.sample.num_image_per_prompt
            selected_indices = scatter_data(torch.randperm(len(expanded_data)))

            selected_prompts = [expanded_data[i]['prompt'] for i in selected_indices]
            selected_metadatas = [expanded_data[i]['metadata'] for i in selected_indices]
            selected_images = [tensor2pil(expanded_data[i]['image']) for i in selected_indices]
            selected_rewards = [expanded_data[i]['reward'] for i in selected_indices]

            # Stage 2: SDE stochastic generation with reflection
            input_lists = [[init_img, reflection_and_regenerate_prompt_with_caption.format(p)]
                           for p, init_img in zip(selected_prompts, selected_images)]

            repeat_think_text = config.sample.repeat_think_text
            prompts_for_sample = selected_prompts
            prompt_metadata_for_sample = selected_metadatas

            if repeat_think_text > 1:
                prompts_for_sample = prompts_for_sample * repeat_think_text
                prompt_metadata_for_sample = prompt_metadata_for_sample * repeat_think_text
                selected_rewards = selected_rewards * repeat_think_text

            if config.train.init_same_noise:
                generators = create_generator(prompts_for_sample, global_step)
            else:
                generators = None

            with autocast(), torch.no_grad():
                images, latents, image_log_probs, think_texts, text_per_token_log_probs, contexts = self.generate_image(
                    input_lists, config,
                    think=config.sample.think,
                    sde_window=sde_window,
                    generators=generators,
                    deterministic=False,
                    repeat_think_text=config.sample.repeat_think_text,
                    prefix_think_token_ids=tokenizer(config.sample.prefix_think_text).input_ids if config.sample.prefix_think_text is not None else None,
                    use_flowcps=config.sample.use_flowcps,
                    task='edit',
                    return_log_probs=True,
                )

            prompt_ids = tokenizer(
                prompts_for_sample,
                padding="max_length",
                max_length=config.model.max_sequence_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            latents = torch.stack(latents, dim=1)
            if len(image_log_probs):
                image_log_probs = torch.stack(image_log_probs, dim=1)

            prompt_ids = prompt_ids.to(dtype=torch.long)

            rewards_future = self.compute_rewards(
                dict(images=images, prompts=prompts_for_sample, metadata=prompt_metadata_for_sample),
                return_future=True)

            sample = {
                "prompt_ids": prompt_ids,
                "prompts": prompts_for_sample,
                "prompt_metadata": prompt_metadata_for_sample,
                "init_rewards_gallery": gathered_stage1_avg,
                "rewards_future": rewards_future,
                "is_postprocessed": False,
                "contexts": contexts,
                "init_images": selected_images,
                "think_texts": think_texts,
                "images": images,
                "task": 'reflect'
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
                task_type='edit',
                images=images,
            ))
            sample['generation_steps'] = steps

            # Compute text format rewards (async, through same pipeline as image rewards)
            if self.text_reward_fn is not None and think_texts:
                sample['text_rewards_future'] = self._compute_text_rewards(
                    dict(prompts=sample['think_texts']), return_future=True)

            # Bind reward_postprocess
            ctx = AdvantageContext(
                accelerator=accelerator, tokenizer=tokenizer,
                global_step=global_step, prompt_ids=prompt_ids
            )
            sample['reward_postprocess_fn'] = partial(
                self.reward_postprocess, ctx=ctx, prompts_mean=prompts_mean,
            )

            if config.train.async_reward_fn:
                if not isinstance(selected_rewards, futures.Future):
                    sample['init_rewards'] = torch.tensor(selected_rewards).clone().cuda()
                else:
                    sample["init_rewards_future"] = selected_rewards
                should_skip = False
            else:
                sample['init_rewards'] = torch.tensor(selected_rewards).clone().cuda()
                sample, should_skip = sample['reward_postprocess_fn'](config, sample, stat_tracker=stat_tracker)

            collect_samples.append(sample)

        _t2i_result_queue = []
        return collect_samples, should_skip

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

    def collect_metrics(self, sample, info):
        """Reflect metrics: refined rewards + init reward for comparison."""
        for key, value in sample["rewards"].items():
            info[f"reward_{key}"].append(torch.mean(value))
        if 'init_rewards_gallery' in sample:
            info["reward_init_avg"].append(torch.as_tensor(sample["init_rewards_gallery"]).float().mean())

    def log_training_step(self, train_samples, accelerator, global_step, config, prefix=""):
        """Log reflect: init + refined images concatenated side by side."""
        sample = train_samples[-1]
        images = concat_images_list(sample.get('init_images', []), sample.get('images', []))
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
    def get_log_sample_index(train_samples):
        return -1

    @staticmethod
    def get_log_images(sample):
        return concat_images_list(sample.get('init_images', []), sample.get('images', []))

    @classmethod
    def reward_postprocess(cls, config, sample, stat_tracker,
                           ctx=None, prompts_mean=None):
        """Reflection reward postprocess: two-stage rewards + multiple transforms + improvement check."""
        if sample["is_postprocessed"]:
            return sample, False

        accelerator = ctx.accelerator

        # 1. Resolve init_rewards_future (stage 1)
        if 'init_rewards_future' in sample:
            init_rewards_future = sample.pop('init_rewards_future')
            init_rewards, init_reward_metadata = init_rewards_future.result()
            init_rewards = {k: torch.as_tensor(v, device=accelerator.device).float() for k, v in init_rewards.items()}
            sample['init_rewards'] = init_rewards['avg']
            sample['init_rewards_metadata'] = init_reward_metadata

        # 2. Resolve rewards_future (stage 2)
        if "rewards_future" in sample:
            rewards_future = sample.pop('rewards_future')
            rewards, reward_metadata = rewards_future.result()
            rewards = {k: torch.as_tensor(v, device=accelerator.device).float() for k, v in rewards.items()}
            sample["rewards"] = rewards
            sample["reward_metadata"] = reward_metadata

        rewards = sample["rewards"]
        sample["rewards"]['init_reward'] = sample['init_rewards']
        sample["rewards"]["image_avg"] = deepcopy(sample["rewards"]["avg"])

        # 3. Reflection-specific reward modifications
        if config.train.clamp_no_improve_reward:  # FPR
            mask = sample['rewards']['avg'] <= sample['rewards']['init_reward']
            gather_prompts = np.asarray(gather_object(sample['prompts']))
            gather_init_rewards = gather(sample['rewards']['init_reward']).cpu().numpy()
            gather_rewards = gather(sample['rewards']['avg']).cpu().numpy()

            prompt2min_init_reward = {}
            for prompt in np.unique(gather_prompts):
                prompt_mask = gather_prompts == prompt
                prompt2min_init_reward[prompt] = min(gather_init_rewards[prompt_mask].tolist())

            for i, is_bad in enumerate(mask):
                if is_bad:
                    prompt = sample['prompts'][i]
                    sample['rewards']['avg'][i] = prompt2min_init_reward[prompt]

        # 4. Resolve text format rewards (submitted async in rollout via text_reward_fn)
        if 'text_rewards_future' in sample:
            text_rewards_future = sample.pop('text_rewards_future')
            text_rewards, _ = text_rewards_future.result()
            text_rewards = {k: torch.as_tensor(v, device=accelerator.device).float() for k, v in text_rewards.items()}

            if 'think_tag_format' in text_rewards:
                sample['rewards']['think_text_reward'] = text_rewards['think_tag_format']
            if 'reflective_think_format' in text_rewards:
                # Shift from [0,1] to [-1,0] to match think_tag_format range
                sample['rewards']['reflective_think_text_reward'] = text_rewards['reflective_think_format'] - 1

        # 5. Combine format rewards
        think_text_reward = sample['rewards'].get('think_text_reward', torch.zeros_like(sample['rewards']['avg']))
        reflective_reward = sample['rewards'].get('reflective_think_text_reward', torch.zeros_like(sample['rewards']['avg']))
        sample['rewards']['text_avg'] = sample['rewards']['avg'] + (
            reflective_reward + think_text_reward
        ) / 2 * config.train.think_text_format_reward_weight

        # 6. Improvement check
        gathered_rewards = {k: gather(v) for k, v in sample["rewards"].items()}
        gathered_rewards = {k: v.cpu().numpy() for k, v in gathered_rewards.items()}

        if torch.all(torch.tensor(gathered_rewards['image_avg']) <= torch.tensor(gathered_rewards['init_reward'])):
            if accelerator.is_local_main_process:
                print(f"Skipping rollout for step {ctx.global_step} - no improvement in this step")
            return sample, True

        # 7. Shared advantage computation
        return compute_advantages(config, sample, stat_tracker, ctx, prompts_mean=prompts_mean)

    def eval(self, test_dataloader, config, global_step, autocast, **kwargs):
        """Reflection evaluation: generate -> reflect -> refine."""

        all_rewards = []
        all_rewards_refined = []
        all_text_rewards = []
        all_think_texts = []
        all_prompts = []
        all_images = []
        all_images_refined = []

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
                    images, _ = inferencer.batch_generate_images(
                        input_lists=[[p] for p in prompts],
                        image_shape=(config.resolution, config.resolution),
                        cfg_text_scale=config.sample.cfg_text_scale,
                        cfg_img_scale=1.0,
                        num_timesteps=config.sample.eval_num_steps,
                        cfg_interval=config.sample.cfg_interval,
                        timestep_shift=config.sample.timestep_shift,
                        cfg_renorm_min=config.sample.cfg_renorm_min,
                        cfg_renorm_type=config.sample.cfg_renorm_type,
                        deterministic=True,
                    )
                else:
                    prompts, metadatas, images = batch
                    images = [img.resize((config.resolution, config.resolution)) for img in images]

                rewards = self.compute_rewards(
                    dict(images=images, prompts=prompts, metadata=metadatas),
                    return_future=config.train.async_reward_fn)

                input_lists = [[init_img, reflection_and_regenerate_prompt_with_caption.format(p)]
                               for p, init_img in zip(prompts, images)]

                images_refined, think_texts = inferencer.batch_generate_images(
                    input_lists=input_lists,
                    image_shape=(config.resolution, config.resolution),
                    cfg_text_scale=config.sample.cfg_text_scale,
                    cfg_img_scale=config.sample.cfg_image_scale,
                    num_timesteps=config.sample.eval_num_steps,
                    cfg_interval=config.sample.cfg_interval_edit,
                    timestep_shift=config.sample.timestep_shift,
                    cfg_renorm_min=config.sample.cfg_renorm_min,
                    cfg_renorm_type=config.sample.cfg_renorm_type_edit,
                    deterministic=True,
                    think=True,
                    text_temperature=config.sample.eval_think_text_temperature,
                    text_top_p=config.sample.eval_think_text_top_p,
                    text_top_k=config.sample.eval_think_text_top_k,
                )

                rewards_refined = self.compute_rewards(
                    dict(images=images_refined, prompts=prompts, metadata=metadatas),
                    return_future=config.train.async_reward_fn)

                if not config.train.async_reward_fn:
                    rewards, _ = rewards
                    rewards_refined, _ = rewards_refined

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
                all_rewards_refined.append(rewards_refined)
                all_prompts.append(prompts)
                all_images.append(images)
                all_images_refined.append(images_refined)

        if config.train.async_reward_fn:
            all_rewards_future = all_rewards
            all_rewards = []
            for rewards_future in all_rewards_future:
                rewards, _ = rewards_future.result()
                rewards = {k: torch.as_tensor(v, device=accelerator.device).float() for k, v in rewards.items()}
                all_rewards.append(rewards)

            all_rewards_refined_future = all_rewards_refined
            all_rewards_refined = []
            for rewards_future in all_rewards_refined_future:
                rewards, _ = rewards_future.result()
                rewards = {k: torch.as_tensor(v, device=accelerator.device).float() for k, v in rewards.items()}
                all_rewards_refined.append(rewards)

            all_text_rewards_future = all_text_rewards
            all_text_rewards = []
            for text_rewards_future in all_text_rewards_future:
                text_rewards, _ = text_rewards_future.result()
                text_rewards = {k: torch.as_tensor(v, device=accelerator.device).float() for k, v in text_rewards.items()}
                all_text_rewards.append(text_rewards)

        for rewards, rewards_refined, text_rewards in zip(all_rewards, all_rewards_refined, all_text_rewards):
            rewards["think_text_format_reward"] = text_rewards['avg']
            rewards.update({f"refined_{key}": value for key, value in rewards_refined.items()})

        log_images_to_tracker(
            images=concat_images_list(images, images_refined),
            prompts=prompts, rewards=rewards,
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
                wandb.log({"eval/think_text_length": aggregated_think_text_length}, step=global_step)

                columns = ["Prompt", "Generated image", "Refined image", "Think text", "Reward", "Reward_refined"]
                all_think_texts = all_think_texts if len(all_think_texts) else [[''] * len(all_images[-1]) for _ in range(len(all_images))]
                my_table = wandb.Table(columns=columns)
                for idx in range(2):
                    for img, img_refined, prompt, think_text, reward, reward_refined in zip(
                            all_images[-idx], all_images_refined[-idx],
                            all_prompts[-idx], all_think_texts[-idx],
                            all_rewards[-idx]['avg'], all_rewards[-idx]['refined_avg']):
                        my_table.add_data(prompt, wandb.Image(img), wandb.Image(img_refined), think_text, reward, reward_refined)
                wandb.log({f"eval/step_{global_step}": my_table}, step=global_step)
            else:
                accelerator.log({f"eval/{key}": value.mean().item() for key, value in aggregated_rewards.items()}, step=global_step)
                accelerator.log({"eval/think_text_length": aggregated_think_text_length}, step=global_step)

        inferencer.model = model
        inferencer.model.train()

        if accelerator.is_main_process:
            print('Evaluation results: t2i', {k: v.mean().item() for k, v in aggregated_rewards.items()})
            print('think_text_length', aggregated_think_text_length)
            print('End of evaluation.')
        return aggregated_rewards