# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Mixed task: dispatches to sub-tasks based on the task tag in batch_data.

When using ``mix_dataset`` config, the DataLoader yields batches of the form
``(task_name, prompts, metadatas[, images])``.  MixedTask strips the task_name
prefix and dispatches to the corresponding registered sub-task.
"""

from tasks import register, BaseTask, get_task


@register('mixed')
class MixedTask(BaseTask):

    def __init__(self, **kwargs):
        self._task_kwargs = kwargs
        self._sub_tasks = {}

    def _get_sub_task(self, task_name):
        if task_name not in self._sub_tasks:
            SubTaskClass = get_task(task_name)
            self._sub_tasks[task_name] = SubTaskClass(**self._task_kwargs)
        return self._sub_tasks[task_name]

    def rollout(self, config, tokenizer, batch_data,
                global_step, stat_tracker, **kwargs):
        # batch_data = (task_name, prompts, metadatas[, images])
        task_name = batch_data[0]
        inner_batch_data = batch_data[1:]  # (prompts, metadatas) or (prompts, metadatas, images)

        sub_task = self._get_sub_task(task_name)

        # Temporarily set config.train.task so that sub-task internals
        # (which read config.train.task) see the correct task name.
        original_task = config.train.task
        config.train.task = task_name
        try:
            result = sub_task.rollout(
                config=config, tokenizer=tokenizer,
                batch_data=inner_batch_data,
                global_step=global_step, stat_tracker=stat_tracker,
                **kwargs,
            )
        finally:
            config.train.task = original_task

        return result

    def collect_metrics(self, sample, info):
        """Dispatch to sub-task's collect_metrics."""
        sub_task = self._get_sub_task(sample['task'])
        sub_task.collect_metrics(sample, info)

    def log_training_step(self, train_samples, accelerator, global_step, config, prefix=""):
        """Group samples by task and log each task separately."""
        from collections import defaultdict
        grouped = defaultdict(list)
        for sample in train_samples:
            grouped[sample['task']].append(sample)
        for task_name, samples in grouped.items():
            sub_task = self._get_sub_task(task_name)
            sub_task.log_training_step(samples, accelerator, global_step, config, prefix=task_name)

    def eval(self, test_dataloader, config, global_step, autocast, **kwargs):
        # Dispatch eval to the sub-task specified by config.train.eval_task,
        # defaulting to the first mix_dataset entry's task.
        eval_task_name = getattr(config.train, 'eval_task', None)
        if eval_task_name is None and getattr(config, 'mix_dataset', None):
            eval_task_name = config.mix_dataset[0]['task']
        if eval_task_name is None:
            from accelerate.logging import get_logger
            logger = get_logger(__name__)
            logger.info("MixedTask: config.train.eval_task not set. Skipping eval.")
            return {}

        sub_task = self._get_sub_task(eval_task_name)

        original_task = config.train.task
        config.train.task = eval_task_name
        try:
            result = sub_task.eval(
                test_dataloader, config, global_step, autocast, **kwargs,
            )
        finally:
            config.train.task = original_task

        return result