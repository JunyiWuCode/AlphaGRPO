# Extending AlphaGRPO

All core components use a **registry pattern** — add a new file, decorate with `@register`, and it's auto-discovered by the training loop.

## Table of Contents

- [Adding a New Task](#adding-a-new-task)
- [Adding a New Algorithm](#adding-a-new-algorithm)
- [Adding a New Reward](#adding-a-new-reward)
- [Adding a New Config](#adding-a-new-config)
- [Adding a New Evaluation](#adding-a-new-evaluation)

---

## Adding a New Task

Create a file in `alpha_grpo/tasks/` and inherit from `BagelBaseTask`:

```python
from tasks import register, BagelBaseTask
from common import GenerationStep

@register('my_task')
class MyTask(BagelBaseTask):

    def rollout(self, config, tokenizer, batch_data, global_step, stat_tracker, **kwargs):
        """Generate samples and build generation_steps.

        Use self.generate_image() / self.generate_text() for generation,
        then build sample['generation_steps'] as a List[GenerationStep].
        """
        prompts, prompt_metadata = batch_data

        # Generate with log probs
        images, latents, image_log_probs, think_texts, text_log_probs, contexts = \
            self.generate_image(prompts, config, return_log_probs=True)

        # Build step list — the training loop iterates these
        steps = []
        if think_texts is not None:
            steps.append(GenerationStep(step_type='think_text', texts=think_texts,
                                        text_per_token_log_probs=text_log_probs))
        steps.append(GenerationStep(step_type='image', latents=latents[:, :-1],
                                    next_latents=latents[:, 1:], images=images,
                                    image_log_probs=image_log_probs, sde_window=...))

        sample = dict(generation_steps=steps, images=images, prompts=prompts,
                      contexts=contexts, ...)
        return sample

    def eval(self, test_dataloader, config, global_step, autocast, **kwargs):
        """Run evaluation on test set."""
        ...

    @classmethod
    def reward_postprocess(cls, config, sample, stat_tracker, ctx=None):
        """Modify rewards if needed, then call compute_advantages()."""
        from common import compute_advantages
        return compute_advantages(config, sample, stat_tracker, ctx)
```

Then import it in `tasks/__init__.py`:
```python
from . import my_task
```

### Step Types

The training loop is **step-list driven** — it iterates `sample['generation_steps']` and dispatches each step by `step_type`. Each type has different loss computation and CFG context advancement rules:

| Step Type | Has Loss | Context Advancement |
|-----------|----------|-------------------|
| `think_text` | Yes | Only `gen_context` advances |
| `text` | Yes | `gen` advances; between-step: `cfg_text += text`, `cfg_img += text` |
| `image` | Yes | Between-step: `gen += image`, `cfg_text = deepcopy(gen)` |
| `input_text` | No | `cfg_text = deepcopy(gen)`, `gen += text`, `cfg_img += text` |
| `input_image` | No | `gen += image`, `cfg_text = deepcopy(gen)` |

**Single-turn task** (e.g., T2I): `[think_text, image]`

**Multi-turn task** (e.g., multi-round editing): `[think_text, image, input_text, think_text, image]`

### Generation API

`BagelBaseTask` provides two generation methods:

```python
# Image generation
# return_log_probs=False → (images, think_texts)
# return_log_probs=True  → (images, latents, image_log_probs, think_texts, text_log_probs, contexts)
self.generate_image(prompts, config, return_log_probs=True)

# Text generation
# return_log_probs=False → (output_texts, think_texts)
# return_log_probs=True  → (output_texts, output_lp, think_texts, think_lp, contexts)
self.generate_text(prompts, config, return_log_probs=True)
```

### Reward Computation

Tasks can compute rewards asynchronously via `self.compute_rewards()`:

```python
sample = self.compute_rewards(sample, config)
```

This submits reward computation to a thread pool. Results are resolved in `reward_postprocess()` before training.

### Existing Tasks for Reference

| Task | File | Steps |
|------|------|-------|
| Text-to-Image | `tasks/t2i.py` | `[think_text, image]` |
| Text-Image-to-Text | `tasks/ti2t.py` | `[think_text, text]` |
| Self-Reflective Refinement | `tasks/reflect.py` | `[think_text, image]` (with reflection prompt) |
| Mixed | `tasks/mixed.py` | Dispatches to sub-tasks per batch |

---

## Adding a New Algorithm

Create a file in `alpha_grpo/algorithms/`:

```python
from algorithms import register, BaseAlgorithm

@register('my_algo')
class MyAlgorithm(BaseAlgorithm):

    @classmethod
    def compute_text_loss(cls, config, sample, accelerator, info,
                         gen_step=None,
                         text_per_token_log_probs=None,
                         text_entropies=None,
                         text_token_lens=None,
                         get_high_entropy_mask=None,
                         **kwargs):
        """Compute RL loss for a text generation step.

        Args:
            gen_step: GenerationStep with rollout data
                - gen_step.text_per_token_log_probs: rollout log probs (or recomputed)
                - gen_step.rollout_text_per_token_log_probs: original rollout log probs (if recomputed)
                - gen_step.ref_text_per_token_log_probs: ref model log probs (for KL)
            sample['advantages']: per-sample advantages
            text_per_token_log_probs: current forward pass log probs

        Returns:
            (info, loss_scalar)
        """
        ...

    @classmethod
    def compute_image_loss(cls, config, sample, accelerator, info,
                          gen_step=None, j=0,
                          image_log_probs=None,
                          prev_latents_mean=None,
                          std_dev_t=None,
                          model_output=None,
                          **kwargs):
        """Compute RL loss for an image generation step at timestep j.

        Args:
            gen_step: GenerationStep with rollout data
                - gen_step.image_log_probs[:, j]: rollout log probs
                - gen_step.ref_prev_latents_means[:, j]: ref model mean (for KL)
                - gen_step.ref_model_output[:, j]: ref model output (for KL)
            j: timestep index within sde_window
            image_log_probs: current forward pass log probs
            sample['advantages'][:, j]: per-timestep advantages

        Returns:
            (info, loss_scalar)
        """
        ...
```

Use via config: `config.train.algorithm = 'my_algo'` (or `config.train.image_algorithm` to use a different algorithm for image steps).

### Existing Algorithms

| Algorithm | File | Description |
|-----------|------|-------------|
| GRPO | `algorithms/grpo.py` | Group Relative Policy Optimization |
| ReMax | `algorithms/remax.py` | ReMax baseline variant |
| AWM | `algorithms/awm.py` | Advantage-Weighted Mixing |

---

## Adding a New Reward

Add a scorer function in `alpha_grpo/rewards/rewards.py`:

```python
def my_reward_scorer(device='cuda', my_param=1.0):
    """Create a reward scorer closure."""
    model = MyRewardModel(device=device)

    def _fn(images, prompts, metadata):
        """
        Args:
            images: list of PIL images, torch.Tensor (NCHW float), or np.ndarray (NHWC uint8)
            prompts: list[str]
            metadata: dict with task-specific data (e.g., input_images for editing)

        Returns:
            (scores, reward_metadata)
            - scores: array-like of floats, one per image
            - reward_metadata: dict (can be empty {})
        """
        scores = model.score(images, prompts)
        return scores, {}

    return _fn
```

Register it in the `multi_score()` function's `score_functions` dict:

```python
score_functions = {
    "my_reward": my_reward_scorer,
    "pickscore": pickscore_score,
    ...
}
```

Reference in config with weights:

```python
config.reward_fn = {"my_reward": 1.0, "pickscore": 0.5}
```

The final reward is a weighted sum: `reward = w1 * score1 + w2 * score2 + ...`

### Remote Reward Servers

For reward models with conflicting dependencies, run them as separate API servers and use a remote scorer:

```python
def my_remote_scorer(api_url='http://localhost:8000'):
    def _fn(images, prompts, metadata):
        # POST images + prompts to API, get scores back
        ...
    return _fn
```

### Existing Reward Models

Local: `aesthetic`, `clip`, `siglip`, `pickscore`, `imagereward`, `hpsv2`, `hpsv3`, `qwenvl`, `ocr`, `viescore`

Remote API: `deqa`, `geneval`, `unified_reward`

---

## Adding a New Config

Add a function in `alpha_grpo/config/bagel.py`:

```python
def my_experiment():
    config = reasoning_t2i()  # inherit from existing config
    config.train.learning_rate = 5e-4
    config.train.task = 'my_task'
    config.train.algorithm = 'my_algo'
    config.reward_fn = {"my_reward": 1.0}
    return config
```

Run with:
```bash
torchrun --nnodes=$NUM_NODES --node_rank=$RANK --nproc_per_node=8 \
  alpha_grpo/train.py --config config/bagel.py:my_experiment
```

### Key Config Fields

```python
# Task & algorithm
config.train.task = 't2i'           # registered task name
config.train.algorithm = 'grpo'     # registered algorithm name
config.train.image_algorithm = None # separate algorithm for image steps (defaults to algorithm)

# Sampling
config.sample.num_steps = 40        # total diffusion steps
config.sample.think = True          # enable think text generation
config.sample.group_size = 14       # GRPO group size G

# Loss
config.train.text_beta = 0.0        # KL coefficient for text
config.train.image_beta = 0.0       # KL coefficient for image
config.train.text_loss_weight = 0.2 # text loss scaling factor

# Rewards
config.reward_fn = {"pickscore": 1.0, "aesthetic": 0.5}
```

---

## Adding a New Evaluation

### T2I Evaluation

1. Create a generation script in `Bagel/eval/gen/`:

```python
# Bagel/eval/gen/gen_images_mp_mybench.py
# Generate images for your benchmark using the model
```

2. Create a wrapper script:

```bash
# Bagel/scripts/eval/run_mybench.sh
python eval/gen/gen_images_mp_mybench.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    ...
```

3. Add to `Bagel/scripts/eval/eval_gen.sh` if it should run by default.

### VLM Evaluation

Follow the pattern in `Bagel/eval/vlm/eval/` — each benchmark has its own subdirectory with eval scripts.

### Existing Benchmarks

**T2I**: GenEval, TIIF-Bench, WISE, DPG-Bench, KRIS-Bench, RISE-Bench

**Editing**: GEdit-Bench, ImgEdit

**VLM**: MME, MMBench, MMMU, MMVet, MathVista, MMVP
