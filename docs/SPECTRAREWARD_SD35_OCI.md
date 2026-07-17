# SpectraReward SD3.5-M reproduction on OCI

This runner reproduces the paper's 512 px SD3.5-M experiment with
Qwen3-VL-8B-Instruct and Advantage-Weighted Matching (AWM).

## Exact training shape

- Policy: SD3.5-M LoRA on node 0 (8 A100 GPUs).
- Reward: Qwen3-VL-8B-Instruct with SGLang DP=8 on node 1.
- Rollout: 16 denoising steps, CFG 4, timestep shift 3, 512 x 512.
- Training forward: conditional field without a second CFG forward, following
  the official SD3 AWM `no_cfg` setup.
- Group size: 16 images per prompt.
- Per-policy-rank rollout batch: 2 images.
- 32 rollout batches per update: 32 prompts and 512 images globally.
- AWM: 6 sampled training timesteps from the first 10 denoising steps.
- Optimizer: learning rate 1e-4, gradient accumulation 32 on 8 policy GPUs.
- Stop: 380 optimizer steps; checkpoints are resumable Accelerate states.

The paper used 32 policy GPUs, batch size 2, and gradient accumulation 8. The
8-GPU policy layout uses accumulation 32 so each optimizer update still sees
32 prompts and 512 generated images.

## Launch

Install the pinned requirements in the cluster environment before allocating
GPUs. The Slurm launcher does not install packages on compute nodes.
By default it uses `/home/hcai/workspace/anaconda3/envs/alpha_grpo`; override
this with `ALPHAGRPO_ENV` when needed.
Do not install the public FlashAttention 2.8.3 wheel on OCI A100 nodes: it
requires glibc 2.32. This runner uses PyTorch SDPA and SGLang kernels instead.

```bash
cd /home/hcai/workspace/code/junyiwu/AlphaGRPO
mkdir -p experiments/spectrareward_sd35
sbatch scripts/oci_spectrareward_sd35.sbatch
```

The `interactive` partition sends `SIGUSR1` ten minutes before its four-hour
limit. The trainer finishes the current epoch, writes a checkpoint, and exits.
Submitting the same script again automatically resumes the newest checkpoint.

For a one-step smoke run, export `MAX_TRAIN_STEPS=1` before `sbatch`. To test
only the reward protocol after starting the server:

```bash
PYTHONPATH=alpha_grpo python scripts/smoke_spectrareward_client.py \
  --url http://REWARD_NODE:18090 \
  --model Qwen/Qwen3-VL-8B-Instruct
```

## Provenance and known uncertainty

The SD3 AWM implementation under `third_party/advantage_weighted_matching` is
derived from the official Apache-2.0 AWM release. SpectraReward publishes the
rollout, optimizer, hardware, and timestep settings above but says that other
hyperparameters follow official AWM. Consequently `beta=0.001`, gHuber
weighting, EMA-KL, and LoRA target modules follow the official SD3.5-M GenEval
configuration and should be recorded as inferred settings, not paper-specified
values.

## Paper benchmark generation

The SD3.5 runner has a paired, distributed benchmark generator for GenEval,
TIIF-Bench, DPG-Bench, GenEval2, and WISE. Baseline and RL use identical seeds,
512 px resolution, 16 SA-Solver steps, and CFG 4. Existing images are skipped,
so a four-hour job can be resubmitted safely. Generation starts at batch size
48 per GPU and halves the local batch automatically if CUDA runs out of memory.

```bash
mkdir -p experiments/spectrareward_sd35_benchmarks

VARIANT=baseline \
  sbatch --export=ALL,VARIANT scripts/oci_eval_spectrareward_sd35.sbatch

VARIANT=spectrareward_step380 \
LORA_PATH=$PWD/experiments/spectrareward_sd35/training/checkpoints/checkpoint-380/lora_ema \
  sbatch --export=ALL,VARIANT,LORA_PATH scripts/oci_eval_spectrareward_sd35.sbatch
```

The generated directory layouts are compatible with the benchmark scorers in
`Bagel/eval/gen`. TIIF-Bench and WISE require an OpenAI-compatible judge API;
image generation does not require those credentials.

On Draco, where the interactive QOS may allow one job but two nodes, launch the
paired job. It runs baseline and the step-380 EMA LoRA concurrently on separate
8-GPU nodes:

```bash
sbatch scripts/oci_eval_spectrareward_sd35_pair.sbatch
```
