<div align="center">

# AlphaGRPO
### Self-Reflective Multimodal Generation via Decompositional Verifiable Reward

[![arXiv](https://img.shields.io/badge/arXiv-2605.12495-b31b1b.svg)](https://arxiv.org/abs/2605.12495)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://huangrh99.github.io/AlphaGRPO)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-blue.svg)](https://huggingface.co/collections/huangrh9/alphagrpo)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
</div>

This is the official repo for **AlphaGRPO: Self-Reflective Multimodal Generation via Decompositional Verifiable Reward**.

**TL;DR:** AlphaGRPO enables multimodal generation RL training across text and image generation for AR-Diffusion-native unified multimodal models, such as [BAGEL](https://github.com/bytedance-seed/BAGEL). It supports training on reasoning text-to-image generation and self-reflective refinement.

This codebase flexibly supports different RL methods for image and text generation: **FlowGRPO**, **DiffusionNFT**, and **AWM** for images; **GRPO** for text.

<p align="center">
  <img src="assets/figures/png/teaser.png" width="92%" />
</p>

## 📣 News

- **[2026/06/12]** We release **AlphaGRPO**, an RL framework for multimodal generation training on [BAGEL](https://github.com/bytedance-seed/BAGEL). Supporting tasks include reasoning text-to-image generation and self-reflective refinement.
- **[2026/05/13]** We released the paper on [arXiv](https://arxiv.org/abs/2605.12495).

## 🏗️ Overview

AlphaGRPO trains unified multimodal models with a decompositional reward design: complex prompts are broken into verifiable semantic and quality checks, and text/image generation steps can be optimized with separate RL algorithms.

The framework is step-list driven, so each task can explicitly define its own text and image generation sequence. This supports simple reasoning text-to-image generation as well as multi-stage self-reflective refinement without hardcoding a fixed generation pattern.

<p align="center">
  <img src="assets/figures/png/method.png" width="92%" />
</p>

## 🔧 Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Git LFS

The bundled `alphagrpo20k` JSONL dataset files are tracked with Git LFS. Install Git LFS before cloning, or pull LFS files after cloning:

```bash
git lfs install
git lfs pull
```

If the dataset looks unusually small, check that `alpha_grpo/dataset/alphagrpo20k/train.jsonl` is not a Git LFS pointer file.

## 🚀 Quick Start

Download the base [BAGEL](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) model weights before training.

Then update `huggingface_models_root` in `config/bagel.py` to point to `/path/to/huggingface_models/`. The bundled DVReward dataset is under `alpha_grpo/dataset/alphagrpo20k/`; to use your own data, see [`docs/DVREWARD.md`](docs/DVREWARD.md).

Deploy the reward server in a separate terminal on each reward-server node:

```bash
ip=YOUR_IP
port=YOUR_PORT
bash scripts/serve_reward_model.sh $ip $port
```

Single-node training example:

```bash
ip=YOUR_REWARD_SERVER_IP
port=YOUR_REWARD_SERVER_PORT

export PYTHONPATH=$PYTHONPATH:$(pwd)/Bagel/
# IPv6:
export OPENAI_API_URL=http://[${ip}]:${port}/v1
# IPv4:
# export OPENAI_API_URL=http://${ip}:${port}/v1
export OPENAI_MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct

task=alphagrpo_reflect  # alphagrpo_t2iThink is also available
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 \
  alpha_grpo/train.py --config config/bagel.py:${task}
```

Multi-node training example:

```bash
ip=YOUR_REWARD_SERVER_IP
port=YOUR_REWARD_SERVER_PORT

export PYTHONPATH=$PYTHONPATH:$(pwd)/Bagel/
export OPENAI_API_URL=http://[${ip}]:${port}/v1  # IPv6 reward-server address
export OPENAI_MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct
    
# We use 8 nodes × 8 A100: 1 GPU/node for serving reward model, 7 for training.
# Scale gradient_accumulation_steps inversely with GPU count to maintain batch size.
task=alphagrpo_reflect # for self-reflective refinement task
# task=alphagrpo_t2iThink  # for reasoning text-to-image task
torchrun --nnodes=$NUM_NODES --node_rank=$RANK --nproc_per_node=7 \
  alpha_grpo/train.py --config config/bagel.py:${task}
```
 
## 📊 Evaluation

Install eval dependencies:
```bash
bash Bagel/eval_install.sh
```

Run benchmarks:
```bash
cd Bagel
# Evaluate on downstream benchmarks (GenEval, TIIF, WISE, DPG, GEdit)
task=geneval # dpg, tiif, wise, gedit

# Evaluate multiple LoRA checkpoints. Configure `jobname` and `checkpoint_list` in the corresponding script before running.
bash scripts/eval/run_${task}_multickpt.sh

# Evaluate one LoRA checkpoint directly.
export BAGEL_LORA_PATH=/path/to/checkpoint/hf_model/
bash scripts/eval/run_${task}.sh

# Conduct self-reflective refinement on downstream tasks.
task=geneval # dpg, tiif
bash scripts/eval/run_${task}_multickpt_reflect.sh
```

## 📊 Main Results

<p align="center">
  <img src="assets/figures/png/main_results_t2i.png" width="85%" />
</p>

> **Text-to-image benchmarks.** RT2I denotes the AlphaGRPO variant trained on the reasoning text-to-image task. Inf. SRR denotes inference-time self-reflective refinement.

<p align="center">
  <img src="assets/figures/png/main_results_gedit.png" width="65%" />
</p>

> **GEdit-Bench-EN.** Editing transfer performance — AlphaGRPO improves GEdit scores without training on editing tasks.

## 📦 Train on Your Own Dataset

To train AlphaGRPO on a custom prompt set with our **Decompositional Verifiable Reward (DVReward)**, see [`DVREWARD.md`](docs/DVREWARD.md) for the prompt-decomposition pipeline and how to wire the resulting dataset into training.

## 🧩 Extending AlphaGRPO

Text and image steps support independent RL algorithms (e.g., GRPO for text + AWM or DiffusionNFT for image). **New tasks, algorithms, and rewards can be added modularly.** See [docs/extending.md](docs/extending.md) for details.

## 👍 Acknowledgements

This project builds upon [BAGEL](https://github.com/bytedance-seed/BAGEL) and [Flow-GRPO](https://github.com/yifan123/flow_grpo). We thank the authors for their excellent work.



## ✏️ Citation

If you find AlphaGRPO useful to your research, please consider citing:
```bibtex
@inproceedings{huang2026alphagrpo,
  title={AlphaGRPO: Unlocking Self-Reflective Multimodal Generation in Unified Multimodal Models via Decompositional Verifiable Reward},
  author={Huang, Runhui and Wu, Jie and Yang, Rui and Liu, Zhe and Zhao, Hengshuang},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## 🔒 License

This project is released under the [Apache License 2.0](LICENSE).
