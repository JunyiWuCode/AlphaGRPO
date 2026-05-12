<div align="center">

# AlphaGRPO

<!-- Replace the placeholder arXiv ID when the final link is available. -->
[![arXiv](https://img.shields.io/badge/arXiv-2605.xxxxx-b31b1b.svg)](https://arxiv.org/)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://huangrh99.github.io/AlphaGRPO)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
<!-- [![Models](https://img.shields.io/badge/🤗-Models-blue.svg)](https://huggingface.co/) -->
</div> 

This is the official repo for AlphaGRPO: Self-Reflective Multimodal Generation via Decompositional Verifiable Reward.

AlphaGRPO enables multimodal generation RL training across text and image generation for AR-Diffusion-native unified multimodal models, such as [BAGEL](https://github.com/bytedance-seed/BAGEL). It supports training on reasoning text-to-image generation and self-reflective refinement.

This codebase flexibly supports different RL methods for image and text generation: FlowGRPO, DiffusionNFT, and AWM for images; GRPO for text.

<p align="center">
  <img src="assets/figures/png/teaser.png" width="92%" />
</p>

## 📣 Updates

- [2026.05.13]: We released the paper on arXiv!
- [Coming soon]: The training code and model weights are currently undergoing internal review and will be released once approved.


## 🏗️ Overview of Framework

<p align="center">
  <img src="assets/figures/png/method.png" width="92%" />
</p>

## 📊 Main Results

<p align="center">
  <img src="assets/figures/png/main_results_t2i.png" width="85%" />
</p>

> **Text-to-image benchmarks.** RT2I denotes the AlphaGRPO variant trained on the reasoning text-to-image task. Inf. SRR denotes inference-time self-reflective refinement.

<p align="center">
  <img src="assets/figures/png/main_results_gedit.png" width="65%" />
</p>

> **GEdit-Bench-EN.** Editing transfer performance — AlphaGRPO improves GEdit scores without training on editing tasks.

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
