---
sidebar_position: 0
title: 🧠 基础知识
---

# 🧠 基础知识

> Transformer / Diffusion LM / GPU-CUDA / 分布式训练基础

## 板块导航

### [🏗️ Transformer](./transformer/index.md)

现代大模型的骨架。Pre-Norm Block、SwiGLU FFN、RoPE、Softmax + Flash Attention 等基础模块的图解式讲解。

### [🌊 Diffusion LM](./diffusion-lm/index.md)

自回归以外的另一条大模型路线。从 DDPM 到 LLaDA 的完整脉络：
- [扩散模型基础](./diffusion-lm/diffusion-basics.md) — DDPM / iDDPM / DDIM / D3PM / MDLM
- [LLaDA 方法解析](./diffusion-lm/llada-method.md) — 训练、推理、三种 remask 策略、SFT
- [LLaDA 训练+推理流程图](./diffusion-lm/llada-flow.md) — 可视化版
- [dLLM 的 AI Infra 改进方向](./diffusion-lm/dllm-infra.md) — KV Cache、Kernel、量化、Serving

### 🚧 GPU-CUDA · 分布式训练

板块还在建设中。回来看看吧！

## 阅读建议

如果你在系统学习 AI Infra，推荐顺序：

1. **Transformer 基础**（Pre-Norm → SwiGLU → RoPE → Flash Attention）
2. **量化基础**（如果关心推理侧）
3. **Diffusion LM**（了解自回归以外的另一条路线，为未来 3-5 年的模型演进做准备）

---

*这个花园会持续更新。如果你有想看的话题，欢迎补充。*
