---
sidebar_position: 0
title: Diffusion LM · 从 DDPM 到 LLaDA
description: 扩散语言模型的完整脉络 —— 图像扩散基础、离散扩散演进、LLaDA 方法解析、dLLM 的 AI Infra 挑战
---

# Diffusion LM · 从 DDPM 到 LLaDA

主流大模型（GPT / LLaMA / Claude / Qwen）都是 **自回归（AR）** —— 从左到右一个 token 一个 token 吐。
但这不是唯一路线。**扩散语言模型（dLLM）** 走的是一条完全不同的路：**并行填空、双向、可反复修改**。

这个板块把从图像扩散（DDPM）到语言扩散（LLaDA）的完整演进讲透，并从 Infra 角度分析这类模型的机会与挑战。

## 阅读顺序

1. **[扩散模型基础 · 从 DDPM 到离散扩散](./diffusion-basics.md)** — 图像扩散的理论奠基，以及它怎么迁移到离散 token
2. **[LLaDA 方法解析 · 一篇讲透](./llada-method.md)** — 训练目标、推理调度、SFT、和自回归的具体差异
3. **[dLLM 的 AI Infra 改进方向](./dllm-infra.md)** — 为什么 KV Cache 不好用、kernel 怎么改、量化的新战场

---

## 一句话对照

| 生成范式 | 核心思想 | 代表模型 | Attention 方向 | KV Cache |
|---------|---------|---------|---------------|---------|
| **AR** | 逐 token 从左到右预测 | GPT / LLaMA / Qwen | Causal | ✅ 天然适用 |
| **dLLM** | 从全 [MASK] 迭代填空 | LLaDA / Dream / Mercury | Bidirectional | ⚠️ 需改造（GQA + Tied Emb） |

## 演进时间线速览

```
2015  Sohl-Dickstein     热力学扩散思想
2019  NCSN               Score-based 生成 (Yang Song)
2020  ⭐ DDPM             现代扩散模型基石 (Ho et al.)
2020  DDIM               非马尔可夫,加速采样
2020  Score-based SDE    统一 DDPM + NCSN
2021  ⭐ iDDPM            cosine schedule + 可学习方差
2021  Classifier Guide   条件生成起步
2022  CFG                无分类器引导,文生图标配
2022  LDM                隐空间扩散 → Stable Diffusion
2022  ⭐ DiT              Transformer 骨干,DLM 诞生前提
2021  ⭐ D3PM             离散扩散奠基,把 DDPM 搬到 token
2022  Diffusion-LM       连续 embedding 空间做语言扩散
2024  MDLM               掩码扩散化简为 MLM 变体
2025  ⭐ LLaDA (8B)       首次证明 dLLM 追平 LLaMA-3
2025  ⭐ LLaDA 2.0 (100B) 蚂蚁,扩散 LLM 首次百亿规模
2026  LLaDA 2.1          Token Editing → 892 TPS
2026  Mercury            首个商用 dLLM,1000+ TPS
2026  LLaDA 2.2-flash    面向 Agent + 128K 上下文
```

## 你会学到什么

- 为什么"扩散模型"这个从图像来的东西能做语言
- LLaDA 训练和推理的**每一个细节**（从数学公式到伪代码）
- 为什么 dLLM **推理不能直接用 KV Cache**，工程上怎么绕过
- dLLM 对**量化 / kernel / serving** 意味着什么，AngelSlim 类工具链应该怎么跟

## 前置知识

建议先看这些花园里的笔记：

- [Transformer · Pre-Norm Block](../transformer/pre-norm-block.md)
- [Softmax · Online Softmax · Flash Attention](../transformer/softmax-online-flash.md)
- [量化基础](../../quantization/basics/index.md)（如果你对 Infra 视角感兴趣）
