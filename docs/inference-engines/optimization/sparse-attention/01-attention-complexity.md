---
sidebar_position: 1
title: Attention 复杂度痛点
---

# 📐 Attention 复杂度痛点

## 🎯 面试速记版

> **原始公式**：$\text{Attn}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d}) V$，其中 $Q, K, V \in \mathbb{R}^{n \times d}$，$QK^T$ 是 $n \times n$ 矩阵。
>
> **三个不同的 $O(n^2)$ 痛点**：
> - **时间** $O(n^2 d)$：训练 + 推理 prefill 慢
> - **显存** $O(n^2)$：attention matrix 存不下，128k 上下文训不动
> - **KV cache** $O(n d L H)$：推理 decode 时每 token 都要读取历史 K, V
>
> **不同优化针对不同痛点**——搞清楚攻击面才不会用错工具。

## 1. 公式回顾

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V
$$

- $Q, K, V \in \mathbb{R}^{n \times d}$
- $QK^T \in \mathbb{R}^{n \times n}$ ← **$O(n^2)$ 的罪魁**
- 时间：$O(n^2 d)$
- 显存：$O(n^2)$（存 attention matrix）

## 2. 三个痛点对号入座

| 痛点 | 复杂度 | 场景 | 主要卡点 |
|------|-------|------|---------|
| **时间** | $O(n^2 d)$ | 训练 + prefill | 长序列训练慢 |
| **显存**（attention matrix） | $O(n^2)$ | 训练 | 128k 上下文根本训不动 |
| **KV cache** | $O(nd)$ per layer | 推理 decode | 32k 上下文每 batch 几 GB |

## 3. 长上下文的三道墙

- **训练墙**：BF16 下 32k 上下文的 attention matrix ≈ **2 GB / layer**，H100 80G 撑不了几层
- **推理 prefill 墙**：100 万上下文的一次 forward = $10^{12}$ 次乘法，H100 也要几秒
- **推理 decode 墙**：LLaMA-70B 在 128k 上下文时 KV cache ≈ **40 GB**，一个 batch 就吃光单卡

## 4. 六大流派各打一道墙

| 流派 | 攻击的墙 |
|------|---------|
| 稀疏（Sparse） | 时间 + 显存 |
| 低秩（Linformer） | 时间 + 显存 |
| SSM / RNN 化（Mamba） | 时间 + KV cache（推理 $O(1)$）|
| **IO 感知（FlashAttention）** | 显存（$O(n^2) \to O(n)$）|
| **KV 压缩（GQA / MLA）** | KV cache |
| 分块并行（Ring Attention） | 训练时的分布式扩展 |

**关键**：稀疏 attention 主打**训练时间 + attention matrix 显存**，对推理 decode 阶段的 KV cache 帮助有限（除非配合稀疏 gather）。

## 相关文档

- 下一步：[六大优化流派全景](/docs/inference-engines/optimization/sparse-attention/optimization-taxonomy)
