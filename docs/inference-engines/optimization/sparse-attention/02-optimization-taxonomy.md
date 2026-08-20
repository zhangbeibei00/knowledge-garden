---
sidebar_position: 2
title: 六大优化流派全景
---

# 🌐 六大优化流派全景

## 🎯 面试速记版

> $O(n^2 d)$ 的 attention 瓶颈催生了六条优化路线：
>
> 1. **稀疏**：只算重要的 pair —— Longformer / BigBird
> 2. **低秩**：K, V 投到低维 —— Linformer / Performer
> 3. **核方法 / SSM**：抛弃 softmax，走线性形式 —— Mamba / RWKV
> 4. **IO 感知**：$O(n^2)$ 不变，避免落 HBM —— **FlashAttention 1/2/3 🌟 工业标配**
> 5. **KV Cache 压缩**：推理专项 —— GQA / MLA 🌟 工业标配
> 6. **分块并行**：切块跨卡 —— Ring Attention（Gemini 1.5 10M 上下文）
>
> **胜负手**：稀疏 / 低秩 / 核方法在 GPU 上打不过 FlashAttention。真正统治工业界的是 **FA + GQA/MLA + Ring**。

## 1. 六大流派对比表

| 流派 | 代表方法 | 时间 | 显存 | 精度 | 工业采用 |
|------|---------|-----|------|------|---------|
| 原始 | Standard Attention | $O(n^2 d)$ | $O(n^2)$ | ✅ | - |
| **稀疏** | Longformer / BigBird | $O(nk)$ | $O(nk)$ | 略降 | 🔻 淡出 |
| **低秩** | Linformer | $O(nk)$ | $O(nk)$ | 中 | 🔻 淡出 |
| **核方法** | Performer / Linear TF | $O(nd^2)$ | $O(nd)$ | 中 | 🟡 研究 |
| **SSM** | Mamba / RWKV | $O(n)$ 训 / $O(1)$ 推 | $O(1)$ 推 | 中高 | 🟢 混合架构 |
| **IO 感知** | FlashAttention 1/2/3 | $O(n^2 d)$ | $O(n)$ | ✅ 精确 | 🌟 标配 |
| **KV 压缩** | GQA / MLA | $O(n^2 d)$ | KV 大幅降 | ✅ | 🌟 标配 |
| **分块并行** | Ring Attention | 分布式 | 分布式 | ✅ | 🟢 长上下文 |

## 2. 稀疏派为什么会式微？

**表面原因**：GPU 上稀疏 pattern 不规则 → tensor core 用不上 → 稀疏了反而更慢。

**根本原因**：**FlashAttention 把标准 attention 的常数打得太低**——$O(n^2)$ 看似吓人，但 IO 优化后实际墙推到 128k，够用。

**近期反转**：**Native Sparse Attention（DeepSeek 2025）** 证明 block-level 稀疏 + FA 融合可以打赢稠密 FA。稀疏派没死，只是必须走 block-level 路线。

## 3. 关键时间线

```
2017 ─ Transformer 诞生，O(n²d) 瓶颈
2019 ─ Sparse Transformer (OpenAI)         ← 稀疏派开山
2020 ─ Longformer / BigBird                ← 稀疏派高峰
2020 ─ Linformer / Linear Transformer      ← 低秩 / 核方法
2021 ─ S4                                  ← SSM 起源
2022 ─ 🌟 FlashAttention                    ← 改变游戏规则
2023 ─ Mamba / RWKV / GQA                  ← RNN 复兴 + KV 压缩
2024 ─ FA3 / MLA / Ring Attention          ← 工程巅峰
2025 ─ Native Sparse Attention             ← 稀疏派回归
```

## 4. 关键洞察

**"精确 + IO 优化" > "近似 + 忽略 IO"** —— 这是过去 5 年 attention 优化最深刻的一课。

FlashAttention 证明了：**GPU 时代，算法优化必须考虑硬件层级**。$O(n^2)$ 只是理论墙，真实墙是显存带宽。

## 相关文档

- 前一篇：[Attention 复杂度痛点](/docs/inference-engines/optimization/sparse-attention/attention-complexity)
- 下一步：[稀疏 Attention 的公式表达](/docs/inference-engines/optimization/sparse-attention/sparse-basics)
