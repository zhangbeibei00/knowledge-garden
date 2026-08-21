---
sidebar_position: 0
title: Transformer
---

# Transformer

现代大模型的骨架。这里的笔记聚焦"图解式"理解 —— 张量在每一步的形状、每个模块在做什么、以及为什么这么设计。

## 目录

- [Pre-Norm Block 逐步拆解](./pre-norm-block.md) — 一个 Transformer Block 里 LayerNorm、Attention、残差是怎么串起来的
- [SwiGLU FFN 图解](./swiglu-ffn.md) — LLaMA 系列的门控 FFN，为什么用三个 Linear、为什么能提点
- [RoPE 旋转位置编码 · 一篇讲透](./rope.md) — 数学原理、Attention 中的位置、cos/sin 与 KV cache 分工、Infra 视角

## 阅读顺序建议

如果你在补 Transformer 的基础，推荐顺序：

1. 先看 **Pre-Norm Block** —— 建立一个 Block 内部的整体框架
2. 再看 **SwiGLU FFN** —— 深入 Block 里"FFN"这一步到底做了什么
3. 然后看 **RoPE** —— 补上 Attention 里位置信息是怎么注入的
4. 结合[量化基础](../../quantization/basics/index.md) —— 理解为什么 FFN、QKV 投影是量化的主战场

---

## 📝 章节总结：Decoder Block 与 AI Infra 的对应关系

回顾 Transformer Decoder Block 的完整结构，以及每个模块与 AI Infra 后续学习的关联：

{/* 原图路径 /AIInfraGuide/images/decoder-blocks.png 未迁移到知识花园，暂用文字概览代替 */}

```
                ┌─────────── Decoder Block ───────────┐
     x  ───┬──→ │ RMSNorm → Multi-Head Self-Attention │──→ + ──┬──→
           │    │           (含 RoPE 旋转 Q/K)         │        │
           └────┼───────────────────────────────────────────────┘
                │                                              │
                ├──→ RMSNorm → FFN (SwiGLU) ──→ + ─────────────┘
                │                                              │
                └────────────────────────────────────→ 下一层 ─┘
```

| 模块 | 核心计算 | 后续 AI Infra 关联 |
|------|---------|-------------------|
| **Self-Attention** | $QK^T$、Softmax、$PV$ | FlashAttention（CUDA 优化）、KV Cache（推理）、张量并行沿头切分 |
| **Multi-Head** | 多头独立计算再拼接 | 张量并行（TP）的切分点、GQA/MQA（推理优化） |
| **FFN (SwiGLU)** | 三次大矩阵乘法 | 参数量大头、张量并行的另一个切分点、MoE 专家并行 |
| **LayerNorm** | 均值/方差归一化 | Kernel 融合优化、RMSNorm 简化 |
| **残差连接** | 逐元素加法 | 与 LayerNorm 融合、梯度流分析 |
| **位置编码 (RoPE)** | 旋转 Q/K 向量 | 融合到 Attention kernel、长上下文扩展 |
| **KV Cache** | 缓存历史 K、V | PagedAttention、KV 量化、Prefix Cache |
| **自回归生成** | Prefill + Decode | Prefill/Decode 解耦、Speculative Decoding |

**核心视角**：学习 Transformer 架构不是目的，而是起点。理解了"优化对象长什么样"之后，你会发现后续的每一项 AI Infra 技术都不再是空中楼阁 —— FlashAttention 在优化 Self-Attention 的 $O(N^2)$ 显存问题，张量并行在切分 Multi-Head 的头结构和 FFN 的矩阵，KV Cache 管理在解决自回归生成的显存开销问题。

---

## 🎯 自我检验清单

完成本章节学习后，检验自己是否真正理解了 Transformer 架构：

- [ ] 能不看资料，在白板上画出一个完整的 Decoder Block 结构图（Masked Self-Attention → Add & Norm → FFN → Add & Norm），标注每一步的输入输出维度
- [ ] 能说清 Encoder-Decoder、Encoder-only、Decoder-only 三种架构变体的区别，以及为什么当前大模型普遍采用 Decoder-only
- [ ] 能说清 Q、K、V 三个矩阵各自的含义，以及 Attention 分数矩阵 $(N, N)$ 中每个元素的物理意义
- [ ] 能默写 Attention 完整公式 $\text{softmax}(QK^T / \sqrt{d_k}) \cdot V$，并解释为什么要除以 $\sqrt{d_k}$
- [ ] 能推导 Self-Attention 的 $O(N^2)$ 复杂度，并解释这如何催生了 FlashAttention
- [ ] 能解释 Multi-Head Attention 为什么适合张量并行切分，以及 GQA 相比 MHA 在 KV Cache 上的优势
- [ ] 能手算 LLaMA-2-7B 的总参数量（误差不超过 20%），并说清 FFN 和 Attention 的参数比例
- [ ] 能解释 Prefill 和 Decode 两阶段的计算特性差异（Compute Bound vs Memory Bound），以及 KV Cache 的由来
- [ ] 能估算给定配置下 KV Cache 的显存占用（如 7B 模型、4096 序列长度、batch_size=16 下约 32 GB）

---

## 相关

- 深入 Block 内部：[Pre-Norm Block](./pre-norm-block.md) · [SwiGLU FFN](./swiglu-ffn.md) · [RoPE](./rope.md)
- 推理相关：[推理引擎](../../inference-engines/index.md)（Prefill/Decode、KV Cache 管理、FlashAttention 等）
- 量化视角：[量化基础](../../quantization/basics/index.md)（Norm/QKV/FFN 的量化关注点）
