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
