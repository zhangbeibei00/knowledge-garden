---
sidebar_position: 0
title: 推理优化技术
---

# 🚀 推理优化技术

> vLLM / SGLang / TensorRT-LLM 之外，还有一大类"通用推理加速"技术，可以叠加到任何推理引擎上。
> 这个板块梳理这些技术的原理、演进与工程实践。

## 内容地图

| 子模块 | 关键词 | 定位 |
|--------|--------|------|
| [🚀 投机解码（Speculative Decoding）](/docs/inference-engines/optimization/speculative-decoding) | Medusa / EAGLE / MTP / DFlash / DSpark | 让小模型先猜、大模型并行验证，**无损加速** 2-9× |
| [🕸️ 稀疏 Attention（Sparse Attention）](/docs/inference-engines/optimization/sparse-attention) | Longformer / BigBird / Block Sparse / NSA | 破解 $O(n^2 d)$ 的一条路线——必须 block-level 才在 GPU 上 work |

## 🌱 建设中

- KV Cache 压缩（H2O / SnapKV / StreamingLLM）
- 长序列注意力优化（FlashAttention 家族、Ring Attention）
- Continuous Batching / PagedAttention 原理
- Prefill-Decode 分离部署
