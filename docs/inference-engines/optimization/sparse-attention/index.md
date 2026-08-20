---
sidebar_position: 0
title: 稀疏 Attention 总览
---

# 🕸️ 稀疏 Attention

> Self-attention 的 $O(n^2 d)$ 瓶颈里，稀疏派曾是最先出手的流派——却在 FlashAttention 崛起后被"稠密 + IO 优化"打得节节败退，直到 **block-level 稀疏 + FA 融合**才在 2025 年重新回到工业界（DeepSeek NSA）。

## 内容地图

| 文档 | 关键词 | 定位 |
|------|--------|------|
| [01. Attention 复杂度痛点](/docs/inference-engines/optimization/sparse-attention/attention-complexity) | $O(n^2 d)$ / $O(n^2)$ 显存 / KV cache | 明确要攻击的是哪个"$O(n^2)$" |
| [02. 六大优化流派全景](/docs/inference-engines/optimization/sparse-attention/optimization-taxonomy) | 稀疏 / 低秩 / 核方法 / IO 感知 / KV 压缩 / 分块 | 稀疏只是六条路线之一 |
| [03. 稀疏 Attention 的公式表达](/docs/inference-engines/optimization/sparse-attention/sparse-basics) | $S_i$ / mask 矩阵 / $-\infty$ | 数学本质 |
| [04. 稀疏模式全家福](/docs/inference-engines/optimization/sparse-attention/sparse-patterns) | Sliding / Longformer / BigBird / Reformer | 各种 $S_i$ 的定义 |
| [05. Block-Level 稀疏](/docs/inference-engines/optimization/sparse-attention/block-level-sparsity) | Block vs Element / Tensor Core | GPU 上稀疏的唯一姿势 |
| [06. 是否需要定制算子](/docs/inference-engines/optimization/sparse-attention/custom-kernels) | Triton / block-sparse kernel | 三档实现方式 |
| [07. 稀疏 × 量化](/docs/inference-engines/optimization/sparse-attention/sparse-x-quant) | FP8 + Sparse / NSA / KV cache 量化 | 组合优化 |

## 学习顺序建议

推荐**顺读 01 → 07**：从"痛点"到"方法全景"，再到"稀疏的公式—模式—硬件—实现—组合"，形成完整闭环。

## 🎯 三个必答问题

**Q1: 稀疏 attention 为什么没能取代标准 attention？**

- **GPU 不喜欢稀疏**：不规则稀疏 pattern 打散了 tensor core 结构
- **FlashAttention 太强**：稠密 + IO 优化把 $O(n^2)$ 常数打到了 128k 都够用
- **LLM 需要长距离精确检索**：稀疏丢的正是这部分

**Q2: 为什么稀疏 attention 必须 block-level？**

- Tensor core 一次运算最小单位是 16×16 或 64×256 的整块
- Element-level 稀疏破坏块结构 → tensor core 用不上 → 稀疏了反而更慢
- Block-level 稀疏 = 块间跳过 + 块内稠密，兼顾稀疏与硬件

**Q3: 量化后再做稀疏 attention，最大风险是什么？**

- **精度纠缠**：BF16 下稀疏和量化正交；FP8/INT8 下两者共享精度余量
- **动态 mask 选错**：低比特下 attention score 精度紧张，top-K 决策会漂移
- **工业最佳实践**：**FP8 + Block Sparse**（DeepSeek NSA 路线），不建议 INT4 + Element Sparse
