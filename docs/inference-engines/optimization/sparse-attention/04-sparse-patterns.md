---
sidebar_position: 4
title: 稀疏模式全家福
---

# 🎨 稀疏模式全家福

## 🎯 面试速记版

> 不同的稀疏 attention = 不同的 $S_i$ 定义方式。三类：
> - **局部型**：Sliding Window（只看邻居）
> - **组合型**：Longformer（局部 + 全局 token）/ BigBird（局部 + 全局 + 随机）
> - **规则型**：Sparse Transformer Strided / Fixed，Dilated
> - **学习型**：Reformer（LSH）、Native Sparse Attention（学习门控）
>
> **全局 token** 是组合型稀疏的核心——弥补局部 attention 的信息传递能力，几个 `[CLS]`/`[SEP]` 就能连通全图。

## 1. Sliding Window（Longformer 主体）

$$
S_i = \{j : |i - j| \leq w/2\}
$$

Mask 是**带状矩阵**：

```
       j:  0 1 2 3 4 5 6 7
i=0:      ▓ ▓ ▓ . . . . .
i=1:      ▓ ▓ ▓ ▓ . . . .
i=2:      ▓ ▓ ▓ ▓ ▓ . . .
i=3:      . ▓ ▓ ▓ ▓ ▓ . .
```

复杂度：$O(nw)$，$w$ 通常 512

## 2. Longformer（Sliding + Global Token）

$$
S_i = \{j : |i-j| \leq w/2\} \cup \{g_1, ..., g_m\}
$$

且全局 token $g$ 的 $S_g = \{1, ..., n\}$（看所有）。

Mask 是**带状 + 十字线**：

```
       g   1 2 3 4 5
g:    ▓▓▓ ▓ ▓ ▓ ▓ ▓  ← 全局 token 看所有
i=1:  ▓   ▓ ▓ ▓ . .
i=2:  ▓   ▓ ▓ ▓ ▓ .
i=3:  ▓   . ▓ ▓ ▓ ▓
i=4:  ▓   . . ▓ ▓ ▓  ← 所有 token 都看全局
```

复杂度：$O(n(w + m))$

## 3. BigBird（Sliding + Global + Random）

$$
S_i = S_i^{\text{window}} \cup S_i^{\text{global}} \cup S_i^{\text{random}}
$$

Random 部分（每 token 额外看 $r$ 个随机位置）保证图连通性。

复杂度：$O(n(w + m + r))$

## 4. Sparse Transformer（Strided / Fixed）

**Strided**：每 $s$ 步看一个：$S_i^{\text{stride}} = \{j : (i-j) \mod s = 0\}$

**Fixed**：块内 + 块末尾 token（信息汇聚点）

**奇偶层交替使用**这两种 pattern → 组合覆盖长距离。

复杂度：$O(n\sqrt{n})$

## 5. Reformer（LSH Bucketing）

用 **Locality-Sensitive Hashing** 把 $Q, K$ 投到 $b$ 个桶里，只对**同桶**的 token 算 attention：

$$
S_i = \{j : \text{hash}(q_i) = \text{hash}(k_j)\}
$$

复杂度：$O(n \log n)$

**问题**：桶大小不均、hash 冲突、动态桶结构 GPU 不友好 → 实际很慢。

## 6. Native Sparse Attention（DeepSeek 2025）

**动态 block-level 稀疏**：

- 训练一个小 gate 网络评分每个 $B \times B$ 块的重要性
- Top-K 或阈值筛选 → 决定哪些块参与
- 配合 FA 风格 kernel 跳过整块

详见 [07. 稀疏 × 量化](/docs/inference-engines/optimization/sparse-attention/sparse-x-quant)（NSA 是 FP8 + Sparse 的代表）。

## 7. 一张对比表

| 方法 | $S_i$ 定义 | Mask 形状 | 复杂度 |
|------|-----------|----------|--------|
| Standard | 全部 | 全 1 | $O(n^2)$ |
| Sliding Window | $\|i-j\| \leq w/2$ | 带状 | $O(nw)$ |
| Longformer | 局部 + 全局 | 带状 + 十字 | $O(n(w+m))$ |
| BigBird | 局部 + 全局 + 随机 | 带状 + 十字 + 散点 | $O(n(w+m+r))$ |
| Sparse TF Strided | $(i-j) \mod s = 0$ | 稀疏点阵 | $O(n^2/s)$ |
| Reformer (LSH) | 同 hash 桶 | 动态 | $O(n \log n)$ |
| **NSA (2025)** | 学习的 block gate | 稀疏块矩阵 | 动态 |

## 8. 直观类比：社交网络

- **Standard**：全连接图（每人认识所有人）
- **Sliding**：线状社交圈（只认识邻居）
- **Longformer**：邻居 + 几个网红
- **BigBird**：邻居 + 网红 + 随机远方好友（六度分隔）
- **Sparse TF**：邻居 + 每 8 步一个信息中枢
- **Causal**：只认识早出生的人

## 相关文档

- 前一篇：[稀疏 Attention 的公式表达](/docs/inference-engines/optimization/sparse-attention/sparse-basics)
- 下一步：[Block-Level 稀疏](/docs/inference-engines/optimization/sparse-attention/block-level-sparsity)
