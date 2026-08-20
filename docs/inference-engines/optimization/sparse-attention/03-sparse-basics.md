---
sidebar_position: 3
title: 稀疏 Attention 的公式表达
---

# 📐 稀疏 Attention 的公式表达

## 🎯 面试速记版

> **两种等价写法**：
> - **分段版**：定义每个 token 的注意力集合 $S_i$，只对 $j \in S_i$ 参与 softmax
> - **Mask 版**：给 $QK^T$ 加一个 mask 矩阵 $M$，$M_{ij} \in \{0, -\infty\}$，softmax 后 $-\infty$ 位置自然变 0
>
> **关键**：mask 写法只是**数学等价表达**——如果先算完整 $QK^T$ 再加 mask，复杂度还是 $O(n^2)$，一点没省！要真加速必须**只对 mask 中非 $-\infty$ 的位置做计算**，这就需要定制 kernel。

## 1. 标准 self-attention

$$
\alpha_{ij} = \frac{\exp(q_i k_j^T / \sqrt{d})}{\sum_{l=1}^{n} \exp(q_i k_l^T / \sqrt{d})}, \quad o_i = \sum_{j=1}^{n} \alpha_{ij} v_j
$$

分母的 $\sum_{l=1}^{n}$ 就是 $O(n)$ 的来源；对所有 $i$ 都做一遍就是 $O(n^2)$。

## 2. 稀疏版：注意力集合 $S_i$

定义每个 token $i$ 只关注 $S_i \subseteq \{1, ..., n\}$：

$$
\alpha_{ij} = \begin{cases}
\dfrac{\exp(q_i k_j^T / \sqrt{d})}{\sum_{l \in S_i} \exp(q_i k_l^T / \sqrt{d})} & j \in S_i \\
0 & j \notin S_i
\end{cases}
$$

若 $|S_i| = k$，总复杂度降到 $O(nk)$。

## 3. Mask 矩阵表达（工程常用）

引入 mask $M$：

$$
M_{ij} = \begin{cases} 0 & j \in S_i \\ -\infty & j \notin S_i \end{cases}
$$

则

$$
\boxed{\text{SparseAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + M\right) V}
$$

**为什么 $-\infty$ 能屏蔽**：softmax 里 $\exp(-\infty) = 0$，从分子分母都消失。

## 4. ⚠️ 逻辑稀疏 ≠ 物理稀疏

```python
# 朴素实现：数值对，但复杂度没降
scores = Q @ K.T / sqrt(d)      # (n, n) ← 还是 O(n²)
scores = scores + mask           # (n, n)
attn = softmax(scores) @ V       # (n, n) @ (n, d)
```

**问题**：$QK^T$ 完整算了，$O(n^2)$ 一点没省。

**真加速要**：
- 用 gather 只取 $S_i$ 对应的 $k_j, v_j$
- 或用 block-sparse kernel 跳过整块

详见 [06. 是否需要定制算子](/docs/inference-engines/optimization/sparse-attention/custom-kernels)。

## 5. 一个特例：Causal Mask

GPT 系列的因果 mask 本质就是稀疏 attention：

$$
S_i = \{1, 2, ..., i\}, \quad M_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}
$$

**Mask 形状**：下三角矩阵，平均每 token 只 attend 一半 → 复杂度还是 $O(n^2/2) = O(n^2)$，只是常数减半。

**FlashAttention 对 causal mask 有专门优化**：跳过 mask 全为 $-\infty$ 的 tile，实测比稠密快约 2×。

## 相关文档

- 前一篇：[六大优化流派全景](/docs/inference-engines/optimization/sparse-attention/optimization-taxonomy)
- 下一步：[稀疏模式全家福](/docs/inference-engines/optimization/sparse-attention/sparse-patterns)
