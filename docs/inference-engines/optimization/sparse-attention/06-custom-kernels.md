---
sidebar_position: 6
title: 是否需要定制算子
---

# 🔧 是否需要定制算子

## 🎯 面试速记版

> **答案：需要，而且是"必须"级别**。理由：
> - **数学上不需要**：加个 mask 就能表达
> - **工程上必须要**：不定制的话稀疏比稠密还慢，一点加速都没有
> - **必须走 block-level**：element-level 稀疏无法在 GPU 上真加速
> - **现代方案是 IO-aware + sparse 融合**：稀疏必须在 FlashAttention 基础上做 block 跳过（如 Native Sparse Attention），否则打不过稠密 FA

## 1. 三档实现方式

### Level 1：无定制（PyTorch 原生 mask）

```python
scores = Q @ K.T + mask   # 稠密算完再加 mask
```

- 完整算 $QK^T$，$O(n^2)$ 一点没省
- **实测比稠密还慢**（多了 mask 加法）
- 只适合教学/验证正确性

### Level 2：半定制（PyTorch SDPA / xformers）

```python
from torch.nn.functional import scaled_dot_product_attention
out = scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=True)
```

- **Causal mask**：PyTorch 会检测并调 FlashAttention kernel（真快）
- **自定义 mask**：大概率 fallback 到稠密实现
- xformers 支持 `BlockDiagonalMask` / `LowerTriangularMask` 等**结构化** mask

**适用**：因果 mask、块对角、简单窗口
**局限**：任意 pattern 效果不好

### Level 3：完全定制（Triton / CUDA）🌟

**工业级稀疏 attention 的唯一出路**。

**核心思想**：块级 mask + 跳过整块 + 块内稠密 matmul。

```python
# 伪代码（Triton 风格）
@triton.jit
def block_sparse_attn_kernel(Q, K, V, Out, block_mask, ...):
    q_block_idx = tl.program_id(0)
    q_block = load_q_block(q_block_idx)
    
    # online softmax 状态
    m_i, l_i, o_i = -inf, 0, 0
    
    for kv_block_idx in range(num_kv_blocks):
        if block_mask[q_block_idx, kv_block_idx] == 0:
            continue  # ← 稀疏加速的唯一来源
        
        k_block = load_k_block(kv_block_idx)
        v_block = load_v_block(kv_block_idx)
        
        # 块内稠密 attention（用 tensor core）
        scores = tl.dot(q_block, k_block.T)
        # ... FA 式 online softmax 更新
    
    store(Out, o_i / l_i)
```

**关键几点**：
1. 外层循环粒度是 block
2. `continue` 跳过整个块的加载和计算
3. 块内还是稠密 matmul，用 tensor core
4. 在线 softmax 保证显存高效

## 2. GPU 为什么不擅长稀疏

三大问题（同 [05. Block-Level](/docs/inference-engines/optimization/sparse-attention/block-level-sparsity)）：

| 问题 | 具体表现 |
|------|---------|
| Warp divergence | `if mask[i,j]` 判断导致 warp 里线程分道 |
| 不连续访问 | Gather 到的位置随机 → 带宽利用率暴跌 |
| Tensor core 用不上 | 稀疏破坏 16×16 块结构 |

## 3. 各方法的定制难度

| 方法 | 定制难度 | 有开源实现 |
|------|---------|-----------|
| Causal mask（GPT） | ⭐（FA 内置） | ✅ 完善 |
| Sliding Window（Mistral） | ⭐⭐（xformers） | ✅ 完善 |
| Longformer | ⭐⭐⭐ | ✅ 官方 |
| BigBird | ⭐⭐⭐⭐ | ⚠️ 官方慢 |
| Sparse Transformer | ⭐⭐⭐ | ✅ OpenAI |
| Reformer (LSH) | ⭐⭐⭐⭐⭐ | ⚠️ 实际很慢 |
| **NSA (2025)** | ⭐⭐⭐⭐（FA + block sparse） | ✅ DeepSeek 开源 |

## 4. 一个反例：为什么"能不定制就不定制"

假设 90% 稀疏度：

- **朴素 PyTorch + mask**：慢 20-30%
- **Gather-based 稀疏实现**：慢 2-3×
- **Block-sparse Triton kernel**：**快 3-5×** ✅
- **稠密 FlashAttention**：**基准** ✅

结论：**没有 block 结构的稀疏，稠密 FlashAttention 反而是最快的**——这就是为什么 GPT 训练不用稀疏，直接用 FA 就够了。

## 5. Native Sparse Attention 为什么重要

它证明了：**只要稀疏做得够 block-friendly，稀疏能打过稠密 FlashAttention**。

- **动态 block-level 稀疏**：训练小 gate 网络动态选块
- **配合 FA 式 kernel**：稀疏和 IO-aware 融合
- **实测比稠密 FA 快 1.5-2×**

## 相关文档

- 前一篇：[Block-Level 稀疏](/docs/inference-engines/optimization/sparse-attention/block-level-sparsity)
- 下一步：[稀疏 × 量化](/docs/inference-engines/optimization/sparse-attention/sparse-x-quant)
