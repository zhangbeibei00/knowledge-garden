---
sidebar_position: 5
title: Block-Level 稀疏
---

# 🧱 Block-Level 稀疏

## 🎯 面试速记版

> **Block-level 稀疏** = 稀疏的最小单位不是单个元素，而是一整个 $B \times B$ 块。整块要么保留（内部稠密计算）要么整块跳过。
>
> **为什么必须 block-level**：GPU 的 tensor core 一次运算最小单位就是 16×16 或 64×256 的整块。element-level 稀疏破坏块结构 → tensor core 用不上 → 稀疏了反而更慢。
>
> **Block 大小选 64 是甜点**——够大能吃满 tensor core，够小保留稀疏灵活度。这是 FlashAttention、DeepSpeed Sparse、Native Sparse Attention 的共同选择。

## 1. Element-level vs Block-level 对比

### Element-level（不友好）

每个格子独立决定，pattern 完全不规则：

```
       j: 0 1 2 3 4 5 6 7
i=0:      ▓ . ▓ . . ▓ . ▓
i=1:      . ▓ . ▓ ▓ . ▓ .
i=2:      ▓ . ▓ . . . ▓ .
```

- GPU 头大：warp divergence、内存跳跃、tensor core 用不上

### Block-level（友好）

以 $B=2$ 切成 $B \times B$ 块，整块保留或跳过：

```
       ┌─────┬─────┬─────┬─────┐
       │▓ ▓  │ . .  │▓ ▓  │ . . │
       │▓ ▓  │ . .  │▓ ▓  │ . . │
       ├─────┼─────┼─────┼─────┤
       │ . .  │▓ ▓  │ . .  │▓ ▓  │
       └─────┴─────┴─────┴─────┘

对应 block-mask (4×4)：
       ┌───┬───┬───┬───┐
       │ 1 │ 0 │ 1 │ 0 │
       ├───┼───┼───┼───┤
       │ 0 │ 1 │ 0 │ 1 │
       └───┴───┴───┴───┘
```

- 块内稠密 → tensor core 直接用
- 块间稀疏 → 跳过整块

## 2. GPU 为什么只吃 block-level？

三个硬件层面的原因：

### 2.1 Tensor Core 最小单位

| 架构 | 最小 matmul |
|------|------------|
| Volta (V100) | 16×16×16 |
| Ampere (A100) | 16×16×16 |
| Hopper (H100) | 64×256×16（WGMMA） |

Tensor core 算力是普通 CUDA core 的 **10-30 倍**。element-level 稀疏破坏这个结构 → 速度暴跌 10 倍。

### 2.2 内存合并（Coalescing）

GPU 一次读 128 字节。**block-level** 加载一整块 K, V → 连续读；**element-level** gather 分散位置 → 多次跳跃读。

### 2.3 Warp 一致性

Warp（32 线程）必须做同样的事。**block-level** 全 warp 处理同一块 → 无 divergence；**element-level** 有的算有的跳 → 相当于串行。

## 3. Block 大小怎么选

| Block 大小 | 优点 | 缺点 |
|-----------|------|------|
| 大（如 128） | tensor core 效率高，内存加载大 | 稀疏粒度粗，SRAM 可能超容 |
| 小（如 16） | 稀疏粒度细 | tensor core 用不满 |
| **64（甜点）** | 平衡 | 现代 GPU 上通用最佳 |

**工业实践**：

- FlashAttention: 64 或 128
- DeepSpeed BlockSparse: 16/32/64 可选
- Native Sparse Attention: 64
- Sparse Transformer: 32

## 4. Static vs Dynamic Block-Level

| 类型 | Block-mask 何时定 | 代表 | 优缺点 |
|------|-----------------|------|-------|
| **Static** | 提前定死 | Longformer / Sliding Window | 简单、可预编译 kernel；不灵活 |
| **Dynamic** | 运行时决定 | Reformer / **NSA** | 灵活、精度高；kernel 复杂 |

## 5. Block-Level 的隐藏收益

除了 tensor core 友好，还有：

- **KV Cache 友好**：只加载保留块的 K, V → 稀疏率直接翻译成显存节省
- **可流水线化**：当前块计算 + 下一块 prefetch
- **Mask 存储省**：$n^2$ bit → $(n/B)^2$ bit。$n=128\text{k}, B=64$：2 GB → 512 KB

## 6. 一张图总结

```
                  稀疏 Attention
                       │
        ┌──────────────┴──────────────┐
        │                             │
   Element-level                 Block-level 🌟
   ❌ GPU 不友好                 ✅ tensor core 友好
   ❌ 稀疏反而更慢               ✅ 稀疏才有加速
        │                             │
   理论论文常用                  工业界唯一 work
```

## 相关文档

- 前一篇：[稀疏模式全家福](/docs/inference-engines/optimization/sparse-attention/sparse-patterns)
- 下一步：[是否需要定制算子](/docs/inference-engines/optimization/sparse-attention/custom-kernels)
