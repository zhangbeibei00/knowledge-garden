---
sidebar_position: 3
title: NVFP4 与 MXFP4
description: 4-bit 浮点的两种设计：NVIDIA 专用推理格式 vs OCP 开放标准
tags: [quantization, nvfp4, mxfp4, fp4, blackwell]
---

# NVFP4 与 MXFP4

> 参考：[NVIDIA 官方博客 — Introducing NVFP4](https://developer.nvidia.cn/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

FP4 只有 4 bit，天然的范围极小（值大约在 -6 到 6 之间，可包括 0.0、0.5、1.0、1.5、2、3、4、6，负范围对称）。**原生 FP4 直接用于量化会精度崩塌**——所以业界搞出了两种"带 scale 管理"的 FP4 变体：**NVFP4**（NVIDIA 专用）和 **MXFP4**（OCP 开放标准）。

## 4-bit 格式家族

![4-bit 格式分类](./assets/fp4-formats.png)

原生 FP4 就是那 8 个可表示值（含正负）。NVFP4 和 MXFP4 都是在原生 FP4 之上加了**分组缩放机制**。

## NVFP4 的两个创新点

1. **高精度缩放因子编码**（High-precision scale encoding）：块 scale 用 **FP8 E4M3** 存储，精度足够
2. **两级微块缩放策略**（A two-level micro-block scaling strategy）：全局 FP32 scale + 每 16 元素的 FP8 块 scale

![NVFP4 两级缩放](./assets/nvfp4-two-level-scale.png)

### 两级缩放的分工

- **Per-tensor 应用 FP32 scale**：把张量的全局 scale 缩放到适合块缩放的尺度（压到 FP8 之内）
- **每 16 值的块应用 FP8 E4M3 scale**：把每个微块缩放到适合 FP4 表示的尺度

## NVFP4 举例：手把手推一遍

拿 16 个权重值当作一个 NVFP4 微块来演示。

**微块（16 个 FP32 权重真实值）**：

```
[ 0.23, -0.18,  0.41, -0.33,
  0.59, -0.62,  0.37, -0.29,
  0.71, -0.85,  0.15, -0.09,
  0.53, -0.47,  0.26, -0.31]
```

这些值都在 -0.85~0.71 之间，需要量化为 NVFP4。

### 步骤 1：计算两级缩放因子

- **全局缩放 `global_scale`**：本例中权重范围小，取 `1.0`（FP32 格式）
- **块缩放 `block_scale`**：针对这 16 个值，NVIDIA 会算一个 FP8 E4M3 格式的缩放因子，目的是把所有值缩到 E2M1 的可表示范围（-6~+6）内。本例中，计算得 `block_scale = 0.125`（FP8 E4M3 格式存储，精确到 0.125）

### 步骤 2：将原始值缩放到 E2M1 范围

每个原始值 ÷ (block_scale × global_scale)：

- 0.23 ÷ 0.125 = 1.84（落在 E2M1 的 1.5~2 之间）
- -0.18 ÷ 0.125 = -1.44（落在 -1.5~-1 之间）
- 0.41 ÷ 0.125 = 3.28（落在 3~4 之间）
- 0.59 ÷ 0.125 = 4.72（落在 4~6 之间）
- 0.71 ÷ 0.125 = 5.68（接近 6）

### 步骤 3：量化为 4-bit E2M1（取最近值）

把缩放后的值映射到 E2M1 的固定值列表 `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` 中最近的那个。

### 步骤 4：推理时反量化（还原值）

用 `E2M1 值 × block_scale × global_scale` 还原：

- 0.23 的量化还原：`2 × 0.125 × 1.0 = 0.25`（接近原 0.23）
- -0.18 的量化还原：`-1.5 × 0.125 × 1.0 = -0.1875`（接近原 -0.18）

### NVFP4 特点总结

- 虽然每个值都有小误差，但 16 个值的**整体分布和大小趋势保留**，模型推理精度几乎不受影响
- 存储成本仅为 FP32 的 **1/8**（4bit vs 32bit）

### 额外存储开销核算

以 100 万权重为例：

| 组件 | 大小 |
|------|------|
| NVFP4 核心数据 | 16 元素/块 × 4bit = 64bit = **8 字节/块** |
| 块缩放因子 | 每微块 1 个 FP8 = 8bit = **1 字节/块** |
| 全局缩放因子 | 整张量 1 个 FP32 = **4 字节**（可忽略） |

额外开销 = 1 字节 / 8 字节 = **12.5%**

这就是 NVFP4 相对纯 FP4 的存储代价——**用 12.5% 换来精度不崩塌**。

## MXFP4：OCP 开放标准

理解了 NVFP4 之后，MXFP4 就简单了。**MXFP4** 是 OCP（Open Compute Project）社区在 2024 年提出的一种 FP4 格式，侧重**通用性**——既支持训练也支持推理。

### NVFP4 vs MXFP4 对比

| 维度 | NVFP4 | MXFP4 |
|------|-------|-------|
| **制定者** | NVIDIA | OCP 社区 |
| **提出时间** | 2024（Blackwell） | 2024 |
| **微块大小** | 16 元素 | 32 元素 |
| **块 scale 格式** | FP8 E4M3（256 档，精度高） | E8M0（256 档，纯 2 的幂，无尾数） |
| **全局 scale** | ✅ FP32 per-tensor | ❌ 无 |
| **精度** | 更高（两级缩放） | 稍低（单级缩放） |
| **通用性** | NVIDIA 专用 | 通用（AMD/Intel 也支持） |
| **典型场景** | Blackwell 推理 | 训练 + 推理 |

**关键差异**：
1. **块 scale 精度**：NVFP4 用 FP8 E4M3 能表达任意小数缩放；MXFP4 的 E8M0 只能表达 $2^k$ 形式的缩放（精度粗）
2. **是否有全局 scale**：NVFP4 有 FP32 全局 scale 兜底，能覆盖更宽的数值范围
3. **微块大小**：MXFP4 用 32 元素块，scale 分摊更小但精度贴合度弱一点

### 什么时候选哪个

- **NVIDIA Blackwell 推理** → NVFP4（硬件原生优化）
- **训练场景** → MXFP4 更成熟（OCP 标准，多厂商支持）
- **AMD/Intel 平台** → MXFP4（NVFP4 是 NVIDIA 专属）

---

## 相关文档

- [FP8 详解](./fp8) — NVFP4 的块 scale 就是 FP8 E4M3
- [量化粒度](/docs/quantization/basics/granularity) — NVFP4 是极致的 per-group 量化
- [量化本质](/docs/quantization/basics/quantization-mapping) — 非线性量化的极致代表
