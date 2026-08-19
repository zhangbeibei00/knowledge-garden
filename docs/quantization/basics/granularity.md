---
sidebar_position: 2
title: 量化粒度：per-tensor / per-channel / per-group
description: 从整张量到分组量化，粒度是量化精度与效率的第一权衡
tags: [quantization, basics, granularity, per-channel, per-group]
---

# 量化粒度

量化粒度回答一个问题：**一组 scale/zero_point 到底覆盖多大范围的数据？**

粒度越细 → 精度损失越小，但存储和计算开销越大。所有量化方法的第一个选择，都是在这三档里选。

## Per-Tensor 量化：一刀切

整个张量（一层的权重或激活）**共享一组** `scale` 和 `zero_point`。

- ✅ 简单、快、硬件最友好
- ❌ 对分布差异大的张量（尤其是大模型权重）精度损失显著
- **经典方法**：MinMax 量化——取张量全局 min/max 算 scale，最简单的 PTQ 方案

```
张量 shape = [out=1024, in=4096]
→ 只有 1 个 scale
```

**适用场景**：小模型、分布均匀的层。大模型上基本不用 W-only per-tensor。

## Per-Channel 量化：按输出通道分

每个**输出通道**（Conv 是输出 channel，Linear 是每一行）单独一组 `scale/zero_point`。

- ✅ 精度显著提升，实现成本可接受
- ✅ 硬件依然能高效支持（scale 是一维向量）
- **经典方法**：**SmoothQuant** — 对权重做平滑预处理，让激活值的离群点转移到权重上，从而降低激活量化误差、更适配 Per-Channel 量化

```
张量 shape = [out=1024, in=4096]
→ 有 1024 个 scale（每个输出通道一个）
```

**适用场景**：INT8 权重量化的标配，几乎所有主流 PTQ 方法都默认用 per-channel。

## Per-Group 量化：更细，按块分组

把每个输出通道再切成若干**小组**（group size 常见 32/64/128），每组一个 `scale/zero_point`。

- ✅ 精度最佳，尤其适合 INT4/FP4 这种低比特
- ❌ 存储和计算开销上升（scale 数量 × group 数量）
- ❌ 硬件需要专门优化 kernel 支持

**经典方法**：

| 方法 | 核心思路 |
|------|---------|
| **GPTQ** | 按列分组量化，用二阶信息迭代补偿量化误差 |
| **AWQ** | 按重要性分组，保护权重中"关键 1%"，低比特下精度优于 GPTQ |
| **NVFP4** | 每 16 个元素一个微块 + FP8 块 scale + FP32 全局 scale（两级缩放） |
| **MXFP4** | OCP 标准，每 32 个元素一个块，通用性优先 |

```
张量 shape = [out=1024, in=4096], group_size=128
→ 1024 × (4096/128) = 32768 个 scale
```

**适用场景**：INT4/FP4 大模型量化的必选项。低于 4bit 时基本只能靠 per-group。

## 三档粒度对比

| 维度 | Per-Tensor | Per-Channel | Per-Group |
|------|-----------|-------------|-----------|
| **scale 数量** | 1 | out_channel 数 | out_channel × (in/group_size) |
| **精度损失** | 大 | 中 | 小 |
| **额外存储** | 忽略 | ~0.1% | 1%~15% |
| **硬件支持** | 最好 | 好 | 需专用 kernel |
| **代表方法** | MinMax | SmoothQuant | GPTQ / AWQ / NVFP4 |
| **典型使用比特** | INT8 | INT8 | INT4 / FP4 |

## 常见问题：MinMax PTQ / AWQ / GPTQ / SmoothQuant 存下来的模型有什么区别？

从**存储结构**看：

- **MinMax PTQ (per-tensor)**：每层只多存 1 个 scale + zero_point
- **SmoothQuant (per-channel)**：每层多存 out_channel 个 scale + 一个 smooth 因子（用来在推理时对激活做预处理）
- **AWQ (per-group)**：每层多存 out_channel × n_groups 个 scale，权重顺序可能被重排（把重要通道放前面）
- **GPTQ (per-group)**：类似 AWQ，量化后的权重经过 Hessian 补偿，数值本身也被"改过"

所以就算你看两个 AWQ 和 GPTQ 的 checkpoint 都是同样的位宽和 group size，权重矩阵的具体数值也不一样——它们记录的是**各自补偿算法调整过的量化结果**。

## 权重量化 vs 权重+激活量化

粒度选择还和量化对象有关：

- **W-only（只量化权重）**：AWQ/GPTQ 的典型场景，激活保持 FP16。粒度可以做到很细（per-group INT4），推理时反量化到 FP16 后走原 kernel。
- **W+A（权重和激活都量化）**：SmoothQuant、FP8 训练/推理的典型场景。激活必须动态或半动态处理，粒度通常只能到 per-channel/per-token。

大模型场景现在的主流分工：
- **INT4 部署** → W-only per-group（AWQ/GPTQ）
- **FP8 训练/推理** → W+A per-channel 或 per-tensor
- **NVFP4 部署** → W+A per-group + 两级缩放

---

## 相关文档

- 上一步：[量化本质与映射](./quantization-mapping)
- 下一步：[激活量化：动态 vs 静态](./activation-quantization)
- [PTQ 校准算法](/docs/quantization/ptq/calibration-algorithms) — 不同粒度下 scale 怎么算
- [NVFP4/MXFP4](/docs/quantization/low-precision-formats/nvfp4-mxfp4) — 两级缩放的极致实现
