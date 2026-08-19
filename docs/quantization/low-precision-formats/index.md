---
sidebar_position: 0
title: 低精度格式总览
---

# 🔢 低精度格式

> 硬件和数值格式的博弈：从 FP32 到 FP4，每一次腰斩都改变了训练/推理的物理定律。

## 内容地图

| 文档 | 位宽 | 定位 |
|------|------|------|
| [FP16 与 BF16](/docs/quantization/low-precision-formats/fp16-bf16) | 16-bit | 训练主力，FP16 vs BF16 的取舍 |
| [FP8（E4M3 / E5M2）](/docs/quantization/low-precision-formats/fp8) | 8-bit | H100/H200 训练推理新宠 |
| [NVFP4 / MXFP4](/docs/quantization/low-precision-formats/nvfp4-mxfp4) | 4-bit | 2024-2025 大模型部署主流 |
| [INT8](/docs/quantization/low-precision-formats/int8) | 8-bit | 经典量化，PTQ 老将 |

## FP 家族的共同特点

Floating Point（浮点数）由 IEEE 定义标准，由**符号位（sign）、指数位（exponent）、小数位（fraction/mantissa）**三部分组成。

FP 的量化步长是**中间密、两边稀疏**（非均匀）：

![FP 步长分布](./assets/fp-step-distribution.png)

这个特性天然适合权重和激活的分布——大部分值集中在 0 附近，少量离群点分布在远端。所以现代大模型量化越来越偏向 FP 格式，尤其是 FP8 / FP4。

## 数值格式对比速查

| 格式 | 总 bit | 符号 | 指数 | 尾数 | 数值范围 | 主要用途 |
|------|-------|------|------|------|---------|---------|
| FP32 | 32 | 1 | 8 | 23 | ~±3.4e38 | 训练基准 |
| FP16 | 16 | 1 | 5 | 10 | ~±65504 | 训练/推理 |
| BF16 | 16 | 1 | 8 | 7 | ~±3.4e38 | 训练主力 |
| FP8 E4M3 | 8 | 1 | 4 | 3 | ~±448 | 训练权重 |
| FP8 E5M2 | 8 | 1 | 5 | 2 | ~±57344 | 训练梯度 |
| FP4 E2M1 | 4 | 1 | 2 | 1 | ~±6 | 推理量化 |
| INT8 | 8 | — | — | — | -128~127 | 经典 PTQ |
| INT4 | 4 | — | — | — | -8~7 | GPTQ/AWQ |

## 硬件支持简表

| 硬件 | FP16 | BF16 | FP8 | FP4 | INT8 | INT4 |
|------|------|------|-----|-----|------|------|
| A100 | ✅ | ✅ | ❌ | ❌ | ✅ | ⚠️（需 kernel） |
| H100/H200 | ✅ | ✅ | ✅ | ❌ | ✅ | ⚠️（需 kernel） |
| B200/GB200 | ✅ | ✅ | ✅ | ✅ (NVFP4/MXFP4) | ✅ | ✅ |

---

## 相关文档

- [量化本质](/docs/quantization/basics/quantization-mapping) — FP 也是一种非线性量化
- [PTQ 方法](/docs/quantization/ptq) — 低精度格式怎么落地到实际方法
