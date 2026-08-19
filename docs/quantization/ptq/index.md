---
sidebar_position: 0
title: PTQ 方法总览
---

# 📊 PTQ 方法

> 训练后量化（Post-Training Quantization）：不改训练流程，只在训练后做量化，工程落地成本最低。

## 内容地图

| 文档 | 关键词 | 定位 |
|------|--------|------|
| [校准算法](/docs/quantization/ptq/calibration-algorithms) | MinMax / Percentile / KL / MSE | PTQ 的核心：怎么算 scale |
| [量化掉点排查方法论](/docs/quantization/ptq/debug-quantization-loss) | 敏感层 / Hook / 混合精度 | 量化效果差怎么办 |

## 待补充话题

- 🌱 GPTQ 原理精读（Hessian 补偿的数学推导）
- 🌱 AWQ 原理精读（activation-aware weight quantization）
- 🌱 SmoothQuant 原理精读（activation-to-weight migration）
- 🌱 HQQ（Half-Quadratic Quantization）
- 🌱 llmcompressor / AutoGPTQ / AutoAWQ 工具链对比

## PTQ 全景速览

**校准算法**（怎么算 scale）：
- MinMax、Percentile、KL、MSE — 见 [校准算法](/docs/quantization/ptq/calibration-algorithms)

**权重量化算法**（怎么调整权重减少误差）：
- **GPTQ**：按列分组 + Hessian 二阶信息补偿
- **AWQ**：activation-aware，保护重要权重
- **HQQ**：Half-Quadratic 优化，speed-first

**联合权重+激活量化**：
- **SmoothQuant**：把激活离群点迁移到权重上
- **LLM.int8()**：混合精度，激活离群点走 FP16

**低比特浮点量化**：
- **NVFP4 / MXFP4**：见 [低精度格式](/docs/quantization/low-precision-formats/nvfp4-mxfp4)

## 学习顺序

1. 先看 [校准算法](/docs/quantization/ptq/calibration-algorithms)，把最基础的 MinMax/KL/MSE 跑一遍
2. 再看 [量化掉点排查](/docs/quantization/ptq/debug-quantization-loss)，学会诊断问题
3. 最后深挖 GPTQ/AWQ/SmoothQuant 的原理（笔记补充中）

---

## 相关文档

- [量化粒度](/docs/quantization/basics/granularity) — PTQ 方法多在讨论粒度和补偿策略
- [激活量化](/docs/quantization/basics/activation-quantization) — 静态 PTQ 的核心配套
- [QAT 方法](/docs/quantization/qat) — PTQ 掉点严重时的进阶方案
