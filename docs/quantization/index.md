---
sidebar_position: 0
title: 模型量化总览
---

# 📐 模型量化

> 让大模型跑得更快、更省、更便宜。

## 内容地图

### [📖 基础理论](/docs/quantization/basics)
从零理解量化的数学基础与工程实现。
- [量化本质与映射](/docs/quantization/basics/quantization-mapping) — scale/zero_point、线性 vs 非线性
- [量化粒度](/docs/quantization/basics/granularity) — per-tensor / per-channel / per-group
- [激活量化](/docs/quantization/basics/activation-quantization) — 动态 vs 静态

### [📊 PTQ 方法](/docs/quantization/ptq)
训练后量化（Post-Training Quantization）主流方法。
- [校准算法](/docs/quantization/ptq/calibration-algorithms) — MinMax / Percentile / KL / MSE
- [量化掉点排查](/docs/quantization/ptq/debug-quantization-loss) — 一份实战 debug 手册
- 🌱 GPTQ / AWQ / SmoothQuant / HQQ 精读（待补充）

### [🎯 QAT 方法](/docs/quantization/qat)
量化感知训练（Quantization-Aware Training）。
- [STE 直通估计器](/docs/quantization/qat/ste-fundamentals) — QAT 的核心机制
- 🌱 LSQ / OmniQuant / LLM-QAT（待补充）

### [🔢 低精度格式](/docs/quantization/low-precision-formats)
硬件与数值格式。
- [FP16 与 BF16](/docs/quantization/low-precision-formats/fp16-bf16) — 训练主力的两条路线
- [FP8（E4M3/E5M2）](/docs/quantization/low-precision-formats/fp8) — H100 时代主力
- [NVFP4 / MXFP4](/docs/quantization/low-precision-formats/nvfp4-mxfp4) — Blackwell 4-bit
- [INT8](/docs/quantization/low-precision-formats/int8) — 经典 PTQ 老将

### [⭐ 实践笔记](/docs/quantization/practice-notes)
一手实践与深度对比。
- [QAD vs OPD 对比](/docs/quantization/practice-notes/qad-vs-opd)

## 学习建议

如果你是**从零开始**：
1. 先看基础理论（1~2 天）
2. 挑一个 PTQ 方法（推荐 GPTQ）深挖一次
3. 了解 FP8/FP4 数值格式
4. 看实践笔记里的对比与踩坑

如果你是**有工程背景想补理论**：
1. 直接跳到实践笔记，理解现代方法的问题定义
2. 反向去看 PTQ/QAT 的经典方法
3. 补充数值格式知识

---

*已完成模块：基础理论 · 低精度格式 · PTQ 校准 & debug · QAT STE · 实践笔记 QAD vs OPD。*
*🚧 补充中：GPTQ/AWQ/SmoothQuant 精读、LSQ/OmniQuant、更多实践对比。*
