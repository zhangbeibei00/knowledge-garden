---
sidebar_position: 0
title: 模型量化总览
---

# 📐 模型量化

> 让大模型跑得更快、更省、更便宜。

## 内容地图

### [📖 基础理论](./basics)
从零理解量化的数学基础与工程实现。
- 对称 vs 非对称量化
- Per-tensor / Per-channel / Per-group
- 量化误差与校准

### [📊 PTQ 方法](./ptq)
训练后量化（Post-Training Quantization）主流方法。
- GPTQ、AWQ、SmoothQuant、HQQ
- 各方法的原理、优劣、适用场景

### [🎯 QAT 方法](./qat)
量化感知训练（Quantization-Aware Training）。
- LSQ / LSQ+ / OmniQuant
- QAT 训练技巧

### [🔢 低精度格式](./low-precision-formats)
硬件与数值格式。
- FP8 / MXFP8 / NVFP4 / MXFP4 / INT4
- Tensor Core 支持链路

### [⭐ 实践笔记](./practice-notes)
一手实践与深度对比。
- [QAD - NVFP4 量化感知蒸馏](./practice-notes/qad-nvfp4)
- [QAD vs OPD 对比](./practice-notes/qad-vs-opd)

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

*🚧 大部分子模块还在填充中。已完成 = ⭐ 实践笔记。*
