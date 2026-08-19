---
sidebar_position: 0
title: QAT 方法总览
---

# 🎯 QAT 方法

> **量化感知训练（Quantization-Aware Training）**：在训练过程中引入"伪量化"操作模拟低精度效应，让模型主动适应量化误差。

QAT 通常能获得比 PTQ 更好的精度——因为模型在训练过程中就能适应量化效应，因此更适合对量化误差敏感的模型。代价是**需要重新训练**，消耗更多计算资源和时间，且实现复杂度通常更高。

## 内容地图

| 文档 | 关键词 | 定位 |
|------|--------|------|
| [STE 直通估计器](/docs/quantization/qat/ste-fundamentals) | Straight-Through Estimator / 伪量化 | QAT 的核心机制 |

## 待补充话题

- 🌱 LSQ / LSQ+ — 可学习的 scale
- 🌱 OmniQuant — 大模型 QAT 的现代方案
- 🌱 QAD (Quantization-Aware Distillation) — 量化 + 蒸馏结合
- 🌱 LLM-QAT — Meta 的大模型 QAT 方案

## QAT 核心思想

- **前向传播**：模拟量化误差（fake quantization），让模型看到量化后的结果
- **反向传播**：绕过量化操作的不可导问题，通过 STE 让梯度流过
- **训练完成后**：把 fake quant 模块替换为真实的低精度 kernel

## QAT vs PTQ 快速对比

| 维度 | PTQ | QAT |
|------|-----|-----|
| **是否需要训练** | ❌ | ✅ |
| **计算成本** | 低（分钟~小时） | 高（天~周） |
| **精度上限** | 中 | 高 |
| **对敏感模型** | 可能崩 | 能救回来 |
| **实现复杂度** | 低 | 高 |
| **数据依赖** | 校准集 500~1000 条 | 完整训练/微调数据 |
| **代表方法** | GPTQ / AWQ / SmoothQuant | LSQ / OmniQuant / LLM-QAT |

## 何时选 QAT

- PTQ 掉点严重、业务无法接受
- 量化位宽极低（INT4 甚至 INT2），PTQ 力不从心
- 有充足训练资源和数据
- 模型对量化天然敏感（部分 SR、检测、语音模型）

## 何时选 PTQ

- 快速部署，没时间训练
- 训练数据获取困难
- 8-bit 量化，PTQ 精度已够用

## 现代大模型的中间路线：QAD

对于大模型（GPT/LLaMA 级），完整 QAT 太贵——训练成本可能等于重新训一次模型。所以 2024-2025 出现了 **QAD（Quantization-Aware Distillation）**：以 FP16 原模型为教师，量化模型为学生，用**蒸馏**代替完整 QAT。

详见：[QAD vs OPD 对比](/docs/quantization/practice-notes/qad-vs-opd)

---

## 相关文档

- [STE 直通估计器](/docs/quantization/qat/ste-fundamentals) — QAT 的核心操作
- [PTQ 掉点排查](/docs/quantization/ptq/debug-quantization-loss) — 什么情况下该升级到 QAT
- [QAD vs OPD](/docs/quantization/practice-notes/qad-vs-opd) — 现代大模型的 QAT 替代方案
