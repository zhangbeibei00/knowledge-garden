---
sidebar_position: 0
title: 投机解码总览
---

# 🚀 投机解码（Speculative Decoding）

> 让一个"实习生"（draft 模型）先蒙几个 token，让"老板"（target 大模型）一次性并行验证。
> 蒙对了照单全收，蒙错了老板自己改。
> **加速的核心是：把 target 的多次串行 forward 压缩到一次并行 forward。**

## 内容地图

| 文档 | 关键词 | 定位 |
|------|--------|------|
| [🕰️ 投机解码技术编年史](/docs/inference-engines/optimization/speculative-decoding/chronicle) | 从 2018 到 2026，一路捋清 | 先看这个，建立时间线 |
| [📐 原理基础](/docs/inference-engines/optimization/speculative-decoding/basics) | draft + verify + accept/reject | 投机解码的数学本质 |
| [🐍 Medusa](/docs/inference-engines/optimization/speculative-decoding/medusa) | 多解码头 + tree attention | 第一批"自投机"实践 |
| [🦅 EAGLE 系列](/docs/inference-engines/optimization/speculative-decoding/eagle) | EAGLE-1/2/3 | feature-level 自回归 draft |
| [🎯 MTP（Multi-Token Prediction）](/docs/inference-engines/optimization/speculative-decoding/mtp) | DeepSeek-V3 因果链式 | 训练目标 + 投机解码双用途 |
| [⚡ DFlash](/docs/inference-engines/optimization/speculative-decoding/dflash) | Block Diffusion + KV 注入 | 一次 forward 出整块 token |
| [✨ DSpark](/docs/inference-engines/optimization/speculative-decoding/dspark) | 半自回归 + 并行草稿 | DeepSeek 开源的生产级方案 |

## 学习顺序建议

1. 先看 **[技术编年史](/docs/inference-engines/optimization/speculative-decoding/chronicle)**，建立整体脉络（从 BPD → Medusa → EAGLE → MTP → DFlash → DSpark）
2. 再看 **[原理基础](/docs/inference-engines/optimization/speculative-decoding/basics)**，把「接受/拒绝 + residual + bonus token」的数学推导跑通
3. 然后按时间线深读单篇：**Medusa → EAGLE 系列 → MTP → DFlash → DSpark**

## 全景速览

**自回归 draft 路线**（草稿从左到右一个一个猜）：

- **Medusa**：主干挂多个平行 head，每个 head 独立预测不同 offset → 简单但连贯性弱
- **EAGLE-1/2/3**：单层 AR head，用 feature 而不是 embedding 做条件；EAGLE-2 动态树；EAGLE-3 回退到 next-token + 中浅层 feature 融合
- **MTP**：训练时是辅助目标，推理时天然当 drafter；DeepSeek-V3 因果链式设计

**扩散 draft 路线**（草稿一次性并行猜出整块）：

- **DFlash**（ICML 2026, UCSD）：block diffusion + KV 注入 + 5 层 draft，speedup 4–6×
- **DSpark**（DeepSeek 开源）：半自回归架构，兼顾并行度和连贯性

## 核心 trade-off

$$
\eta = \frac{L_{\text{target}}}{(T_{\text{draft}} + T_{\text{verify}})/\tau}
$$

- **↑ τ**：单轮平均接受的 token 数（draft 越准）
- **↓ T_draft**：draft 一次生成 γ 个 token 的耗时

不同路线优化的位置不一样：

| 路线 | 优化重点 | 典型 speedup |
|------|----------|--------------|
| Medusa | ↑ τ（多路径笛卡尔积） | ~2.2× |
| EAGLE 系列 | ↑ τ + 静态/动态 tree | 2–3× |
| MTP | ↑ τ + 训练收益 | ~1.8× |
| DFlash | **↓ T_draft**（一次 forward） | 4–6× |
| DSpark | 两边都想要 | 生产级最快 |

## 面试速记

> **一句话解释投机解码**：让一个小模型先蒙几个 token，让大模型一次并行验证；通过 rejection sampling 保证最终分布严格等于大模型分布（无损加速）。加速的收益来自"把大模型的多次串行 forward 压缩到一次并行 forward"。

> **无损性怎么保证**：接受阶段用 $\alpha = \min(1, p/q)$ 判决；拒绝时从 residual 分布 $\text{normalize}(\max(0, p-q))$ 采样；全接受时用 bonus token。三者相加恰好等于大模型分布 $p$。

> **加速比公式**：$\eta = L_{\text{target}} / ((T_{\text{draft}} + T_{\text{verify}})/\tau)$，$\tau$ 是每轮接受的 token 数。所以要么"猜得准"（↑τ），要么"猜得快"（↓T_draft）。
