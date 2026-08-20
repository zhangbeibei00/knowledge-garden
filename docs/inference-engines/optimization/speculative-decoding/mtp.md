---
sidebar_position: 5
title: MTP（Multi-Token Prediction）
---

# 🎯 MTP（Multi-Token Prediction）

> **投机解码"焊死"在预训练里的哲学。**
> DeepSeek-V3 的做法：在预训练阶段就让模型学会预测未来多个 token。训练时是辅助目标改善表征，推理时天然当 drafter 用。

## 🎯 面试速记版

> **一句话**：MTP 是一种**训练目标**（不只是推理技巧）——让模型在每次 forward 时不仅预测下一个 token，还并行预测未来第 2、3、...、n 个 token，共享主干 + 独立/因果链头。**训练侧**：拓宽监督信号 → 学到更具前瞻性的表征（DeepSeek-V3 报告 +2.4% NL、+17% code）。**推理侧**：这些辅助头可以直接当投机解码的 drafter，**免独立 draft 模型**，DeepSeek-V3 实测 ~1.8× 加速。

> **两种架构**：
> - **Meta 原始版**（ICML 2024）：**独立并行头**，每个 head 是单层 Transformer block，预测自己的 offset，头之间无依赖。共享 unembedding 矩阵 + 逐头反向传播 → 显存开销与 NTP 相同，训练额外时间仅 ~2%
> - **DeepSeek-V3 因果链式**（2024 末）：第 k 个 MTP module 的输入 = 第 k-1 个模块的输出 + 未来 token embedding，链式串行。信息流通更好，但必须串行 → V3 只用 D=1

> **推理时的双重用途**：
> - MTP Vanilla：每个 module 分别负责固定 offset
> - MTP Eagle：单个 module 循环调用多次（套娃模式），最大化复用 KV Cache

> **理论深意**：2026 年的理论工作证明 MTP 通过**梯度解耦**改变了学习动力学——浅层预测头的梯度绕过未初始化的深层，让 Layer 1 先学"通用前驱指针"、Layer 2 再学"内容匹配"，NTP 做不到的规划能力被激活。

---

## 1. 什么是 MTP

### 传统范式：Next-Token Prediction (NTP)

$$
L_{NTP} = -\sum_t \log P_\theta(x_{t+1} \mid x_1, ..., x_t)
$$

模型只看当前 token，预测下一个。每步只给一个 token 的监督信号。

### MTP 的核心思路

$$
L_{MTP} = -\sum_t \sum_{k=1}^{n} \log P_\theta(x_{t+k} \mid x_1, ..., x_t)
$$

一次预测未来第 1、2、...、n 个 token。**把「信息通道」拓宽了 n 倍**。

> 直觉类比：NTP 像戴着窥视孔看世界（每次一个像素），MTP 把窥视孔变成窗户（一次看到更多画面）。

---

## 2. Meta 原始方案：独立并行头（ICML 2024）

> 论文：*Better & Faster Large Language Models via Multi-token Prediction*

### 架构

```
    输入序列
      │
      ▼
    ┌──────────────────────┐
    │   共享 Transformer   │
    │    主干网络          │ ← 只做一次前向
    └──────────┬───────────┘
               │ 隐藏表示 h
         ┌─────┼─────┬─────┬─────┐
         ▼     ▼     ▼     ▼     ▼
      Head 1 Head 2 Head 3 ... Head n
      (1层TF) (1层TF) (1层TF)    (1层TF)
         │     │     │           │
         ▼     ▼     ▼           ▼
      共享 Unembedding 矩阵 (一个!)
         │     │     │           │
         ▼     ▼     ▼           ▼
      P(x_t+1) P(x_t+2) P(x_t+3) P(x_t+n)
```

- **共享主干**：无论多少 head，Transformer 主干只算一次
- **独立预测头**：每个 head 是单层 Transformer block，独立预测各自的 offset token
- **共享 Unembedding**：所有 head 共用一个输出投影层，大幅节省显存

### 显存优化：逐头反向传播

朴素实现需要 $O(nV + d)$ 显存。Meta 的解法：

```
for k = 1 to n:
    前向计算 Head k → 得到 logits
    计算交叉熵损失
    反向传播 → 得到梯度
    立即丢弃 Head k 的中间激活
    累积梯度
```

峰值显存降至 $O(V + d)$，与单头 NTP 完全相同。**额外训练时间仅增加约 2%。**

### 实验结果

| 指标 | 结果 |
|---|---|
| 代码生成 (HumanEval) | +12% 解决问题数 |
| 代码任务总体 | **+17%** |
| 自然语言任务 | +2.4%（温和但一致）|
| 推理加速 | 最高 **3 倍** |
| 最优 head 数（子词分词器）| n=4 |
| 最优 head 数（字节级分词器）| n=8 |

---

## 3. DeepSeek-V3 改造：因果链式架构

> 论文：*DeepSeek-V3 Technical Report*

### 架构差异

Meta 的独立并行头 → DeepSeek 的**因果链式**：

```
输入序列 → 共享主干 → h
                       │
                       ▼
            ┌──────────────────────┐
            │  MTP Module 1        │
            │  输入 = h ⊕ emb(t+1) │
            │  → RMSNorm → TF块    │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  MTP Module 2        │
            │  输入 = out1 ⊕ emb(t+2)│
            │  → RMSNorm → TF块    │
            └──────────┬───────────┘
                       │
                       ▼
                     ...
```

**核心差异**：第 k 个 MTP 模块的输入 = 第 (k-1) 个模块的输出 + 未来 token 的 embedding，两者拼接后送入 Transformer block。

### 两种架构对比

| 维度 | Meta (2024) | DeepSeek-V3 (2024) |
|---|---|---|
| 头间依赖 | 无（独立并行）| **有（因果链式）** |
| 信息流通 | 各自独立猜 | 逐步传递，站前人肩膀上 |
| 最优深度 | n=4 | D=1（仅一个额外 MTP 模块）|
| 额外参数 | n 个单层 TF 块 | 14B（主模型 671B）|
| 推理用途 | 自投机解码 | 同样用于投机解码 |

### 为什么因果链更好？

独立并行头预测 $x_{t+3}$ 时，不知道 $x_{t+1}$ 和 $x_{t+2}$ 是什么。因果链式让每一步预测都能利用前一步的信息。**但因果链必须串行计算（不能并行）**，DeepSeek-V3 折中只用 D=1。

### 双重用途设计

```
       训练时                      推理时
   ┌──────────────┐          ┌──────────────┐
   │ MTP 辅助训练  │          │ 投机解码      │
   │ 目标          │    →     │ Draft 模型    │
   │              │          │              │
   │ 改善表示质量  │          │ 加速推理      │
   │ 丰富训练信号  │          │ 无需额外模型  │
   └──────────────┘          └──────────────┘
```

MTP 模块训练时是辅助目标，提升主模型质量；**推理时摇身一变成为投机解码的 draft 模型**——不需要额外训练、不需要额外部署。

---

## 4. 推理策略：MTP Vanilla vs MTP Eagle

- **MTP Vanilla**：每个 MTP 模块分别负责预测未来固定偏移位置的 token
- **MTP Eagle**：借鉴 EAGLE 思路，把同一个 MTP 模块**循环调用多次**（"套娃模式"）
  - 契合官方只有单个 MTP 模块（D=1）的情况
  - 推理时能最大化复用同一份 KV Cache

验证阶段依然是标准投机解码：把一串草稿 token 拼上主模型自己刚采样出的"锚点" token，塞给主模型做一次前向验证，比对哪些位置的预测和草稿吻合，取最长匹配前缀通过。

---

## 5. 理论基础：为什么 MTP 有效？

### 信息论视角：更密的训练信号

- NTP 每步只给一个监督信号，MTP 拓宽了 n 倍
- 模型内部表示必须足够丰富，才能同时支撑 n 个位置的准确预测
- 迫使模型学习更具全局性的特征

### 梯度解耦与规划能力（2026 年理论突破）

论文 *"How Transformers Learn to Plan via Multi-Token Prediction"*（2026）提供了严格的理论分析：

**NTP 失败模式**：
- 在两层 Transformer 路径规划任务中，NTP 的梯度在上下文中扩散
- 当 Layer 2 未初始化时，模型无法发现"反向推理回路"
- 导致不能完成路径规划

**MTP 成功机制**：
- 浅层预测头（预测 $x_{t+2}$）的梯度**绕过**未初始化的 Layer 2
- **梯度解耦** → 各层独立学习自己的功能：
  - Phase I：Layer 1 通过浅层 MTP 损失收敛到"通用前驱指针"
  - Phase II：Layer 1 固定后，深层损失优化 Layer 2 进行内容匹配

**核心结论**：MTP 不只是提升准确率，它**从根本上改变了学习动力学**，使模型能发展出 NTP 无法实现的规划能力。

### 隐式"思考"能力

NTP 训练的 LLM 在隐藏状态中已经编码了关于多个未来 token 的信息，只是未被显式利用。MTP 训练把这种**潜在能力激活并强化**。

```
MTP (隐式规划)  ←→  CoT (显式推理)
  │                      │
  │  改善每一步推理      │  组织整体推理链条
  │  的基础质量          │
  └──────┬───────────────┘
         ▼
    DeepSeek-R1 / MiMo
    最强推理模型
```

---

## 6. 工业落地

### DeepSeek-R1

- 继承 DeepSeek-V3 的因果链式 MTP 架构
- **MTP（隐式规划）+ RL（显式推理）** = 最强组合
- NVIDIA TensorRT-LLM 为 R1 的 MTP 投机解码做了 Blackwell GPU kernel 级优化

### Qwen3.5（阿里巴巴）

- 首个在模型卡上明确标注 "MTP: trained with multi-steps" 的主流开源模型
- MTP 被烘焙进预训练过程
- vLLM 和 SGLang 均提供原生 MTP 投机解码支持

### MiMo（小米）

- MiMo-7B-Base：7B 参数，使用 MTP，在 25 万亿 token 上预训练
- **7B 模型超越多个 32B 模型**，经 RL 微调后在编程和数学上超 OpenAI o1-mini

### 服务框架原生支持

| 框架 | MTP 支持方式 |
|---|---|
| **vLLM** | `method: "mtp"` 原生支持 DeepSeek/Qwen3.5/MiMo |
| **SGLang** | `--speculative-algo NEXTN --speculative-num-steps 3` |
| **TensorRT-LLM** | Blackwell GPU kernel 级优化 |

---

## 7. 优缺点

**优点**：
- ✅ 训练质量提升（代码 +17%，NL +2.4%）
- ✅ 推理加速（~1.8× on DeepSeek-V3）
- ✅ 训练开销极小（~2%）
- ✅ **无需独立 draft 模型，无需额外部署**
- ✅ 与 RL 互补（MTP + CoT 最强组合）
- ✅ 生态成熟（vLLM/SGLang/TensorRT-LLM 全面支持）

**缺点**：
- ❌ 多 token 预测的最佳数量 n 依赖数据分布、词表大小
- ❌ 某些自然语言任务上 MTP 效果不如 NTP
- ❌ 需要**从预训练阶段**就设计，不能"事后加装"

---

## 8. 前沿方向

- **L-MTP**（NeurIPS 2025）：跳跃式多 token 预测，预测非相邻 token，提供更好的长距离依赖梯度信号
- **Future Summary Prediction**：不预测离散 token，而是预测未来序列的紧凑摘要向量
- **FastMTP**（腾讯）：解决 MTP 头训练-推理不匹配，位置共享权重 + 语言感知动态词表压缩，**2.03 倍加速**
- **Register Tokens**（NeurIPS 2025）：在输入序列中交错插入可学习的 register token 支持 MTP

---

## 相关文档

- 前置：[原理基础](/docs/inference-engines/optimization/speculative-decoding/basics)
- 竞品：[EAGLE 系列](/docs/inference-engines/optimization/speculative-decoding/eagle)（事后补课，独立训练 draft head）
- 后续：[DFlash](/docs/inference-engines/optimization/speculative-decoding/dflash) / [DSpark](/docs/inference-engines/optimization/speculative-decoding/dspark)（并行 draft）

## 参考

- Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction*, ICML 2024
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024
- 内网原文：[MTP 多 Token 预测技术报告 (iWiki)](https://iwiki.woa.com/p/4022911238)
