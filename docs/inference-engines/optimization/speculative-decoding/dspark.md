---
sidebar_position: 7
title: DSpark（半自回归 + 生产级调度）
---

# ✨ DSpark

> **DeepSeek 开源的投机解码框架（配套完整训练代码库 DeepSpec）**
> 同时解决两个问题：DFlash 路线暴露的**草稿连贯性**问题，以及**生产环境服务调度**问题。

## 🎯 面试速记版

> **一句话**：DSpark 保留 DFlash 那样的并行主干（快），但在其上挂一个轻量**串行马尔可夫头（Markov head）**，让每个位置在并行预测的同时能"回头看一眼"紧邻的前一个 token —— 用极小的成本给纯并行草稿重新引入**局部因果性**，治好 "suffix decay"（缝合怪草稿）问题。

> **两大创新**：
> 1. **半自回归架构（semi-autoregressive）**：并行主干 + 串行马尔可夫头 = "块内并行、块间自回归" 的折中，兼顾速度与连贯性
> 2. **置信度调度（confidence-scheduled verification）**：给每个草稿位置配置信度头（confidence head）预测"能被接受"的概率，配合**硬件感知调度器**动态决定验证深度 —— 闲时验完整段追求延迟，忙时只验高置信度前缀追求吞吐

> **工程价值**：DSpark 被设计成一个可**无缝挂载**的插件模块（DSparkBlock），例如 `DeepSeek-V4-Pro-DSpark` 不是新模型，而是原版 V4-Pro 加装了一套 DSparkBlock，通过读取主模型最后三层的隐藏状态来工作。真实生产环境的极限高并发压测下，DSpark 可以给单用户带来 **57%~85%** 的吞吐提速。

---

## 1. 背景：DFlash 的 "Suffix Decay" 问题

DFlash 的并行生成范式暴露出一个结构性缺陷：**块内 token 是并行、独立生成的，缺乏彼此之间的因果依赖建模**。

**举例**：
> 草稿模型可能在某个位置独立地觉得 "of course" 和 "no problem" 都合理，但由于各位置分开打分、独立采样，可能把两个"分开看都对、拼在一起却别扭"的词拼接在一起，出现前后不搭的"缝合怪"草稿。

这个问题在 DSpark 论文里被明确命名为 **"suffix decay"（后缀衰减）**：
- 块靠前的 token 接受率还行
- 但越往块后期走，token 之间不连贯累积，被拒绝的概率越大
- 起草预算堆得越多浪费越严重

DSpark 要解决的核心就是**在保留并行主干效率的同时，把因果依赖重新缝合回来**。

---

## 2. 半自回归架构：局部因果性

### 2.1 核心设计

```
输入 block: [anchor, MASK, MASK, ..., MASK]
                │
                ▼
       ┌──────────────────────┐
       │  DFlash 式并行主干    │  ← 保留大部分起草工作，保证速度
       │  (block diffusion)   │
       └──────────┬───────────┘
                  │
                  ▼
           每个位置的初始预测
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   ┌─────────────────────────────┐
   │  串行马尔可夫头（Markov head） │  ← 每个位置"回头看一眼"前一个
   │  只看邻近的前一个 token       │
   └──────────┬──────────────────┘
              │
              ▼
        最终草稿 tokens
```

**关键**：**每个位置在并行预测的同时，能"回头看一眼"紧邻的前一个 token**（瞄一眼前面那个 token 预测的是啥）。

### 2.2 为什么 work？

- **成本极低**：马尔可夫头只看**紧邻的一个前驱**，不是完整因果链
- **不影响并行度**：主干仍然一次前向出整块，马尔可夫头只是一个轻量后处理
- **补齐关键弱点**：直接干掉了 "of course + no problem" 这种独立采样导致的不连贯
- **命名精准**："马尔可夫" 头 —— 每个位置只依赖前一个位置（一阶马尔可夫性）

### 2.3 定位：不是纯自回归也不是纯并行

| 路线 | draft 速度 | 连贯性 |
|---|---|---|
| EAGLE-3（纯自回归） | 慢（γ 次 forward）| 高 |
| DFlash（纯并行） | 快（1 次 forward） | 中低（suffix decay） |
| **DSpark（半自回归）** | **中（1 次并行 + 1 次马尔可夫后处理）** | **中高** |

**DSpark 是 DFlash 的自然演进**：不放弃并行的速度收益，用极小的串行代价换回连贯性。

---

## 3. 置信度调度的验证

这一块体现出了 DeepSeek 的"工程狂魔"底色。

### 3.1 置信度头（Confidence Head）

DSpark 给每个草稿位置配了一个 **confidence head**，用来预测这个位置的 token"能活到最后被接受"的概率。

这不是简单的"draft 模型自己的输出概率"，而是**专门训练的独立小头**，学习目标就是"预测这个位置在最终 verify 阶段是否会被 accept"。

### 3.2 硬件感知调度器（Hardware-Aware Scheduler）

```
                ┌──────────────────┐
                │  实时负载监控     │
                └────────┬─────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
      ┌───────────────┐    ┌────────────────┐
      │  服务器空闲    │    │  高并发算力紧张 │
      │               │    │                │
      │  验证整段草稿  │    │  只验高置信度  │
      │  追求单次延迟  │    │  前缀，节省算力 │
      │  最低          │    │  服务更多请求  │
      └───────────────┘    └────────────────┘
```

- **闲时**：追求极致的单次请求延迟（Latency），让主模型老老实实验证整段草稿
- **忙时**：调度器根据实时的置信度分数，**只验证"大概率能中"的草稿前缀**，直接砍掉后面大概率要被打回的部分，节省验证算力去服务更多并发请求

### 3.3 为什么这个调度重要？

传统投机解码的加速比公式假设 verify 成本是固定的：

$$
L = \frac{T_{draft} + T_{verify}}{\tau}
$$

但在**高并发生产环境**下，$T_{verify}$ 会变得非常关键：
- 每个请求都要占用 verify 的计算资源
- 并发数一多，verify 的 GPU 就成为瓶颈
- 与其把资源浪费在"验证注定被拒绝的后半段"，不如砍掉去服务下一个请求

DSpark 把优化战场从"单请求延迟" **扩展到了"整个服务集群的吞吐"**。

---

## 4. 部署：DSparkBlock 插件化

**在部署上，DSpark 被设计成了一个可以无缝"挂载"的插件模块**。

### 4.1 例子：`DeepSeek-V4-Pro-DSpark`

它**不是一个新模型**，而是原版 V4-Pro 加装了一套**独立的 DSparkBlock**：

- 通过读取主模型**最后三层的隐藏状态**来工作
- 独立训练、独立部署
- 主模型完全不用改

### 4.2 好处

- ✅ 主模型和 DSpark 解耦，可以独立迭代
- ✅ 不影响主模型的部署路径
- ✅ 关闭 DSpark 就是标准 V4-Pro
- ✅ 可以针对不同模型分别训练适配的 DSparkBlock

---

## 5. 实测性能

**在真实生产环境的极限高并发压测下，DSpark 可以给单用户带来 57%~85% 的吞吐提速。**

对比其他方法：

| 方法 | 场景 | 加速比 |
|---|---|---|
| EAGLE-3 | 单并发 | 2-3× |
| DFlash | 单并发 | 4-6× |
| **DSpark** | **高并发生产环境** | **57%-85% 吞吐提升** |

⚠️ 注意 DSpark 的加速比不能直接和 DFlash 比 —— DFlash 论文的 4-6× 是单并发 latency 视角，DSpark 的 57%-85% 是**生产集群的吞吐视角**。两者优化的目标不同。

---

## 6. 与 DFlash 的对比

| 维度 | DFlash | DSpark |
|---|---|---|
| 优化目标 | 单请求 latency | **生产集群吞吐 + latency** |
| 起草方式 | 纯并行 block diffusion | **半自回归**（并行 + 马尔可夫头）|
| 因果依赖 | 无（suffix decay 问题）| **一阶马尔可夫依赖** |
| 验证策略 | 全部草稿都验证 | **置信度调度**动态决定 |
| 部署形态 | 独立 draft 模型 | **DSparkBlock 插件** |
| 训练代码 | SpecForge | **DeepSpec**（DeepSeek 官方）|
| 硬件依赖 | B200/H200 | 生产级通用 |

**核心哲学差异**：
- DFlash 想的是 "how to make draft as fast as possible"
- DSpark 想的是 "how to make the whole serving pipeline as efficient as possible"

---

## 7. 后续演进：JetSpec

JetSpec（阶跃星辰, 2026）从另一个角度解决同样的因果性问题——把因果掩码引入**树状草稿**：

- **因果并行树起草（causal parallel tree drafting）**
- 一次前向计算整棵草稿树的所有节点 logits
- 但用**树状因果注意力掩码**约束，每个节点只能看祖先节点
- 兼顾并行性和因果分解

JetSpec 论文题目就叫"打破投机解码的扩容天花板"——在 Qwen3-8B MATH-500 做到 **9.64×** 加速。

**DSpark（马尔可夫头补一手局部依赖）和 JetSpec（树状因果掩码）几乎是从两个不同切口，把"因果性"重新缝合回并行起草框架里。**

---

## 相关文档

- 前置：[DFlash](/docs/inference-engines/optimization/speculative-decoding/dflash)（DSpark 的前身，暴露了 suffix decay 问题）
- 前置：[原理基础](/docs/inference-engines/optimization/speculative-decoding/basics)
- 综合：[技术编年史](/docs/inference-engines/optimization/speculative-decoding/chronicle)（有 DSpark vs JetSpec 的深度对比）

## 参考

- DeepSeek-AI, *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*, 2026
- 训练代码库：DeepSpec (DeepSeek 官方)
- 内网原文：[投机解码：从 EAGLE 到 DSpark (iWiki)](https://iwiki.woa.com/p/4027810632) / [投机解码技术编年史 (iWiki)](https://iwiki.woa.com/p/4024247289)
