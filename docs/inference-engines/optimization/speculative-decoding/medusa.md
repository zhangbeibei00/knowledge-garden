---
sidebar_position: 3
title: Medusa
---

# 🐍 Medusa

> 在大模型主干最后一层之上挂多个并行"头"，每个头独立预测未来第 k 个位置的 token，配合 tree attention 一次前向验证多路径。
> —— 自我投机（self-speculation）的开山之作之一。

## 🎯 面试速记版

> **一句话**：Medusa 在 target 模型上再挂 k 个轻量 FFN head（1 个 resblock + 1 个 linear），每个 head 独立预测未来第 1、2、...、k 个位置的 token；每个 head 各取 top-p 后**笛卡尔积**成候选路径，用 **tree attention**（静态稀疏树）一次 target forward 并行验证所有路径，选最长通过路径。**免独立 draft 模型**是最大工程价值，但每个头**看不到别的头的输出**，位置间无因果依赖，接受率上限受限。典型加速 ~2×。

> **两种训练模式**：
> - **Medusa-1**：冻结骨干、只训 head，资源省，可配 QLoRA
> - **Medusa-2**：骨干 + head 联合 SFT，加速比更高但需专门 protocol 防止生成质量掉

> **突破 greedy**：**Typical Acceptance** —— 用固定阈值 + 条件熵阈值取 min，保留超过阈值的 token 做贪心采样。支持 top-k / top-p 采样场景。

> **痛点**：
> - **算力上限**：一次验证 64 条路径，最多接收 4 个 token → 算力需求是自回归的 16 倍，大 batch 下达到 compute bound 后反而变慢
> - **head 独立性**：预测 next-next token 时不知道 next token 是什么，接受率天花板明显

---

## 1. 背景

Medusa（Cai et al., 2024）针对传统投机解码的两个痛点：

1. 独立小模型作为 draft 部署复杂（词表、KV Cache 都要单独维护）
2. 分布式系统下集成 draft 模型工程成本高

解法：沿用 Blockwise Parallel Decoding 的思路，**用多个 decoding heads 并行生成多个 token**，头结构用 FFN（1 层 resblock + 1 层 linear）。

## 2. 架构

```
                    input
                      │
                      ▼
            ┌──────────────────────┐
            │  target LLM 主干     │  ← 只做一次前向
            └──────────┬───────────┘
                       │ 最后一层 hidden
              ┌────────┼────────┬──────────┐
              ▼        ▼        ▼          ▼
           Head 1   Head 2   Head 3   ...  Head k
           (FFN)    (FFN)    (FFN)         (FFN)
              │       │       │            │
              ▼       ▼       ▼            ▼
           预测      预测      预测         预测
           t+1       t+2       t+3          t+k
```

- **Head 结构**：1 个 resblock + 1 个 linear，即单层 FFN
- **参数初始化**：`W1 = 0`，`W2` 同 LLM 原始 head
- **偏移量不同**：原始头预测 $i$，medusa heads 分别预测 $i+1, i+2, ...$

## 3. 候选路径：笛卡尔积 + Tree Attention

每个头可以指定 top-k 结果，把所有的 top-k 组装成候选路径。

例：`LM head (top-2) × Medusa head 1 (top-3) = 6` 种候选。**扩大候选集合增加命中概率。**

问题：解码路径随 top-k 和 head 数量急剧增长 → **构造稀疏树结构**。

论文手动设计了稀疏树的形状。以 MEDUSA-2 Vicuna-7B（4 个 medusa heads）为例：

- 树的层数 = medusa head 数 - 1
- 越靠左/前的节点有更多子节点路径
- 路径可提前结束，不一定要遍历到最后一层
- 最终剪枝到 64 条路径

## 4. Typical Acceptance：突破贪心限制

以往投机解码支持 top-k / top-p 采样比较难，Medusa 用了很朴素的方法：

- 保留超过一定概率阈值的 token
- 阈值取 **固定阈值 vs 条件熵阈值** 的 min
- 在这些 token 中做贪心采样选择 top-k

## 5. 训练策略

| 模式 | 骨干 | head | 适用 |
|---|---|---|---|
| **Medusa-1** | 冻结 | 训练 | 计算资源有限，占用小，可配 QLoRA |
| **Medusa-2** | 联合 SFT | 训练 | 追求加速比，需专门 protocol 保证质量不下降 |

## 6. 优缺点

**优点**：
- ✅ 支持 top-k / top-p 采样，打破贪心解码限制
- ✅ 免独立 draft 模型，工程简洁
- ✅ 训练成本低（Medusa-1 只训 head）

**缺点**：
- ❌ **接受率**：预测 next-next token 时，不知道上一个 token 是什么，预测不确定性增加
- ❌ **算力问题**：Medusa 本质是"用算力换时间"。单次推理验证 64 条路径，最多接收 4 个 token
  - 算力需求对比常规自回归增大 **16 倍**
  - 计算访存比也增长 16 倍，达到 compute bound 的 batch size 临界点大大缩小
  - **当 batch size 超越临界点后，性能对比自回归会下降**
  - Medusa 论文自己也提到大流量下性能可能劣化

## 7. 与后续方法对比

| 维度 | Medusa | EAGLE-1 | MTP |
|---|---|---|---|
| head 输出 | 每个头**独立**预测不同 offset | AR head 链式生成 | 因果链式 module |
| 位置间依赖 | 无（各自看不到别人） | 有（AR） | 有（因果链） |
| 接受率 | 中低 | 高 | 中高 |
| 加速 | ~2.2× | 2-3× | ~1.8× |

**Medusa 的根本瓶颈**：并行头之间的独立性，直接催生了 EAGLE 的"特征级自回归"设计。

## 相关文档

- 前置：[原理基础](/docs/inference-engines/optimization/speculative-decoding/basics)
- 演进：[EAGLE 系列](/docs/inference-engines/optimization/speculative-decoding/eagle) 继承了自我投机思路，用 AR head 补上位置间依赖

## 参考

- Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, 2024
- 内网原文：[投机采样算法调研 (iWiki)](https://iwiki.woa.com/p/4015805555)
