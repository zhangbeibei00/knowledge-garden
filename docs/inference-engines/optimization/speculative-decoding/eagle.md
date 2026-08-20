---
sidebar_position: 4
title: EAGLE 系列
---

# 🦅 EAGLE 1/2/3

> EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）由北大团队提出，从 EAGLE-1（ICML'24）到 EAGLE-3（NeurIPS'25），是**过去两年投机解码领域的标杆**。
> EAGLE-3 已经成为 vLLM、SGLang 等主流推理框架的"标配"投机解码方案。

## 🎯 面试速记版

> **EAGLE-1**：
> - 核心洞察：**不猜 token，猜 feature**（在特征空间自回归）。复用 target 倒数第二层的 hidden state 作为草稿模型的输入，训练一个单层 Transformer 的 AR head 外推下一个特征，再过共享 LM head 得到 token。
> - 关键工程细节：把**当前 token 的 embedding** 也拼进 AR head 的输入 → 消除特征空间的分叉不确定性
> - 加速 2-3×，训练成本极低（8×3090 训 1-2 天）

> **EAGLE-2**：
> - 核心洞察：草稿小模型是 **well-calibrated** 的——它自己的置信度分数就能预测大模型的接受率
> - 引入**上下文感知的动态草稿树**：动态扩展（按累计路径概率 $V_i$ 挑分支）+ 重排序（保留 top-K 拼成验证序列）
> - 无需额外训练，比 EAGLE-1 再快 20-40%

> **EAGLE-3**：
> - 核心洞察：feature-level 预测有语义天花板，撞到 draft 模型自己的 scaling law
> - 两项改动：
>   1. **改回 next-token 预测**（放弃 feature 预测）
>   2. **多层特征融合**：拼接浅层（词法）+ 中层（句法）+ 深层（语义）特征作为条件
> - **Training-Time Test (TTT)**：训练时模拟推理的多步 rollout，消除 train-test gap
> - 突破 draft 模型的 scaling law，最高 6.5× 加速；已成为 vLLM/SGLang 标配

---

## EAGLE-1（ICML 2024）

### 两个核心 insight

**① Feature-level vs Token-level**

用 token 的隐藏层特征（feature level）预测 next token，比直接用 token embedding（token level）**更简单**：

- `token embedding` 只是简单转换，没有经过深层网络提取特征，表达能力不足
- `feature`（LLM 最后一层 TransformerLayer 的输出，LM head 的输入）是深层特征，表达能力强，隐藏大量 dark knowledge

因此在特征层做自回归预测、再通过 LM head 得到 token，比直接预测 token 更容易。

**② 融合当前 token 的 embedding**

采样算法的不确定性/随机性会限制 next token 生成性能：

> 不同的 next token "am"和"always"会走向完全不同的 next-next token

隐藏层特征的生成非常依赖采样结果。为了提高稳定性，EAGLE 在生成当前 token 的隐藏层特征时，**融合当前 token 的 embedding 以及上一个 token 的隐藏层特征**，保证 draft token 生成时具备足够信息。

### 架构

```
输入： prev_token_hidden + curr_token_embedding
              │  (concat)
              ▼
       ┌─────────────┐
       │  FC layer   │
       └─────┬───────┘
             ▼
       ┌─────────────┐
       │  1 Transformer Layer  │  ← AR Head（唯一需要微调的部分）
       └─────┬───────┘
             ▼
       LM Head（复用 target 的）
             │
             ▼
        next token
```

- Embedding 层和 LM Head 均**复用原始 target**
- 中间的 One Auto-regression Head（AR Head）由一层 FC + 一层 Transformer Layer 组成
- **AR Head 是唯一需要微调的部分**，训练成本极低

### Draft Tree

采用 tree attention。与 Medusa 的静态笛卡尔积不同：

- 根据不同 draft token，生成它们的子节点
- 天然去掉无用分支，缩短验证序列长度
- 路径中节点的相关性更高

**draft tree 是静态的**，作为一种先验。高概率 token 的分支应该更宽更深。

### 优缺点

**优点**：
- ✅ 相比 Medusa 直接拉多个头预测 token prob，让输出预测 last-layer feature，可以很好提升接受率
- ✅ 避免通过笛卡尔积生成 draft 树，验证序列长度大大减少
- ✅ 在更大流量下仍可提升推理速度

**缺点**：
- ❌ 不适用于 MoE 模型（MoE 在验证阶段需要读更多专家权重）
- ❌ batch size 增加时加速比降低，限制了在高吞吐量场景下的应用
- ❌ 静态树的结构不适合语境动态变化的场景

---

## EAGLE-2（EMNLP 2024）

### 两个 insight

**① 接受率不仅依赖位置，还依赖语境**

如果语境是动态变化的，那么 EAGLE-1 中的**静态树结构无法适应**。

**② Draft 模型的置信度分数与接受率强正相关**

可以用置信度分数近似接受率，**无需模型额外推理**。

### 动态草稿树：Expand + Rerank

```
Expand 阶段（生成 draft tree）：
  每个节点包含路径的累积概率 V_i（路径权重连乘）
  V_i < 阈值 → 停止展开
  V_i ≥ 阈值 → 继续展开

Rerank 阶段：
  所有叶子和非叶子节点参与重排
  保留概率最高的 K 个节点
  相同概率时保留浅层节点（保证树的结构）
  → 展开成 1 维序列作为验证阶段输入
  → 更新 mask 矩阵
```

**「好猜的地方多猜，难猜的地方少猜」**

### 优点

- ✅ 无需额外训练，只需根据置信度分数调整 draft tree
- ✅ 严格保证文本分布与原始 LLM 完全一致
- ✅ 相比 EAGLE-1 提速 20-40%

---

## EAGLE-3（NeurIPS 2025）

### 三个改动

**① 改回 next-token 预测**

训练数据扩增后，feature-level 预测限制了 draft 模型的表达能力，无法从 scaling law 中获益。所以又改回了 next-token 预测。

**② 多层特征融合**

不再只用倒数第二层的 feature：

- **浅层 feature**（词法信息）
- **中层 feature**（句法信息）
- **深层 feature**（语义信息）

三者拼接融合后喂给 draft 模型，补齐信息。

**③ Training-Time Test (TTT)**

传统训练用 teacher-forcing（每步都喂标准答案），导致推理时连续套娃生成草稿时，微小误差会累积。EAGLE-3 修改微调任务：

> 把每一步模型推理时产生的 feature，添加到下一步的输入中，而非全部使用目标模型产生的 feature，从而让模型在训练时便适应新的分布

也就是**在训练阶段就模拟推理时的多步自回归起草过程**。

### 突破 scaling law

EAGLE-3 的 draft 模型终于**迎来了属于自己的 scaling law**，能随数据量变大而持续变强。

**完美向下兼容 EAGLE-2 的动态草稿树**，两者可以无缝叠加。

**加速比最高 6.5×**。

### 场景定制

可以针对垂直场景使用特定数据集训练专属草稿模型，只需要很少的训练资源即可获得推理加速提升。

**目前 EAGLE-3 已经成为 vLLM、SGLang 等主流推理框架里的"标配"投机解码方案。**

---

## 三代对比

| 维度 | EAGLE-1 | EAGLE-2 | EAGLE-3 |
|---|---|---|---|
| 草稿目标 | 预测 feature | 预测 feature | **预测 token** |
| 条件输入 | 倒数第二层 feature + curr token emb | 同 EAGLE-1 | **浅+中+深多层 feature** |
| Draft tree | 静态 | **动态**（expand + rerank） | 动态 |
| 训练 | AR Head SFT | 无需额外训练（复用 EAGLE-1 权重） | + **TTT (Training-Time Test)** |
| 加速比 | 2-3× | 比 EAGLE-1 快 20-40%，最高 5× | 比 EAGLE-2 快 ~1.4×，最高 6.5× |
| Scaling law | ❌ | ❌ | ✅ |

## 与其他方法对比

| 方法 | Draft 输出 | 位置依赖 | 加速 |
|---|---|---|---|
| Medusa | 多头独立预测 | 无 | ~2× |
| **EAGLE-1** | feature AR | 有 | 2-3× |
| **EAGLE-2** | feature AR + 动态树 | 有 | 最高 5× |
| **EAGLE-3** | token AR + 多层 feature + TTT | 有 | 最高 6.5× |
| MTP | 预训练内置 | 有（因果链） | ~1.8× |

## 相关文档

- 前置：[原理基础](/docs/inference-engines/optimization/speculative-decoding/basics) / [Medusa](/docs/inference-engines/optimization/speculative-decoding/medusa)
- 竞品：[MTP](/docs/inference-engines/optimization/speculative-decoding/mtp)（预训练内置）
- 后续：[DFlash](/docs/inference-engines/optimization/speculative-decoding/dflash)（用 KV 注入代替 EAGLE-3 的输入拼接）

## 参考

- Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*, ICML 2024
- Li et al., *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*, EMNLP 2024
- Li et al., *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*, NeurIPS 2025
- 内网原文：[EAGLE-3 精读笔记 (iWiki)](https://iwiki.woa.com/p/4023076508) / [投机采样算法调研 (iWiki)](https://iwiki.woa.com/p/4015805555)
