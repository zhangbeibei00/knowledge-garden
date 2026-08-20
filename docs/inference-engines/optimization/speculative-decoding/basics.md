---
sidebar_position: 2
title: 原理基础
---

# 📐 投机解码原理基础

> 投机解码的核心不是"小模型替代大模型"，而是"小模型便宜地猜 + 大模型并行验证 + 拒绝采样修正分布"。
> 最终输出分布**严格等于**大模型原始分布——这叫无损（lossless）加速。

## 🎯 面试速记版

> **一句话解释投机解码**：
>
> 让一个轻量草稿模型（draft）先自回归地猜 γ 个 token，让目标大模型（target）一次并行 forward 同时验证这 γ 个位置的概率，通过 **modified rejection sampling** 决定接受哪几个。数学上可以证明：最终生成的 token 序列分布**严格等于**只用大模型逐词采样的分布，是**无损加速**。

> **加速原理**：$L = (T_{draft} + T_{verify}) / \tau$。大模型的一次 forward 从"只产 1 个 token"变成"验证并接受 $\tau \geq 1$ 个 token"（$\tau$ 是每轮平均接受长度）。核心是**把大模型的多次串行 forward 压缩到一次并行 forward**。

> **为什么能无损**：接受概率 $\alpha = \min(1, p/q)$；被拒绝后从 residual 分布 $\text{norm}(\max(0, p-q))$ 采样替代；全接受时用 bonus token（$p_{\gamma+1}$）。三部分相加恰好等于 $p$。

---

## 1. 为什么能加速？

传统自回归解码：

> 大模型每次 forward 只生成 1 个 token

投机解码：

> 小模型先生成 γ 个草稿 token
> 大模型一次 forward 同时验证这些 token
> 如果草稿命中率高，一轮可以接受多个 token

因此，投机解码**减少了大模型的串行调用次数**。关键收益来自：

- 小模型生成草稿成本低
- 大模型验证多个位置可以并行（GPU 更 compute-bound、更充分利用）
- 一次大模型 forward 可能产出多个 token

---

## 2. 基本流程

设当前上下文是 `prefix`，草稿长度是 γ。

### 阶段 1：draft

小模型按自回归方式生成草稿 token：

```
y1, y2, ..., yγ
```

并保存每个位置的小模型概率分布：

```
q1, q2, ..., qγ
```

### 阶段 2：verify

大模型一次性输入：

```
prefix + y1 + y2 + ... + yγ
```

输出多个位置的大模型概率分布：

```
p1, p2, ..., pγ, pγ+1
```

其中：

- `p1` 用来验证 `y1`
- `p2` 用来验证 `y2`
- ...
- `pγ` 用来验证 `yγ`
- `pγ+1` 用于全接受时采样 bonus token

### 阶段 3：accept / reject

从左到右逐 token 判断：

- 如果 `yi` 被接受，继续判断 `yi+1`
- 如果 `yi` 被拒绝，从 residual 采样 replacement，然后**丢弃后续所有草稿 token**
- 如果 `y1~yγ` 全部接受，再从 `pγ+1` 采样 bonus token

---

## 3. 核心原则：分布无损，而非"猜中大模型的选择"

投机解码的核心**不是**判断：

> 小模型猜的 token 是不是大模型最想要的 token？

而是判断：

> 接受这个 token 后，整体采样分布还能不能严格等于大模型分布？

因此，投机解码关注的是 **分布无损**，不是每一步都选大模型 top1。

---

## 4. 接受概率 $\alpha = \min(1, p/q)$ 的推导

设：

- $q(x)$：小模型认为 token $x$ 的概率
- $p(x)$：大模型认为 token $x$ 的概率

小模型先按 $q$ 抽出 token $x$，然后用接受概率：

$$
\alpha(x) = \min\left(1, \frac{p(x)}{q(x)}\right)
$$

最终通过"被小模型抽中 + 被接受"贡献给 $x$ 的概率是：

$$
q(x) \cdot \alpha(x) = \min(q(x), p(x))
$$

也就是说，接受阶段最多只保留大模型允许的概率质量，不会让某个 token 出现概率超过 $p(x)$。

### 为什么只是 top-5 也能被接受？

因为在 sampling 解码里，大模型不是只允许 top1 出现。

比如大模型分布：

| token | 概率 | 排名 |
|---|---|---|
| A | 0.30 | 1 |
| B | 0.25 | 2 |
| C | 0.18 | 3 |
| D | 0.12 | 4 |
| E | 0.08 | 5 |

`E` 虽然只是第 5 大，但正常大模型采样时，它仍然有 8% 概率出现。所以投机解码不会因为 `E` 不是 top1 就拒绝它，而是判断：

> 接受 E 后，E 的最终出现概率是否仍然等于大模型给出的 0.08？

如果能保持这个分布，就可以接受。

---

## 5. Residual 分布（拒绝时的替代采样）

如果某个草稿 token 被拒绝，需要从残差分布采样 replacement：

$$
\text{residual} = \text{normalize}(\max(0, p - q))
$$

含义：

- $p - q > 0$：大模型比小模型更想要这些 token，小模型覆盖不够 → 需要补
- $p - q \leq 0$：小模型已经覆盖过多，不再补

所以 residual 只补齐小模型没覆盖够的概率质量。

### 采样示例

| token | $p$ | $q$ | $\max(0, p-q)$ |
|---|---|---|---|
| A | 0.35 | 0.50 | 0 |
| B | 0.25 | 0.10 | 0.15 |
| C | 0.20 | 0.05 | 0.15 |
| D | 0.15 | 0.25 | 0 |
| E | 0.05 | 0.10 | 0 |

残差总和：`0.15 + 0.15 = 0.30`，归一化后：`B: 0.5, C: 0.5`。

所以拒绝后，replacement 会以 50% 概率采样 `B`，50% 概率采样 `C`。

代码形式：

```python
residual = p_dist - q_dist
residual = torch.clamp(residual, min=0.0)
residual = residual / residual.sum()
replacement = torch.multinomial(residual, num_samples=1)
```

### 为什么不能拒绝后直接从 $p$ 采样？

因为接受阶段已经保留了一部分概率质量 $\min(p, q)$。拒绝后只应该补剩下的部分：

$$
p - \min(p, q) = \max(0, p - q)
$$

如果直接从完整 $p$ 采样，会导致部分 token 的概率被重复计算，最终分布不再等于大模型分布。

---

## 6. Bonus Token

假设草稿长度 $\gamma = 4$，小模型生成 $y_1, y_2, y_3, y_4$。

大模型 verify 时一次 forward 会得到：

- $p_1, p_2, p_3, p_4$：分别验证 $y_1 \sim y_4$
- $p_5$：prefix + $y_1 + y_2 + y_3 + y_4$ 之后的下一个 token 分布

如果 $y_1 \sim y_4$ **全部接受**，那么 $p_5$ 正好就是大模型下一步本来要用的分布。所以可以直接从 $p_5$ 再采样一个 token——这就是 bonus token。

它的作用是：

> 一轮最多从 γ 个 token 提升到 **γ + 1** 个 token

这是"白赚"的，因为 $p_5$ 已经在 verify 阶段算出来了。**如果中间发生拒绝，则不能使用 bonus**，因为后续路径已经变了。

---

## 7. 分布无损的最终证明

投机解码通过三步保证无损：

- **接受阶段**：保留 $\min(p, q)$
- **拒绝阶段**：补齐 $\max(0, p - q)$
- **两者相加**：$\min(p, q) + \max(0, p - q) = p$ ✅

所以它不是让小模型替代大模型，而是：

> 小模型负责便宜地猜；
> 大模型负责并行验证；
> 拒绝采样负责把最终分布校正回大模型分布。

---

## 8. 加速比公式

$$
L = \frac{T_{\text{draft}} + T_{\text{verify}}}{\tau}, \qquad \eta = \frac{L_{\text{target}}}{L}
$$

- $\tau \in [1, \gamma+1]$：每 cycle 实际接收的 token 数（含 bonus token）
- 提速来自两条路：**↑ τ**（draft 越准）或 **↓ T_draft**（draft 越快）

**这两条路对应了后续所有投机解码方法的分野**：

- EAGLE 系列 / MTP → 主打 **↑ τ**（feature 级 draft、动态树、TTT）
- DFlash / DSpark → 主打 **↓ T_draft**（block diffusion，一次 forward 出整块）

---

## 9. 相关文档

- 前置：[投机解码技术编年史](/docs/inference-engines/optimization/speculative-decoding/chronicle)
- 方法演进：[Medusa](/docs/inference-engines/optimization/speculative-decoding/medusa) → [EAGLE 系列](/docs/inference-engines/optimization/speculative-decoding/eagle) → [MTP](/docs/inference-engines/optimization/speculative-decoding/mtp) → [DFlash](/docs/inference-engines/optimization/speculative-decoding/dflash) → [DSpark](/docs/inference-engines/optimization/speculative-decoding/dspark)

## 参考

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023 Oral
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023
- 内网原文：[投机解码（speculative_decoding）](https://iwiki.woa.com/p/4024269073)
