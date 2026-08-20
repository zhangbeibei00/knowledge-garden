---
sidebar_position: 6
title: DFlash（Block Diffusion）
---

# ⚡ DFlash: Block Diffusion for Flash Speculative Decoding

> **ICML 2026, UCSD, Z Lab**
> 第一个把扩散模型系统性地用作投机解码"起草引擎"的工作。用 block diffusion 打破"draft 必须逐词"的假设，单次 forward 出整块 token，Qwen3-8B 上 4-6× speedup。

## 🎯 面试速记版

> **一句话**：DFlash 用一个 5~8 层的**块扩散（block diffusion）模型**当 drafter，在**一次双向 attention 的 forward** 中并行生成一整块 (block_size=16) token，target 模型的多层 hidden 通过 **KV 注入**（不是输入拼接）在**每一层**都作为条件信号送进 draft 的 attention，最后 target 一次并行验证。

> **三大设计支柱**：
> 1. **Block diffusion drafting**：draft 阶段从 γ 次串行 forward 压缩到 **1 次并行 forward**，$T_{draft}$ 与 γ 解耦
> 2. **KV Injection 代替 Input Fusion**：target hidden 不进 input，而是通过 draft 每层 attention 的 K/V 塞进去。信号不随深度衰减，让 draft **敢加深到 5-8 层**
> 3. **Context features from target**：抽 target 的 5 层均匀分布的 hidden（层 1, 9, 17, 25, 33 for 36 层 Qwen3-8B），concat → fc 融合成 1 份 → 广播给 draft 每层

> **实测 speedup**：Qwen3-8B 单并发 **4.6-6.1×**（EAGLE-3 只有 ~2×），Qwen3-Coder-30B-A3B 上 3.5×

> **核心洞察**："The Target Knows Best" —— 大 AR 模型的 hidden features 已经隐式编码了未来多个 token 的信息（借鉴 Samragh et al., 2025），所以 draft 只需要做一个**轻量级 diffusion adapter**：以 target 的中间层 hidden 为条件，去并行预测下一个 block

> **痛点**：块内 token 并行独立生成，**缺乏因果依赖建模** → 可能出现"of course + no problem"式的缝合怪草稿 → 被 DSpark 命名为 "suffix decay"，也催生了 DSpark 的半自回归架构

---

## 1. 背景与动机

### 1.1 投机解码速度公式

$$
L = \frac{T_{\text{draft}} + T_{\text{verify}}}{\tau}, \qquad \eta = \frac{L_{\text{target}}}{L}
$$

- $\tau \in [1, \gamma+1]$：每 cycle 实际接收的 token 数
- 提速来自：**↑ τ** 或 **↓ T_draft**

### 1.2 自回归 draft 的瓶颈

EAGLE-3 等主流方法仍是自回归 draft：

$$
T_{\text{draft}} = \gamma \cdot t_{\text{step}}
$$

为控制 $T_{\text{draft}}$，draft 模型只能做得很浅（EAGLE-3 默认 1 层 Transformer）。**γ 越大 draft cost 线性涨，但 τ 因模型容量不足很快饱和 → 实际加速被卡死在 2-3×。**

### 1.3 扩散语言模型的诱惑与困境

- **优点**：block diffusion 一次 forward 并行去噪整块 mask token，$T_{\text{draft}} = t_{\text{parallel}} \ll \gamma \cdot t_{\text{step}}$
- **缺点**：现有开源 dLLM 质量远不如 AR LLM，且常需多步去噪才能保质量

已有的扩散投机解码（DiffuSpec / SpecDiff-2）用 **7B 级别**的扩散 drafter，draft 质量尚可但内存与 latency 都太重，实际只能做到 3-4× speedup。

### 1.4 DFlash 的核心 insight："The Target Knows Best"

> 借鉴 Samragh et al. (2025)：大 AR 模型的 hidden features 已经隐式编码了未来多个 token 的信息。

因此与其让小 drafter "从零推理"，不如让它做一个**轻量级 diffusion adapter**——以 target 的中间层 hidden 为条件，去并行预测下一个 block。

---

## 2. 系统总览

### 2.1 推理时一个 cycle 的流程

```
Prompt + 已接收前缀
    │
    ▼
Target forward → logits + 多层 hidden
    │            │
    │            └── 抽 5 层 hidden + fc → target_hidden
    │
    ├── sample bonus token → block: [anchor + N×MASK]
    │                              │
    ▼                              ▼
DFlash Draft (5~8 层，一次 forward，双向)
    │
    ▼
共享 target lm_head → N 个 draft token
    │
    ▼
Target verify 并行
    │
    ▼
最长一致前缀 → accept k 个 + bonus → 下一个 cycle
```

**关键点**：
- Draft 一次 forward 出 16 个 token（block size = 16），而不是 16 次串行
- **Verify 阶段顺便给出下一轮的 target_hidden**，不需要额外一次 target forward
- 已接收前缀 + bonus token 作为下一个 block 的 anchor

### 2.2 三大设计支柱

| 支柱 | 解决什么 | 怎么做 |
|---|---|---|
| **1. Context features from target** | draft 模型容量小，"独立预测未来"会失败 | 抽 target 5 个均匀分布层 hidden，concat → fc → RMSNorm |
| **2. KV injection conditioning** | 仅在 input 融合（EAGLE-3 风格），depth ↑ 后信息被稀释 | 把 fused hidden 投到**每一个 draft layer 的 K/V**，与 noise token 拼接并 cache |
| **3. Block diffusion drafting** | AR draft 必须串行 → drafting cost 不可加深 | 一个 block 内所有 mask 位置在同一次 forward 里 attend 双向 self-attention 并行解码 |

---

## 3. KV 注入：DFlash 最关键的创新

### 3.1 对比：Input Fusion (EAGLE-3) vs KV Injection (DFlash)

**Input Fusion（EAGLE-3）**：

```
输入 layer 0:
  [target_hidden(D) | embed(token)(D)]  ← 拼成 2D 维
              ↓
      q/k/v_proj (in_dim=2D=8192)
              ↓
        进入 attention → FFN
              ↓
      layer 1 (信息已经被打散混合)
```

**KV Injection（DFlash）**：

```
每一层 draft attention:
    Q     ← q_proj(hidden_states)     只从 draft 自己算
    K_ctx ← k_proj(target_hidden)     从 target 算
    V_ctx ← v_proj(target_hidden)     从 target 算
    K_noise ← k_proj(hidden_states)   从 draft 自己算
    V_noise ← v_proj(hidden_states)   从 draft 自己算

    K = concat(K_ctx, K_noise)
    V = concat(V_ctx, V_noise)
    attention(Q, K, V) → 下一层
```

**每一层都重复这个过程，target 信号永远新鲜。**

### 3.2 为什么 KV 注入更强？

| 角度 | 输入拼接 (EAGLE-3) | KV 注入 (DFlash) |
|---|---|---|
| target 信号注入位置 | 只在 layer 0 | **每一层** |
| 信息衰减 | 经过 N 层后被反复线性变换、稀释 | 每层重新读，**永远新鲜** |
| 加深 draft 的收益 | 几乎不涨 τ（层数一多信号丢了）| τ 随深度稳定 scale |
| 参数量 | q/k/v_proj 输入 2D，参数多一倍 | q/k/v_proj 输入 D，参数省一半 |

**打比方**：
- 输入拼接 = 老师把答案在**开头**告诉学生一次，之后学生自己往下推理，走 5 层后已经忘得差不多了
- KV 注入 = 老师把答案卡片**放在桌上**，学生每一步思考时都能瞄一眼，永远新鲜

论文消融（Table `tab:ablation_kv_injection`）：**即便 draft 是纯自回归的（不做 diffusion），把 input fusion 换成 KV injection 也能稳稳提高 τ。**

### 3.3 特别澄清：不是"复用 target 的 KV cache"

论文名字叫 "KV injection" 很容易让人以为是"把 target 的 KV cache 直接拿给 draft 用"。**不是的。**

| 维度 | 假想中的"复用 target KV cache" | DFlash 实际做的 KV 注入 |
|---|---|---|
| 注入的原料 | target 的 K/V（已经过 q/k/v_proj） | target 的 **hidden**（attention 之前的中间表示） |
| 投影矩阵归谁 | target 的 k/v_proj | **draft 自己**的 k/v_proj |
| 形状 | `(B, 32H, T, 128)` × 36 层 | `(B, 8H, T, 128)` × draft 层数 |
| 谁做的投影 | target 现成的 | draft 现场算 |

**关键**：DFlash 拿的是 target 的 **hidden**（注意，不是 KV），然后用 **draft 自己的** `k_proj_i / v_proj_i` 重新投一遍，得到属于 draft 的 K_ctx/V_ctx。draft 和 target 各自维护独立 KV cache，互不共享。

论文用的是**位置语义**："这份信号被塞进了 attention 的 **K/V 位置**"，不是"来自 target 的 KV cache"。

### 3.4 5 层 target hidden ≠ 一一对应 5 层 draft

```
target 36 层
   ↓ 抽 5 层 [1, 9, 17, 25, 33]，concat 成 (B, T, 5*D)
   ↓ fc (5D → D) + RMSNorm    ← 融合成 1 份，只做 1 次
   ↓
共享的 target_hidden (B, T, D)   ← 广播给 5 层 draft
   ↓
draft 每一层用自己独立的 k/v_proj 从这份 target_hidden 里投出自己需要的 K_ctx/V_ctx
```

- 5 层 target hidden 在**入口处融合成一份**共享特征
- draft 5 层通过**各自独立的 k/v_proj** 把这份共享特征"解读"成各自视角的 KV

论文原话："The projected target features are **shared by all draft layers**."

---

## 4. 训练策略

### 4.1 Anchor 采样 + Flex Attention BlockMask

训练一条样本（长度 S）被组织为：

```
KV: [ Context (S 个 clean token) | Block_0 | Block_1 | ... | Block_{n-1} ]
Q : [                            | Block_0 | Block_1 | ... | Block_{n-1} ]
```

- 每个 block 长度 = block_size，第 0 位是从 response 中随机采样的 **anchor token**（clean），后面是 `<MASK>`
- 不同 block 之间**互不可见**（防止信息泄漏）
- 每个 block 只能看：位置严格小于自己 anchor 的上下文 + 自己 block 内部所有位置（双向）

用 Flex Attention 的 **BlockMask** 把这种稀疏模式编译进 Triton kernel，**一次 forward 训 N 个 block**，等价于 EAGLE 训练里的 N 次 forward。

### 4.2 Loss 加权（Eq. 4）

```python
# 论文 Eq.4：early position 权重大
decay_weights = exp(-(k - 1).clamp(min=0) / loss_decay_gamma)
```

**核心直觉**：block 内 token 不平等。block 第 1 个 token 错了就直接 reject，后面所有预测都白做。所以早期位置的正确率应该被显著加权：

$$
w_k = \exp\left(-\frac{k-1}{\gamma}\right)
$$

γ = 7（block 16）/ 5（block 10）/ 4（block 8），位置 0 权重最大，越靠后越小。

### 4.3 训练超参对比 EAGLE-3

| 超参 | DFlash | EAGLE-3 (SpecForge) |
|---|---|---|
| Learning rate | **6e-4** | 1e-4 |
| Num epochs | **6** | 10 |
| Max length | **3072** | 2048 |
| Warmup ratio | **0.04** | 0.015 |
| Max grad norm | **1.0** | 0.5 |

DFlash 用大 6× 的 lr 跑短 epoch，因为：
- 多 anchor block 天然平均了梯度方差 → 可承受大 lr
- Loss decay 集中信号到前几位 token
- Frozen embed + shared lm_head，可训参数少，需要大 lr 补偿

---

## 5. 实测性能

### 5.1 Qwen3 系列（H200, Transformers, greedy）

| 方法 | 平均 speedup |
|---|---|
| DFlash | **4.9×** |
| EAGLE-3(16) | 2.0× |
| EAGLE-3(60) | ~2.3× |

### 5.2 SGLang on B200, FA4, Spec-v2

| 模型 | 单并发 speedup | 32 并发 speedup | τ |
|---|---|---|---|
| Qwen3-4B / Math500 | 4.8× | 2.9× | 8.01 |
| Qwen3-8B / Math500 | **5.1×** | 2.8× | 8.01 |
| Qwen3-Coder-30B-A3B / HumanEval | 3.5× | 3.1× | 8.09 |

### 5.3 严格同数据对比：LLaMA-3.1-8B（与 EAGLE-3 同训练数据）

| 任务 | EAGLE-3(10) | EAGLE-3(60) | **DFlash(b=10)** |
|---|---|---|---|
| GSM8K | 1.6× | 1.9× | **2.4×** |
| HumanEval | 2.0× | 2.0× | **2.8×** |
| Alpaca | 1.5× | 1.8× | **2.2×** |

→ 同数据、同 target，DFlash 仍快约 1.4-1.5×。**DFlash 的 win 主要来自架构，而不是更多数据。**

---

## 6. 缺陷：Suffix Decay

DFlash 的并行生成范式暴露出一个结构性缺陷：**块内的 token 是并行、独立生成的，缺乏彼此之间的因果依赖建模**。

**举例**：草稿模型可能在某个位置独立地觉得 "of course" 和 "no problem" 都合理，但由于各位置分开打分、独立采样，可能把两个"分开看都对、拼在一起却别扭"的词拼接在一起，出现前后不搭的"缝合怪"草稿。

这个问题后来被 DeepSeek 的 DSpark 论文明确点名，称为 **"suffix decay"（后缀衰减）**。DSpark 用**半自回归架构**来解决这个问题。

---

## 7. 为什么 DFlash 加速效果如此突出

### 7.1 投机解码速度方程的双侧优化

| 方法 | $T_{draft}$ | τ |
|---|---|---|
| EAGLE-3 (1L, γ=7) | 7 × t_step | 3.5 |
| EAGLE-3 (1L, γ=60) tree | 7 × t_step + tree | 4.5 |
| DFlash (5L, block=16) | 1 × t_parallel | **8.0** |

- EAGLE-3 想 ↑ τ → 必须 ↑ γ → $T_{draft}$ 直接线性涨
- DFlash 想 ↑ τ → 加 layer 或加 block_size → $T_{draft}$ 几乎平的

→ DFlash 处在 (T_draft, τ) 的**更优 Pareto 前沿**。

### 7.2 GPU 利用率：单 forward 多 token

EAGLE-1 forward 处理 1 个 query，attention 和 FFN 都是严重 memory-bound（典型 < 5% peak FLOPs）。DFlash 一次 forward 处理 16 个 query：
- attention 的 K, V 复用，QKV proj 直接 batched matmul
- FFN 在 16 token 上批处理，更接近 compute-bound
- 加深 layer 数也只是多走几遍，单步代价基本不变

### 7.3 训练侧的对齐

- **Anchor 采样**：训练数据分布 ≡ 推理数据分布（每个 block 第 0 位永远是 clean token）
- **同 forward 同 mask 同 RoPE**：训练几乎没有 train-test gap
- **Loss decay**：直接优化 acceptance length 的代理指标
- **Frozen embed + LM head**：draft 学的是 target 表征空间的"邻域映射"，不是另起炉灶

---

## 相关文档

- 前置：[原理基础](/docs/inference-engines/optimization/speculative-decoding/basics) / [EAGLE 系列](/docs/inference-engines/optimization/speculative-decoding/eagle)
- 后续：[DSpark](/docs/inference-engines/optimization/speculative-decoding/dspark)（用半自回归解决 DFlash 的 suffix decay 问题）

## 参考

- Chen, Liang & Liu, *DFlash: Block Diffusion for Flash Speculative Decoding*, ICML 2026
- 主页：https://dflash.z-lab.ai
- 代码：https://github.com/z-lab/dflash
- 训练：https://github.com/sgl-project/SpecForge
- Samragh et al., *Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential*, 2025
- 内网原文：[DFlash Speculative Decoding (iWiki)](https://iwiki.woa.com/p/4021437783)
