---
sidebar_position: 2
title: LLaDA 方法解析 · 一篇讲透
description: LLaDA 的前向加噪、反向去噪、训练目标、推理调度（三种 remask 策略）、SFT 处理、和自回归的具体差异，配伪代码
tags: [llada, diffusion-lm, masked-diffusion, training, inference, remasking]
---

# LLaDA 方法解析 · 一篇讲透

**核心一句话**：LLaDA = **Masked Discrete Diffusion** —— 把语言生成建模成"从全 [MASK] 逐步填空到完整句子"的**离散扩散过程**，用**双向 Transformer** 做去噪器。

这篇按下面这条脉络展开：

```
1. 三层方法总览
2. 前向过程: 独立位置的掩码扩散
3. 反向过程: 双向 Transformer 做去噪
4. 训练目标: (1/t) 加权的 masked CE
5. 推理过程: 迭代填空 + 置信度调度
6. 三种 remask 策略
7. Prompt 保护与 SFT
8. 和自回归的具体对比
9. 方法层面的创新点
```

---

## 一、三层方法总览

```
┌──────────────────────────────────────────────────────────┐
│  Layer 1 · 概率建模: 前向加噪 + 反向去噪                 │
│     - 前向: 按 t ∈ [0,1] 独立 mask 每个 token            │
│     - 反向: Transformer 从 mask 恢复原 token             │
│                                                          │
│  Layer 2 · 训练: Masked Cross-Entropy Loss               │
│     - 随机采样掩码率 t ~ Uniform(0, 1)                   │
│     - 按 t 概率独立 mask, 预测被 mask 位置               │
│     - (1/t) 加权归一化,是 NLL 的上界                     │
│                                                          │
│  Layer 3 · 推理: 迭代去噪 + 置信度调度                   │
│     - 从全 mask 起步                                     │
│     - 每步:模型预测 → 保留高置信位置 → 剩下 remask       │
│     - 迭代 N 步得到完整输出                              │
└──────────────────────────────────────────────────────────┘
```

---

## 二、前向过程：独立位置的掩码扩散

给定原始序列 x₀ = (x₀¹, x₀², ..., x₀^L)，定义连续时间 t ∈ [0, 1]。

**前向加噪的转移概率**（每个位置独立）：

```
q(x_t^i | x_0^i) = {
    t,      如果 x_t^i = [MASK]     ← 有 t 概率被 mask
    1-t,    如果 x_t^i = x_0^i       ← 有 1-t 概率保留
}
```

### 关键性质

| 时间 | 状态 |
|------|------|
| t = 0 | 完全无噪声，等于原序列 |
| t = 0.5 | 平均一半 token 被 mask |
| t = 1 | 完全掩码，等于全 [MASK] |

**每个位置独立采样** —— 不像连续扩散有跨位置相关性。

### 和 BERT-MLM 的关键区别

| | BERT-MLM | LLaDA |
|---|---------|-------|
| 掩码率 | 固定 15% | **t ~ Uniform(0, 1)** 均匀采样 |
| 用途 | 只做表示学习 | **生成**（从全 mask 到完整句子） |

**这就是 LLaDA 最核心的方法差异之一** —— 让模型见过所有强度的掩码，才能承担从 100% mask 到 0% mask 的完整生成任务。

### 前向过程可视化

```
原序列 x_0:   [今, 天, 天, 气, 很, 好]

t=0.3 (30% mask):
x_t:          [今, [M], 天, [M], 很, 好]

t=0.6:
x_t:          [[M], [M], 天, [M], [M], 好]

t=1.0:
x_t:          [[M], [M], [M], [M], [M], [M]]  ← 全 mask
```

---

## 三、反向过程：双向 Transformer 做去噪

去噪器是一个**双向 Transformer**（不加 causal mask），任务是：

给定被 mask 的序列 x_t 和时间 t，**并行预测所有 [MASK] 位置的原始 token**：

```
p_θ(x_0^i | x_t, t)     对所有 x_t^i = [MASK] 的位置
```

### 和自回归 Transformer 的架构差异

| 组件 | 自回归 (GPT/LLaMA) | LLaDA |
|------|---------------------|-------|
| **Attention Mask** | Causal (下三角) | **Bidirectional (无 mask)** |
| **输入** | 完整历史 token | 部分 [MASK] 的序列 |
| **一次预测** | 只预测下一个 token | **并行预测所有 mask 位置** |
| **位置编码** | 支持 KV Cache 增量 | 每步都要**全序列前向** |
| **RoPE / SwiGLU / RMSNorm** | ✅ | ✅ 完全一样 |
| **参数量** | 8B | 8B (LLaDA-8B) |

**除了 attention mask 变了以外，Transformer 结构本身完全一样** —— 这是 LLaDA 一个非常聪明的地方：**可以复用所有现有 Transformer 组件**。

---

## 四、训练目标：(1/t) 加权的 masked CE

### 4.1 数学定义

```
L(θ) = E_{x_0, t, x_t} [  (1/t) · Σ_i 𝟙[x_t^i = MASK] · (-log p_θ(x_0^i | x_t))  ]
                         └───┘   └──────────────────┘   └────────────────────────┘
                       归一化    只算被 mask 的位置       预测原 token 的 log-prob
```

**三个关键组件**：
1. `𝟙[x_t^i = MASK]` — 只在 mask 位置计算 loss
2. `-log p_θ(x_0^i | x_t)` — 标准交叉熵
3. `1/t` — 加权归一化，让不同掩码强度的贡献均衡

### 4.2 训练伪代码

```python
def training_step(x0, model):
    L = x0.shape[1]                          # 序列长度

    # Step 1: 采样掩码强度 t ~ Uniform(0, 1)
    t = uniform(0, 1)

    # Step 2: 独立 mask 每个位置
    mask = bernoulli(t, size=L)              # 每位置以 t 概率被 mask
    xt = where(mask, MASK_TOKEN, x0)

    # Step 3: 双向 Transformer 前向
    logits = model(xt)                        # 输出 [L, V]

    # Step 4: 只在 mask 位置算 CE loss
    ce = cross_entropy(logits[mask], x0[mask])
    loss = ce.mean() / t                      # 关键: 除以 t 做归一化

    return loss
```

### 4.3 为什么要 (1/t) 加权

这是**变分下界推导**出来的，直觉如下：

- t 小（mask 少）时：任务简单，per-token loss 小 —— **但样本贡献要小**，因为覆盖信息少
- t 大（mask 多）时：任务难，per-token loss 大 —— **需要放大梯度**让模型多学
- **1/t 系数让不同噪声强度的贡献均衡**

论文里证明这个目标是 **negative log-likelihood 的上界（ELBO）**，所以是 principled 训练目标。

---

## 五、推理过程：迭代填空 + 置信度调度

### 5.1 基本流程

```
输入 prompt: "介绍一下扩散模型"
目标: 生成 200 tokens 回复,总步数 N=50

Step 0: 初始化 response 区为全 [MASK]
   [prompt] + [M M M M M M ... M]     ← 200 个 M

Step 1: 双向 Transformer 前向
   → 得到所有 M 位置的 token 概率分布
   → 每个位置采样 token,同时得到"置信度"(概率最大值)
   → 保留置信度最高的 k = 200/50 = 4 个位置
   → 剩下 196 个位置 remask

Step 2: 在保留 4 个真实 token 的基础上再前向
   → 得到剩下 196 个 M 的预测
   → 保留最高置信度的 4 个
   → 剩下 192 个 remask

...

Step 50: 全部填完 ✓
```

### 5.2 关键调度参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| **响应长度 L** | 一次生成多少 token | 需**预先指定**（不像 AR 可以动态） |
| **采样步数 N** | 迭代多少步 | 1 ≤ N ≤ L，典型 N = L/4 到 L/8 |
| **每步 unmask 数 k** | k = L / N | 均匀分配（也可以非均匀） |
| **温度 τ** | 采样温度 | 通常 0（贪心）或 1.0 |
| **Top-p / Top-k** | 采样截断 | 类似 AR |

### 5.3 采样步数的 trade-off

```
N = 1     → 一步到位,并行度最高,质量最差(几乎不可用)
N = L/16  → 极端加速,质量下降明显
N = L/8   → LLaDA 论文的甜点区,速度快 + 质量近似最优
N = L/4   → 更保险的选择
N = L     → 每步只填 1 个 token,等价于自回归,质量最好但最慢
```

**关键洞察**：**扩散模型的推理步数是超参** —— 可以在速度和质量之间**滑动 trade-off**。AR 做不到这个。

---

## 六、三种 Remask 策略

推理时"保留哪些位置、remask 哪些位置"是核心工程细节。LLaDA 论文和后续工作里有三种主要策略：

### 6.1 策略 A · Low-Confidence Remasking（默认）

**规则**：每步预测后，把**置信度低的位置**重新 mask，保留高置信度的。

```python
def low_confidence_remask(logits, mask, keep_k):
    # logits: [L, V], mask: [L] (True 表示 mask 位置)
    probs = softmax(logits)
    top_probs, top_tokens = probs.max(dim=-1)      # 每个位置最高概率和对应 token

    # 只在 mask 位置里挑最高置信度的 keep_k 个
    mask_positions = where(mask)[0]
    confidences = top_probs[mask_positions]
    top_k_indices = confidences.argsort(descending=True)[:keep_k]

    # 保留这些位置,其他继续 mask
    new_mask = mask.clone()
    new_mask[mask_positions[top_k_indices]] = False
    return top_tokens, new_mask
```

**直觉**："先填容易的，难的留后面填"。**这是 LLaDA 默认策略**。

### 6.2 策略 B · Random Remasking（基线）

**规则**：随机选一半重新 mask。

**效果**：远不如策略 A，因为无法利用置信度信号。

### 6.3 策略 C · Semi-Autoregressive Remasking（SFT 版常用）

**规则**：
- 把 response 区切成多个 **block**（比如每 32 token 一个 block）
- **块内并行填空**，**块间从左到右**

```
Response 区分成 [Block1 | Block2 | Block3 | Block4]

Round 1: 先填 Block1 (内部并行 N/4 步)
Round 2: 再填 Block2 (Block1 已填,Block2 内部并行)
Round 3: 再填 Block3
Round 4: 再填 Block4
```

**为什么这样效果好**：
- **更接近传统对话生成的思维方式**（有前文再想后文）
- **SFT 效果显著优于纯并行**
- **可以做流式输出**：每完成一个 block 就吐给用户

**代价**：并行度下降，速度不如纯 Low-Confidence。

### 6.4 三种策略对比

| 策略 | 生成质量 | 速度 | 流式输出 | 用途 |
|------|---------|------|---------|------|
| Low-Confidence | ★★★★☆ | ★★★★☆ | ❌ | Base 版默认 |
| Random | ★★☆☆☆ | ★★★★☆ | ❌ | 基线对比 |
| Semi-AR | ★★★★★ | ★★★☆☆ | ✅ | **SFT/对话推荐** |

---

## 七、Prompt 保护与 SFT

### 7.1 Prompt 保护

一个很重要但容易忽略的细节：**推理时 prompt 部分永远不被 mask、永远不被 remask**。

```
序列结构:  [prompt] [response 区]
              ↓         ↓
        保持完整   从全 [M M M ...] 迭代填出真实 token
```

这就实现了**条件生成** —— 模型看到完整的 prompt，只预测 response 区。

### 7.2 SFT 阶段的方法

Base 版 LLaDA 用第四节的目标在 2.3T tokens 上从零训练。SFT 版（Instruct）用同样目标但做了调整：

**数据格式**（和 AR SFT 类似）：

```
输入: <user> 什么是量化 </user> <assistant> 量化就是... </assistant>
      └──── prompt ────┘       └────── response ──────┘
```

**LLaDA 1.0 的 SFT 掩码策略**：
- Prompt 部分：**保持完整**，不 mask，不算 loss
- Response 部分：**按 t 采样 mask 强度，只 mask 这里**

**iLLaDA 的重要改进**（[iwiki 4023224165](https://iwiki.woa.com/p/4023224165) 提到）：
> 打破传统"只 mask response"的 SFT 方式，使用**与预训练相同的全序列随机掩码**，实现训练-推理的完美对齐。

这是 iLLaDA 一个重要的稳定性提升。

---

## 八、和自回归的具体对比

用一个具体例子把方法差异讲透。

### 8.1 AR 生成"扩散模型是一种生成模型"

```
Step 1: "扩" → 输出 "散"      (输入 1 token, 只看左)
Step 2: "扩散" → 输出 "模"     (输入 2 token, KV Cache 复用前 1 token)
Step 3: "扩散模" → 输出 "型"   (输入 3 token, KV Cache 复用前 2 token)
...
Step 10: "扩散模型是一种生成模" → 输出 "型"

特点:
- 每步只吐 1 个 token
- KV Cache 增量算,前面 K/V 复用
- 输出长度 = 生成步数
- 无法回头修改
```

### 8.2 LLaDA 生成同样的句子（N=4 步，L=10）

```
Step 0: [M M M M M M M M M M]  (10 个 mask)

Step 1: 双向 Transformer 一次预测所有 10 个位置
        → 采样得到 [扩,散,模,型,是,一,种,生,成,型]
        → 置信度: [.9, .85, .7, .6, .95, .9, .8, .7, .6, .5]
        → 保留 top-3: 位置 0(扩) 位置 4(是) 位置 1(散)
        → [扩, 散, M, M, 是, M, M, M, M, M]

Step 2: 在 3 个已定 token 基础上预测剩下 7 个
        → 保留最高置信度的 3 个
        → [扩, 散, 模, M, 是, 一, 种, M, M, M]

Step 3: 保留 top-2 → [扩, 散, 模, 型, 是, 一, 种, 生, M, M]

Step 4: 全部填完 → [扩, 散, 模, 型, 是, 一, 种, 生, 成, 模]  ⚠️ 最后一个字?!
```

**看到问题了吗**？

扩散模型可能会**出现自相矛盾** —— 第 1 步蒙对了大概轮廓（"生成型"），但后续步骤并不知道之前蒙对的位置，可能会填出"生成模型"或者"生成型"这样的重复/矛盾。

**这就是 LLaDA 2.1 引入 Token Editing 的动机** —— 不只 remask，还要**主动纠错已生成的 token**。

### 8.3 全方位对比表

| 维度 | AR | LLaDA |
|------|-----|-------|
| **生成方向** | 只能左到右 | 任意顺序，双向 |
| **一步吐几个 token** | 1 个 | 多个（并行填空） |
| **可否修改已生成** | ❌ 不能 | ✅ 后续步可覆盖 |
| **Attention 方向** | Causal | Bidirectional |
| **KV Cache** | ✅ 天然适用 | ⚠️ 需要改造 |
| **训练目标** | Next Token Prediction | (1/t) 加权 masked CE |
| **推理复杂度** | O(L²) 一次 | O(L²) × N 次（每步全序列） |
| **反向推理** | ❌ 弱（补上句难） | ✅ 强（天然双向） |
| **中间填空** | ❌ 需专门微调 | ✅ 原生支持 |
| **代码 patch** | ⚠️ 一般 | ✅ 强项（可局部改） |

---

## 九、方法层面的创新点（论文级贡献）

| 创新 | 意义 |
|------|------|
| **均匀采样掩码率** | 覆盖 0-100% 强度，学到任意程度的填空能力 |
| **(1/t) 加权 loss** | 变分下界推导，principled 训练目标 |
| **Prompt 保护** | 优雅实现条件生成，不需要额外 encoder |
| **迭代置信度调度** | 推理时 speed-quality trade-off 可调 |
| **完全兼容 AR 架构** | 只改 attention mask，其他组件全部复用 |
| **证明扩散语言模型的 Scaling Law** | 8B 效果追平 LLaMA-3-8B |
| **反向推理天然优势** | "给下句写上句"这类任务超过 GPT-4o |

---

## 十、一图流总结

```
┌─────────────────────────────────────────────────────────────┐
│                      LLaDA 方法全景                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   训练:                                                     │
│      x0 --随机采样 t → 独立 mask--> xt                      │
│                                     │                       │
│                                     ▼                       │
│                     双向 Transformer 去噪器                 │
│                                     │                       │
│                                     ▼                       │
│              masked CE loss 只算 mask 位置                  │
│                                     │                       │
│                                     ▼                       │
│                        (1/t) 加权归一化                     │
│                                                             │
│   推理:                                                     │
│      [prompt] + [全 mask response]                          │
│                        │                                    │
│                        ▼ (迭代 N 步)                        │
│         每步:  预测所有 M 位置                              │
│                → 采样 token + 得到置信度                    │
│                → 保留高置信度                               │
│                → 剩下 remask                                │
│                        │                                    │
│                        ▼                                    │
│               全部填完,输出完整回复                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

配套的流程图：[LLaDA 训练 + 推理流程图](./llada-flow.md)（包含文字版 + HTML 可视化版链接）

---

## 十一、参考

1. Nie et al. *Large Language Diffusion Models.* NeurIPS 2025. [arXiv 2502.09992](https://arxiv.org/abs/2502.09992)
2. Nie et al. *Improved Large Language Diffusion Models (iLLaDA).* 2026. [arXiv 2606.25331](https://arxiv.org/abs/2606.25331)
3. Inclusion AI. *LLaDA 2.0: Scaling Up Diffusion Language Models to 100B.* [arXiv 2512.15745](https://arxiv.org/abs/2512.15745)
4. Sahoo et al. *Simple and Effective Masked Diffusion Language Models (MDLM).* NeurIPS 2024. [arXiv 2406.07524](https://arxiv.org/abs/2406.07524)
5. 内网 iwiki: [DLM 深度研究报告](https://iwiki.woa.com/p/4026624940)
6. 内网 iwiki: [iLLaDA 深度解读](https://iwiki.woa.com/p/4023224165)

---

## 下一步

- 视觉化流程图 → [LLaDA 训练+推理流程图](./llada-flow.md)
- Infra 视角 → [dLLM Infra 改进方向](./dllm-infra.md)
