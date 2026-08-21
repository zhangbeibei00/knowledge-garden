---
sidebar_position: 1
title: Pre-Norm Block 逐步拆解
description: 用张量形状追踪的方式，把 Transformer Block 的 LayerNorm、Attention、残差、FFN 每一步讲清楚
tags: [transformer, layernorm, attention, pre-norm, residual]
---

# Pre-Norm Block 逐步拆解

现代大模型（LLaMA、Qwen、Mistral、GPT 系列）几乎都采用 **Pre-Norm** 结构。这一节把一个 Block 内部的每一步张量形状、每个 Norm 归一化对象、每次残差加的到底是什么，都拆开讲清楚。

---

## 一、Block 结构总览

Pre-Norm Block 的核心公式：

```
Block(x):
  h   = x + Attention(LayerNorm(x))
  out = h + FFN(LayerNorm(h))
  return out
```

- 每个子模块（Attention / FFN）**前面**加一个 LayerNorm
- 残差始终加的是 **未归一化** 的量（`x` 和 `h`），不是 `x_norm` / `h_norm`
- 主干道（residual highway）全程"裸奔"，这是 Pre-Norm 好训的关键

---

## 二、张量维度约定

```
x.shape = [batch_size, seq_len, hidden_dim]
        = [B,          L,       H         ]
```

例如 LLaMA-7B: `B=2, L=1024, H=4096`。

**LayerNorm/RMSNorm 沿最后一维 `H` 做归一化**：
- 每个 token 独立归一化自己的 H 维向量
- 不跨 token、不跨 batch

---

## 三、Block 逐步图解

```text
输入 x: [B, L, H]              H = hidden_dim (如 4096)
  │
  │   ┌─────────── 第①个 LayerNorm ────────────┐
  │   │ 对 x 的每个 token 的 H 维向量做归一化    │
  │   │ 输出 x_norm: [B, L, H]                  │
  │   │ Norm 的是 "Attention 的输入"            │
  │   └─────────────────────────────────────────┘
  │           │
  │           ▼
  │       x_norm  ← 只有它进 Attention，x 本身留着做残差
  │           │
  │      ┌────┴────┐
  │      │ Attention │
  │      │           │
  │      │  Q = x_norm @ W_Q     [B, L, H]
  │      │  K = x_norm @ W_K     [B, L, H]
  │      │  V = x_norm @ W_V     [B, L, H]
  │      │        ↑
  │      │        └─ Q/K/V 从"已经归一化的 x_norm"投影出来
  │      │           所以 QKV 天然继承了 Norm 的稳定性
  │      │
  │      │  score = softmax(QK^T / √d) @ V
  │      │  attn_out = score @ W_O     [B, L, H]
  │      └────┬────┘
  │           │
  └───────►  + ◄──── 残差连接：h = x + attn_out
             │        注意：加的是"原始的 x"，不是 x_norm
             │
             h: [B, L, H]
             │
             │   ┌─────────── 第②个 LayerNorm ──────────┐
             │   │ 对 h 的每个 token 的 H 维向量归一化   │
             │   │ 输出 h_norm: [B, L, H]                │
             │   │ Norm 的是 "FFN 的输入"                │
             │   └───────────────────────────────────────┘
             │           │
             │           ▼
             │       h_norm
             │           │
             │       ┌───┴───┐
             │       │  FFN  │  h_norm → Linear → SwiGLU → Linear
             │       │       │  ffn_out: [B, L, H]
             │       └───┬───┘
             │           │
             └────────► + ◄──── 残差连接：out = h + ffn_out
                        │        又是加的原始 h，不是 h_norm
                        │
                        out: [B, L, H] → 送到下一个 Block
```

---

## 四、Q/K/V 需要额外 Norm 吗？

**标准 Pre-Norm 结构里：Q、K、V 不再单独 Norm。** 因为它们的输入 `x_norm` 已经是归一化后的，稳定性已经"传递"给了 Q/K/V。

但现代新模型出现了 **QK-Norm** 变种：

```python
Q = x_norm @ W_Q
K = x_norm @ W_K
V = x_norm @ W_V

# 额外多做一步（QK-Norm）：
Q = RMSNorm(Q)   # 对 Q 再归一化
K = RMSNorm(K)   # 对 K 再归一化
# V 通常不加

score = softmax(QK^T / √d) @ V
```

**为什么要加 QK-Norm？**
- 训练超大模型（22B+）或超长上下文时，`QK^T` 数值容易爆炸
- softmax 输入过大 → 分布"尖锐化" → 梯度消失
- Norm 一下让 QK 的模长可控，softmax 稳定

**用了 QK-Norm 的模型**：Google Gemma 2 / Gemma 3、ViT-22B、Grok、部分 Qwen 变种。普通 LLaMA/GPT-2/Mistral 不加。

---

## 五、代码对照（LLaMA 风格）

```python
class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        self.input_layernorm = RMSNorm(hidden_size)          # 第①个 Norm
        self.self_attn = LlamaAttention(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size) # 第②个 Norm
        self.mlp = LlamaMLP(hidden_size)

    def forward(self, x):
        # === Attention 分支 ===
        residual = x                              # 保留原始 x
        x = self.input_layernorm(x)               # ← Norm 输入
        x = self.self_attn(x)                     # 内部做 Q/K/V + attn
        x = residual + x                          # ← 加回未归一化的 x

        # === FFN 分支 ===
        residual = x                              # 保留 h
        x = self.post_attention_layernorm(x)      # ← Norm 输入
        x = self.mlp(x)                           # FFN
        x = residual + x                          # ← 加回未归一化的 h

        return x
```

**Norm 和残差是"配套"的**：Norm 只在进入子模块前用，残差始终加 Norm 之前的量。

---

## 六、Pre-Norm vs Post-Norm

|  | Post-Norm（原始 Transformer） | Pre-Norm（现代主流） |
|---|---|---|
| 结构 | `LN(x + Sublayer(x))` | `x + Sublayer(LN(x))` |
| 梯度稳定性 | 深层易爆炸/消失 | 残差路径干净，梯度稳定 |
| 训练难度 | 需要 warmup，调参敏感 | 几乎不需要 warmup |
| 深度可扩展性 | 24 层已经吃力 | 100+ 层也能训 |
| 代表模型 | 原始 Transformer、BERT | GPT-2 之后所有 LLM |

**为什么 Pre-Norm 稳？** 因为 residual highway（`x + f(x)` 里的 `x`）从头到尾没被 Norm 触碰，反向传播时梯度可以沿这条高速路径无损传回浅层。

---

## 七、Norm 位置汇总表

| 位置 | 对象 | 形状 | 作用 |
|---|---|---|---|
| Attention 前 Norm | 每 token 的 H 维向量 | `[B, L, H]` | 稳定 Q/K/V 投影输入 |
| QK-Norm（可选） | 每头每 token 的 Q/K 向量 | `[B, L, n_heads, head_dim]` | 稳定 softmax，防大模型爆炸 |
| FFN 前 Norm | 每 token 的 H 维向量 | `[B, L, H]` | 稳定 FFN 输入 |
| Final Norm | 每 token 的 H 维向量 | `[B, L, H]` | 稳定 LM head 输入 |

---

## 八、核心记忆点

1. 🎯 Norm 永远沿 `hidden_dim` 做，每个 token 独立
2. 🎯 Norm 在**进子模块前**做，残差加的是**未归一化**的量
3. 🎯 Q/K/V 通常不额外归一化，除非用了 QK-Norm 变种
4. 🎯 主干道（残差路径）全程"裸奔"，这是 Pre-Norm 好训的原因

---

## 相关

- [SwiGLU FFN 图解](./swiglu-ffn.md) —— Block 里 FFN 那一步的展开
- [量化基础](../../quantization/basics/index.md) —— 为什么 Norm 层通常保 FP16、Q/K/V 投影是量化主战场
