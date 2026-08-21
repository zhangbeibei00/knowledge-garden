---
sidebar_position: 2
title: SwiGLU FFN 图解
description: LLaMA 系列 FFN 的三个 Linear + 门控相乘 —— 为什么这么设计、参数量分布、量化关注点
tags: [transformer, ffn, swiglu, silu, glu, llama]
---

# SwiGLU FFN 图解

Transformer Block 里的 FFN（Feed-Forward Network，也叫 MLP）是 **参数最多的部分**（占一个 LLM 约 2/3 参数）。LLaMA 之后的主流大模型都用 **SwiGLU FFN**，比原始 Transformer 的 FFN 复杂一点，但效果稳定更好。

---

## 一、先看原始 Transformer 的 FFN（作为对比）

**GPT-2 / BERT 时代**：

```
h_norm → Linear(H → 4H) → GELU/ReLU → Linear(4H → H) → 输出
```

维度变化：

```
[B, L, H]  →  [B, L, 4H]  →  [B, L, 4H]  →  [B, L, H]
   ↑              ↑              ↑              ↑
 输入           升维           非线性         降维
```

**核心思路**：升维 → 非线性 → 降维。经典的两层 MLP。

---

## 二、SwiGLU FFN（LLaMA/Qwen/Mistral 主流）

```text
                    ┌─→ Linear_gate (H → 4H) ─→ SiLU ─┐
                    │                                  │
h_norm [B, L, H] ───┤                                  ×  (逐元素相乘)
                    │                                  │
                    └─→ Linear_up   (H → 4H) ──────────┘
                                                       │
                                                       ▼
                                          intermediate [B, L, 4H]
                                                       │
                                                       ▼
                                        Linear_down (4H → H)
                                                       │
                                                       ▼
                                          output [B, L, H]
```

**三个 Linear**（不是两个！）：

| 名称 | 形状 | 作用 |
|---|---|---|
| `gate_proj` (W_gate) | H → 4H | 生成"门控信号" |
| `up_proj` (W_up) | H → 4H | 生成"值信号" |
| `down_proj` (W_down) | 4H → H | 把结果降回原维度 |

---

## 三、SwiGLU 是什么？

### 1. SiLU（Swish）激活函数

$$
\mathrm{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

图形上像 ReLU 但更平滑：
- x 很大时 ≈ x（像 ReLU）
- x 很小时 ≈ 0（像 ReLU）
- 0 附近**平滑可导**，梯度传播比 ReLU 稳

### 2. GLU（Gated Linear Unit，门控线性单元）

用一路信号"控制"另一路信号：

$$
\mathrm{GLU}(x) = (x W_1) \odot \sigma(x W_2)
$$

其中 `⊙` 表示逐元素相乘（element-wise multiply）。

### 3. SwiGLU = GLU 里把 sigmoid 换成 SiLU

$$
\mathrm{SwiGLU}(x) = (x W_{up}) \odot \mathrm{SiLU}(x W_{gate})
$$

**完整 FFN 公式**：

$$
\mathrm{FFN}(h) = W_{down} \cdot \bigl(\, \mathrm{SiLU}(W_{gate} h) \odot (W_{up} h) \,\bigr)
$$

---

## 四、完整代码（LLaMA 的 MLP）

```python
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        # hidden_size = H            (如 4096)
        # intermediate_size ≈ 4H 或 8H/3  (如 11008，LLaMA-7B)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, h_norm):
        # h_norm: [B, L, H]
        gate = self.gate_proj(h_norm)               # [B, L, 4H]  门信号
        up   = self.up_proj(h_norm)                 # [B, L, 4H]  值信号
        intermediate = self.act_fn(gate) * up       # ← 关键：门 × 值
        output = self.down_proj(intermediate)       # [B, L, H]
        return output
```

---

## 五、维度变化全过程（LLaMA-7B 为例）

| 步骤 | 张量 | 形状 |
|---|---|---|
| 输入 | h_norm | `[B, L, 4096]` |
| 上路 | `up = W_up @ h_norm` | `[B, L, 11008]` |
| 下路 | `gate = W_gate @ h_norm` | `[B, L, 11008]` |
| 激活 | `SiLU(gate)` | `[B, L, 11008]` |
| 门控相乘 | `SiLU(gate) ⊙ up` | `[B, L, 11008]` |
| 降维 | `W_down @ ...` | `[B, L, 4096]` |
| 输出 | ffn_out | `[B, L, 4096]` |

**参数量**（LLaMA-7B 一层 FFN）：
- gate_proj: 4096 × 11008 ≈ 45M
- up_proj:   4096 × 11008 ≈ 45M
- down_proj: 11008 × 4096 ≈ 45M
- **合计 ~135M 参数/层**（一层 Attention 只有 ~67M，FFN 占了大头）

---

## 六、为什么 SwiGLU 比传统 FFN 好？

### 1. 门控机制带来选择性

```
SiLU(gate)  是一个"软性开关"（不是 0/1 而是连续值）
up          是原始"值"
两者相乘：门决定值能通过多少
```

FFN 就有了"按输入内容动态过滤信息"的能力，比 `激活(线性)` 表达力更强。

### 2. 平滑可导

SiLU 处处可导，梯度传播比 ReLU 稳定。

### 3. 实验验证

- **GLU Variants Improve Transformer**（Shazeer, 2020）：SwiGLU / GeGLU 在 GLUE、SuperGLUE 上稳定超过 ReLU/GELU，同参数下 loss 更低、收敛更快
- **LLaMA 论文**验证：SwiGLU 效果最好，被采纳为标配

### 4. 代价与设计

- 多了一个 Linear（3 个 vs 2 个）
- 为保持总参数量相当，intermediate_size 从 4H 缩小到 **8H/3**
- LLaMA-7B: 4096 × 8/3 ≈ 10922 → 取 11008（对齐 128 的倍数，便于 GPU 计算）

---

## 七、其它 GLU 变种

| 名字 | 公式 | 用于 |
|---|---|---|
| GLU | `(xW1) ⊙ σ(xW2)` | 原始 GLU 论文 |
| ReGLU | `(xW1) ⊙ ReLU(xW2)` | 早期变种 |
| GeGLU | `(xW1) ⊙ GELU(xW2)` | PaLM、部分 Gemma |
| **SwiGLU** | `(xW1) ⊙ SiLU(xW2)` | **LLaMA、Qwen、Mistral（主流）** |

---

## 八、直觉理解（类比）

把 FFN 想象成一个「筛选 + 加工」的车间：

```
h_norm 是原料
  │
  ├─→ [W_up]                把原料加工成产品候选（值）
  │
  ├─→ [W_gate + SiLU]       给每个产品打分（通过率）
  │
  ▼
产品 × 通过率 = 实际输出的产品
  │
  └─→ [W_down]              打包压缩回原尺寸
```

- **传统 FFN**：只加工，不筛选（激活函数只是简单非线性）
- **SwiGLU**：加工 + 智能筛选，模型自己学"哪些特征该保留、哪些该压制"

---

## 九、量化视角（AngelSlim / 部署相关）

FFN 是量化/剪枝的**主战场**，因为参数最多、计算最重。SwiGLU 的量化关注点：

1. **gate_proj / up_proj / down_proj 三个 Linear** —— 主要量化对象，权重通常 INT4/INT8/FP8，激活按方案选择
2. **SiLU(gate) ⊙ up 逐元素乘** —— 量化难点：
   - 两路乘积会放大量化误差（误差 × 误差）
   - `SiLU(gate)` 输出范围较大（可正可负），配合 `up` 后动态范围进一步扩大
   - 通常需要在乘积后重新校准 scale，或对 gate/up 分别做 per-channel 量化
3. **down_proj 的输入** —— 是 `SiLU(gate) ⊙ up`，异常值（outlier）较多，是 SmoothQuant/AWQ 等方法的重点处理点
4. **激活函数本身** —— SiLU 通常不量化，或在量化推理时用查表/多项式近似

---

## 十、TL;DR

| 组件 | 作用 |
|---|---|
| `gate_proj` (H→4H) | 生成"门控信号"，决定信息通过率 |
| `up_proj` (H→4H) | 生成"值信号"，原始特征 |
| `SiLU(gate) ⊙ up` | 门 × 值，动态筛选 |
| `down_proj` (4H→H) | 压缩回原维度 |

**放回 Pre-Norm Block 里就是**：

```
h_norm = RMSNorm(h)                       ← Block 里第②个 Norm
   │
   ├─→ gate = h_norm @ W_gate  [B,L,4H]
   ├─→ up   = h_norm @ W_up    [B,L,4H]
   │
   inter = SiLU(gate) ⊙ up     [B,L,4H]   ← FFN 内部升维空间
   │
   ffn_out = inter @ W_down    [B,L,H]    ← 降回原维度
   │
out = h + ffn_out                          ← 残差加回未归一化的 h
```

---

## 相关

- [Pre-Norm Block 逐步拆解](./pre-norm-block.md) —— SwiGLU FFN 在 Block 里的上下文
- [量化基础](../../quantization/basics/index.md) —— 为什么 FFN 是量化的主战场
