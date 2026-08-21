---
sidebar_position: 3
title: RoPE 旋转位置编码 · 一篇讲透
description: 从 2D 旋转到高维子空间组合、Attention 内的代码位置、长序列外推、cos/sin cache 与 KV cache 的分工、Infra 视角的收益
tags: [transformer, rope, positional-encoding, kv-cache, attention, infra]
---

# RoPE 旋转位置编码 · 一篇讲透

LLaMA、Qwen、Mistral、DeepSeek 等现代大模型的位置编码全部换成了 **RoPE（Rotary Position Embedding，旋转位置编码）**。它不是"更好的一个 trick"，而是一个**数学上优雅、工程上友好**的双赢设计。

这篇按下面这条脉络展开：

```
1. 核心洞察：为什么"旋转"能编码位置
2. 数学原理（纯文本版）
3. 一个 token 在 Attention 里的完整 RoPE 流程
4. 长序列会不会"转回来"？子空间组合编码的救赎
5. cos/sin cache：为什么这样组织
6. cos/sin cache vs KV cache：两个 cache 的分工
7. Infra 视角：为什么 RoPE 横扫工业界
```

---

## 一、核心洞察

### 1. 位置编码要解决的问题

Transformer 的 Attention 本身是"集合运算"—— 交换任意两个 token 的位置，输出不变（除了 mask）。要区分 "我打你" 和 "你打我"，必须把**位置信息**注入进去。

老派做法是在 Embedding 层加一个位置向量：

```
x = word_emb(token_id) + pos_emb(pos)     ← 加法融合
```

问题：
- 位置信息经过 32 层 Attention 会被"稀释"
- 加法融合会污染词义（词义 + 位置塞进同一维度）
- 学习式 PE 无法外推（训练 2K 就永远只能 2K）

### 2. RoPE 的核心洞察

不在输入层加位置，而是在**每一层 Attention 内部，通过"旋转" Q 和 K 来注入位置**。

关键性质：**当两个旋转后的向量做内积时，结果只依赖于它们的旋转角度差**。

如果让旋转角度正比于位置索引，那么：
- 绝对位置通过旋转注入
- **相对位置在 Attention 内积时自动涌现**
- 一举两得

这就是 RoPE 的全部秘密。下面展开数学。

---

## 二、数学原理（纯文本版）

### 1. 从 2D 旋转开始

在 2D 平面上，把向量 `v = [a, b]` 逆时针旋转 θ 角度：

```
┌ a' ┐   ┌ cos θ   -sin θ ┐   ┌ a ┐
│    │ = │                 │ · │   │
└ b' ┘   └ sin θ    cos θ ┘   └ b ┘

展开:
    a' = a·cos θ - b·sin θ
    b' = a·sin θ + b·cos θ
```

记这个旋转矩阵为 `R(θ)`，它有两个关键性质：

```
R(θ_1) · R(θ_2) = R(θ_1 + θ_2)     旋转可加
R(θ)^T · R(θ)   = I                 正交，不改变模长
```

### 2. 神奇的内积性质

两个 2D 向量 q、k 各自旋转 θ_q 和 θ_k 后做内积：

```
<R(θ_q)·q,  R(θ_k)·k> = q^T · R(θ_q)^T · R(θ_k) · k
                      = q^T · R(-θ_q) · R(θ_k) · k       ★ 用正交性
                      = q^T · R(θ_k - θ_q) · k           ★ 用可加性
```

**结论**：旋转后的 q 和 k 做内积，只依赖旋转差 `(θ_k - θ_q)`，与 θ_q、θ_k 的绝对值无关。

### 3. 把"旋转差"换成"位置差"

让 θ 正比于位置索引 `pos`：

- 第 m 个 token 的 q 旋转 `m·ω` 度
- 第 n 个 token 的 k 旋转 `n·ω` 度

内积：

```
<R(m·ω)·q,  R(n·ω)·k> = q^T · R((n-m)·ω) · k
                                 ↑
                            只依赖 (n-m)
```

**绝对位置注入了、相对位置自动出现了**。这是 RoPE 的核心方程。

### 4. 高维扩展：子空间组合编码

实际 `head_dim = 64` 或 128，不是 2。RoPE 的做法是：**把 head_dim 拆成很多个 2D 子空间，每个子空间独立旋转，用不同的频率**。

假设 `head_dim = 8`：

```
q = [q0, q1, q2, q3, q4, q5, q6, q7]

分成 4 组，每组 2 维：
    组 0: [q0, q1]    用频率 ω_0
    组 1: [q2, q3]    用频率 ω_1
    组 2: [q4, q5]    用频率 ω_2
    组 3: [q6, q7]    用频率 ω_3
```

频率用几何递减（跟原始 Sinusoidal 一样）：

```
ω_i = 1 / base^(2i / head_dim)     base 通常 = 10000

head_dim=8 时:
    ω_0 = 1 / 10000^(0/8)  = 1.0        （最快）
    ω_1 = 1 / 10000^(2/8)  ≈ 0.1
    ω_2 = 1 / 10000^(4/8)  ≈ 0.01
    ω_3 = 1 / 10000^(6/8)  ≈ 0.001      （最慢）
```

**低维旋转快，高维旋转慢**。第 m 个 token 的每组各自旋转：

```
[q0', q1'] = R(m·ω_0) · [q0, q1]
[q2', q3'] = R(m·ω_1) · [q2, q3]
[q4', q5'] = R(m·ω_2) · [q4, q5]
[q6', q7'] = R(m·ω_3) · [q6, q7]
```

整体旋转矩阵是"分块对角"：

```
        ┌ R(m·ω_0)                                ┐
        │           R(m·ω_1)                       │
R_m  =  │                     R(m·ω_2)             │
        │                                R(m·ω_3)  │
        └                                          ┘

q_rotated = R_m · q
```

**内积性质对每一组独立成立**，所以整体也成立：

```
<R_m · q,  R_n · k> = Σ_i  <R(m·ω_i)·[q_2i, q_2i+1],  R(n·ω_i)·[k_2i, k_2i+1]>
                    = Σ_i  只依赖 (n-m)·ω_i 的项
                    = 只依赖相对位置 (n-m)
```

这就是**子空间组合编码**：用 `head_dim/2` 个不同频率的 2D 旋转"联合"编码位置。每个位置产生一个独一无二的"旋转指纹"。

---

## 三、一个 token 在 Attention 里的完整 RoPE 流程

先定位：**RoPE 到底发生在哪一步？**

```
LLaMA 模型的完整 forward
═══════════════════════════════════════════════

输入 token_ids  [B, L]
    │
    ▼
┌──────────────────────────────────────────┐
│ Token Embedding: x = W_embed[token_ids]  │  ← 【只有词义，没有位置】
│ x.shape = [B, L, H]                      │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ Decoder Layers × N (LLaMA-7B: N=32)      │
│                                          │
│  每一层:                                  │
│    x_norm = RMSNorm(x)                   │
│    ┌── Attention 内部 ────────────────┐  │
│    │ Q = x_norm @ W_Q                  │  │
│    │ K = x_norm @ W_K                  │  │
│    │ V = x_norm @ W_V                  │  │
│    │                                   │  │
│    │ ★ Q = apply_rope(Q, cos, sin)    │  │ ← RoPE 就在这里
│    │ ★ K = apply_rope(K, cos, sin)    │  │
│    │   V 不旋转                        │  │
│    │                                   │  │
│    │ scores = softmax(QK^T/√d)         │  │
│    │ out    = scores @ V @ W_O         │  │
│    └───────────────────────────────────┘  │
│    x = x + out                            │
│    x = x + FFN(RMSNorm(x))                │
└──────────────────────────────────────────┘
    │
    ▼
Final RMSNorm → LM Head → logits
```

**关键定位**：
- ❌ RoPE 不在 Embedding 层（跟老派 PE 完全不同）
- ✅ 在**每一层 Attention 内**，Q/K 投影完之后、算 attention 之前
- ✅ **每层都重新做一次**（因为每层 Q/K 是新的）
- ✅ **只旋转 Q 和 K，V 不动**（只有 Q/K 参与位置匹配，V 只是内容载荷）

### 代码里的 apply_rope

不去构造那个稀疏的分块对角矩阵，用逐元素计算就够：

```python
def apply_rope(x, cos, sin):
    """
    x:   [B, n_heads, L, head_dim]  Q 或 K
    cos: [L, head_dim/2]            当前 batch 各位置的 cos 值
    sin: [L, head_dim/2]
    """
    # 每两个相邻元素一组，分成"偶数索引"和"奇数索引"两半
    x_even = x[..., 0::2]   # 取 q0, q2, q4, ...
    x_odd  = x[..., 1::2]   # 取 q1, q3, q5, ...

    # 对每对 (q_2i, q_2i+1) 应用 2D 旋转：
    #   q_2i'   = q_2i · cos - q_2i+1 · sin
    #   q_2i+1' = q_2i · sin + q_2i+1 · cos
    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot  = x_even * sin + x_odd * cos

    # 交错拼回去
    out = torch.stack([x_even_rot, x_odd_rot], dim=-1).flatten(-2)
    return out
```

核心就是 **4 次逐元素乘法 + 2 次加减**，加上拼接。计算开销可以忽略。

### 一次 token 的完整旅程

以 `"苹果"` 为例（LLaMA-7B, H=4096, n_heads=32, head_dim=128），假设它在 pos=3：

```
Step 1  文本 → id           "苹果" → 5237
Step 2  Embedding            x[3] = W_embed[5237]，纯词义向量
Step 3  进入 Layer 1
        RMSNorm → Q/K/V 投影 → [1, 32, 4, 128]
        ★ apply_rope(Q, cos[0..3], sin[0..3])   ← 用 pos=3 的角度旋转
        ★ apply_rope(K, cos[0..3], sin[0..3])
        Attention 输出 → 残差 → FFN → 残差
Step 4  重复 Layer 2..32
        每层重新投影 Q/K/V，每层都重新做一次 RoPE
Step 5  Final Norm → LM Head → 采样下一 token
```

---

## 四、长序列会不会"转回来"？

一个非常自然的直觉：**旋转是周期的，pos 很大时是不是会跟前面的位置一样？**

分两层回答。

### 1. 单组会绕圈，但整体不会

每组的旋转周期：

```
组 i 的周期 T_i = 2π / ω_i = 2π · 10000^(2i / head_dim)

head_dim = 128 时:
    i=0   (最高频):  T_0  ≈ 2π      ≈ 6.28
    i=1:             T_1  ≈ 7.36
    ...
    i=32:            T_32 ≈ 628
    i=48:            T_48 ≈ 6283
    i=63  (最低频):  T_63 ≈ 61395
```

第 0 组每 6.28 步就转一圈。所以**单看某一组，pos=7 和 pos=0.72 会几乎重合**。

但整体状态由**所有 64 组的角度联合**决定。整体重复需要**所有组同时回到原位**，这要求所有 ω_i 都是同一个 2π 的整数倍。

由于 `10000^(2i/128)` 之间彼此是无理数比，**这样的 pos 在有限长度内几乎不存在**。

**类比**：钟表的时/分/秒指针。秒针 60 秒回一次，分针 60 分回一次，时针 12 小时回一次。三根指针同时回到同一状态要 12 小时 = 43200 秒。RoPE 用 64 根不同周期的"指针"联合编码位置，每个位置产生独一无二的"指纹"。

这也回到了上一节说的 **子空间组合编码**：单个子空间会绕回来，但组合起来就唯一了。

### 2. 真正的问题：训练分布外

RoPE 单在训练长度内没问题，但**推理超过训练最大长度时，事情变糟**。假设训练 L_train = 2048：

```
第 32 组:  ω_32 ≈ 1/100
    训练时 pos ∈ [0, 2047]，角度分布 [0, 20.48] 弧度 (约 3.3 圈)
    推理 L=8192 时，角度到 [0, 81.92] 弧度 (约 13 圈)
    ★ 模型没见过 20~82 这个角度区间的分布

最低频组:  ω_63 ≈ 1/9772
    训练时角度只从 0 转到 0.21 弧度 (约 12°)
    推理 L=8192 时角度到 0.84 弧度 (约 48°)
    ★ 这一组几乎是"静止"的，训练时压根没转开，模型无法用它区分远距离
```

所以**长上下文外推的真正难题不是"角度绕回来"，而是"训练分布外的角度组合模型不认识"**。

业界的解决方向都是**改造 cos/sin 表**：Position Interpolation (PI) 把位置索引缩放；NTK-aware scaling 只对低频组做缩放；YaRN 分频段处理并加 attention 温度；LongRoPE 逐组搜索最佳缩放。**关键：这些方案不需要重训，改几行 cos/sin 生成代码即可**。

本文不深入展开外推方案，只强调这个数学结构给了扩展的空间。

---

## 五、cos/sin cache：为什么这样组织

RoPE 每次要用 `cos(pos · ω_i)` 和 `sin(pos · ω_i)`。自然的想法：**提前把所有可能的值算好，用的时候查表**。

### 1. cache 长什么样

```
cos_cache[pos, i] = cos(pos · ω_i)
sin_cache[pos, i] = sin(pos · ω_i)

shape: [max_seq_len, head_dim/2]

以 LLaMA-7B (max_len=4096, head_dim=128) 为例:
    每个表 = 4096 × 64 = 262144 个数
    两个表 (cos + sin) FP32 = 2.1 MB
```

**总共几 MB，非常小**。

### 2. 组织策略

```
✅ 模型初始化时算一次
    precompute_freqs_cis() 在 __init__ 里跑一次
    之后 forward 全部查表，不再算 cos()/sin()

✅ 存 FP32 或 BF16
    三角函数结果 ∈ [-1, 1]，低精度损失敏感
    RoPE 涉及小角度旋转，FP16 或更低会明显掉点
    主流：FP32 或 BF16 存储

✅ GPU 上一直驻留
    register_buffer 注册进模型，跟 .to("cuda") 一起上显存
    只占几 MB，一次性上传，永久使用
    forward 时无 CPU-GPU 通信

✅ 所有 layer 共享一份
    每层的 head_dim 都一样，所以每层的 ω_i 都一样
    LLaMA-7B 32 层如果各存一份：32 × 2.1 MB = 67 MB
    共享后只需 2.1 MB
    这也是为什么 cos/sin 存在 LlamaModel 而不是 LlamaAttention
```

### 3. 组织进模型的代码

```python
class LlamaModel(nn.Module):
    def __init__(self, config):
        cos, sin = precompute_freqs_cis(
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
        )
        self.register_buffer("cos_cache", cos)  # 全局一份
        self.register_buffer("sin_cache", sin)

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, input_ids, position_ids):
        x = self.embed_tokens(input_ids)

        # 按当前 batch 的 pos 从表里切片一次
        cos = self.cos_cache[position_ids]
        sin = self.sin_cache[position_ids]

        # 传给每一层共享
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x
```

---

## 六、cos/sin cache vs KV cache：两个 cache 的分工

推理里有两个 cache，都叫 cache，但**完全不同的东西**。

|  | cos/sin cache | KV cache |
|---|---|---|
| 存什么 | 位置编码的常量表 | 已生成 token 的 K、V 向量 |
| 依赖什么 | 只依赖 head_dim 和 max_seq_len | 依赖具体输入内容 |
| 何时算 | 模型初始化时算一次 | 每生成一个 token 追加一次 |
| 大小 | 几 MB（固定） | 几 GB 到几百 GB（随长度增长） |
| 是否共享 | 所有层共享一份 | 每层每头独立 |
| 本质 | 数学函数的物化 | 中间计算结果的复用 |

### 1. 它们如何配合

推理生成第 N 个 token 的流程：

```
输入新 token x_new  [1, 1, H]
    │
    ▼
Q_new = x_new @ W_Q     ← 未旋转
K_new = x_new @ W_K
V_new = x_new @ W_V
    │
    ▼
┌───── 用 cos/sin cache ─────┐
│  cos = cos_cache[N]        │  ← 只查一次，只查 pos=N
│  sin = sin_cache[N]        │
│  Q_new_rot = apply_rope(Q_new, cos, sin)   ← 给新 token 做 RoPE
│  K_new_rot = apply_rope(K_new, cos, sin)
└────────────────────────────┘
    │
    ▼
┌───── 更新 KV cache ────────────────────────────┐
│  K_cache = [K_0_rot, K_1_rot, ..., K_{N-1}_rot, K_new_rot]
│  V_cache = [V_0,     V_1,     ..., V_{N-1},     V_new]
│  ★ 历史 K 都已经旋转过，直接复用；新 K 加进去
└────────────────────────────────────────────────┘
    │
    ▼
scores = Q_new_rot @ K_cache^T / √d
attn   = softmax(scores)
out    = attn @ V_cache
```

**关键理解**：
- **cos/sin cache** 是"位置到旋转的翻译手册"（不变）
- **KV cache** 是"已经翻译好的历史文本"（每步追加）
- 要"翻译"新 token，两个都得用；两者是配合关系，不是替代关系

### 2. 为什么 K 存"旋转后"而不是"旋转前"

方案 A：KV cache 存原始 K，每次 attention 时全部重新旋转
- 每次都要旋转 N 个历史 K
- 完全浪费缓存意义

方案 B（实际做法）：K 生成时立刻旋转，存 K_rot
- 每个 K 只旋转一次（生成时）
- 后续复用无成本

**方案 B 是所有主流实现的选择**。这也是为什么说"RoPE 与 KV cache 天然兼容"。

### 3. 大小对比（震撼）

LLaMA-7B, `n_layers=32, n_heads=32, head_dim=128`：

```
cos/sin cache:
    2.1 MB   （固定，跟输入无关）

KV cache（一个 token）:
    32 heads × 128 dim × 2 (K和V) × 32 层 × 2 bytes (FP16) ≈ 512 KB

推理 2K 上下文, batch=1:
    KV cache ≈ 512 KB × 2048 ≈ 1 GB

推理 32K 上下文, batch=1:
    KV cache ≈ 16 GB    ← 已超过 7B 模型权重本身
```

**cos/sin cache 是"便利贴"，KV cache 是"仓库"**，两个数量级完全不同。这也是为什么后者是推理系统的头号敌人（Paged Attention、Prefix Cache、GQA/MQA 都是在解决 KV cache 问题）。

---

## 七、Infra 视角：为什么 RoPE 横扫工业界

RoPE 之所以成为工业主流，除了效果好，更是**在基础设施上非常友好**。这一节把关键 infra 收益列出来。

### 1. KV Cache 天然兼容（推理最关键）

上面已经详细讲过。核心一句：**K 存旋转后的，每个 K 只旋转一次，后续复用无成本**。这与推理系统里"K/V 只算一次"的理念完全对齐。

对比其他方案：

```
T5 relative bias:
    每次新 Q 来，都要为 (new_pos, all_cache_pos) 重新查/算 bias
    显存和计算都要付代价

RoPE:
    KV cache 里 K_rot 直接用，新 Q_rot 也算好
    内积时相对位置自动出现
    ★ 零额外开销
```

**推理框架（vLLM、SGLang、TRT-LLM）把 RoPE 当一等公民优化**，就是因为这个特性。

### 2. Kernel Fusion 友好

Naive 实现每步一个 kernel：

```
q_proj → reshape → apply_rope → attention_scores → softmax → ...
一大堆 kernel launch
```

Fused 实现把 QKV 投影 + RoPE 合并成一个 kernel：

```
fused_qkv_proj_rope(x, W_QKV, cos, sin) → 直接输出 Q_rot, K_rot, V
```

- 减少 kernel launch 开销
- Q/K 中间结果不落显存
- 提升 memory bandwidth 利用率
- **decode 阶段 (L=1) 时收益最大**，因为此时 kernel launch 相对计算占比大

### 3. 显存 / 参数量

```
Learned PE:       max_len × H 可训练参数
                  LLaMA-7B 场景下 8 MB 参数 + 优化器状态 ~24 MB
                  训练时占显存，还要存 checkpoint

RoPE:             0 参数
                  只有 cos/sin buffer ~2 MB
                  register_buffer，不算 param_count
```

**参数量为零** 是 RoPE 的一个隐性优势 —— 模型体量小几 MB 在大规模训练时不是"可忽略"，是"可积累"的。

### 4. 分布式训练天然友好

大模型训练的三种并行方式，RoPE 全部零负担：

**TP（张量并行）—— W_Q, W_K 按 n_heads 切分**
```
每个 rank 负责一部分 head
RoPE 是逐 head 独立做的
★ 无需 rank 间通信
```

**SP（序列并行）—— 序列在 rank 间切分**
```
每个 rank 只有部分 token 的 Q/K
RoPE 逐 token 独立
★ cos/sin 表广播一次即可（几 MB）
```

**PP（流水线并行）—— 层在 rank 间切分**
```
RoPE 在每层内做，跨层无依赖
★ 完美支持
```

**对比 Learned PE 分布式**：位置 embedding 是一个 `[max_len, H]` 的大表，要么每个 rank 复制一份（显存浪费），要么切分（要 all-gather 通信）。都不如 RoPE 干净。

### 5. Flash Attention 兼容

Flash Attention 把 softmax 融合进 attention 计算，避免存 [L, L] 中间矩阵：

```
Flash Attention 的输入: Q, K, V
                        ↑
                这里 Q/K 已经是旋转后的了
                Flash Attention 内部不 care Q/K 是不是旋转过
```

**RoPE 与 Flash Attention 完美解耦**。对比 T5 relative bias 或 ALiBi：需要在 Flash Attention kernel 里加位置相关的 bias 项，要写变种 kernel，工程复杂。

### 6. 量化视角

从量化部署角度看 RoPE 的特点：

```
cos/sin 表:
    必须保持高精度 (FP32 或 BF16)
    - cos/sin ∈ [-1, 1]，绝对值不大
    - 但 x_even·cos - x_odd·sin 这种组合对精度敏感
    - 低精度下小角度旋转会有累积误差
    - INT8/FP8 存 cos/sin 会明显影响长距离位置区分

Q/K 本身:
    可以按标准激活量化方案（INT8/FP8/INT4）
    ★ 关键实践：先做 RoPE，再量化
    - 若先量化再旋转：量化误差会被旋转放大
    - 若先旋转再量化：量化针对最终数值范围，误差可控

主流 quant kernel:
    QKV 投影 (低精度) → RoPE (fp16/bf16) → 后续 attention (低精度)
    中间保持 FP16/BF16 精度做 RoPE
```

### 7. 长上下文扩展的 infra 优势

RoPE 支持一系列**训练后扩长**技巧（PI / NTK / YaRN / LongRoPE），共同特点：

```
不需要重训
不需要改 attention kernel
不需要改 KV cache 逻辑
只需要重新生成 cos/sin 表
```

改几行 `precompute_freqs_cis` 就能把上下文从 4K 扩到 32K 甚至 128K。**这在工程上是极大的优势**——上下文长度成了一个"部署参数"而不是"模型参数"。

### 8. Infra 收益一句话总结

```
RoPE 之所以横扫工业界，除了效果好，更是因为它在 infra 上：

    KV cache 天然兼容    → 推理框架深度优化
    可 kernel 融合       → 计算开销可忽略
    零参数               → 显存 / checkpoint 友好
    分布式友好           → TP/SP/PP 全部零负担
    与 Flash Attention 解耦 → 现代 attention 优化即插即用
    可后训扩展           → 长上下文改 cos/sin 表即可
    量化路径清晰         → 先 RoPE 后量化即可

这是一个"数学上优雅 + 工程上友好"的罕见双赢设计。
```

---

## 八、TL;DR

**核心洞察**：不在 Embedding 加位置，而是在每层 Attention 里旋转 Q 和 K；旋转角度正比于位置，使得内积时相对位置自动涌现。

**数学**：
- 2D 旋转 R(θ)，性质 `<R(θ_q)q, R(θ_k)k> = q^T R(θ_k - θ_q) k`
- 高维 = head_dim/2 个 2D 子空间分别旋转，不同频率
- 子空间组合编码：每个位置产生独一无二的"旋转指纹"

**位置**：在每一层 Attention 内，Q/K 投影后、attention 之前；V 不旋转；每层重新做

**长序列**：单组会绕圈，但组合起来不会。真正难题是训练分布外的角度，用 PI/NTK/YaRN 改 cos/sin 表解决

**两个 cache**：
- cos/sin cache（几 MB，共享，不变）—— 位置的常量表
- KV cache（几 GB，每层独立，追加）—— 已生成 token 的 K_rot 和 V
- 分工明确，配合工作

**Infra 收益**：KV cache 兼容、Kernel Fusion、零参数、分布式友好、Flash Attention 解耦、可后训扩长、量化路径清晰

---

## 相关

- [Pre-Norm Block 逐步拆解](./pre-norm-block.md) —— RoPE 所在的 Block 上下文
- [SwiGLU FFN 图解](./swiglu-ffn.md) —— 同一 Block 内的另一半
- [量化基础](../../quantization/basics/index.md) —— cos/sin 精度与 Q/K 量化的关系
