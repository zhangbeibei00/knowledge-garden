---
sidebar_position: 4
title: Softmax 的 AI Infra 难点 · Online Softmax · Flash Attention
description: 从 memory-bound、数值溢出、Attention O(N²) 三大痛点切入，讲透 Online Softmax 用补偿因子实现 tile 化增量计算的原理，进而通向 Flash Attention
tags: [transformer, softmax, online-softmax, flash-attention, attention, infra, memory-bound]
---

# Softmax 的 AI Infra 难点 · Online Softmax · Flash Attention

Softmax 从数学上只有一行公式，但在 AI Infra 视角下，它是**推理长上下文的头号硬骨头**。理解 Softmax 的三大工程难点，才能真正理解 Flash Attention 为什么是过去几年最重要的 infra 突破。

这篇按下面这条脉络展开：

```
1. Softmax 的数学定义
2. AI Infra 三大难点（溢出 / memory-bound / Attention 中间矩阵）
3. GPU 内存模型速览（HBM vs SRAM，为什么 O(N²) 访存要命）
4. Online Softmax:补偿因子如何让 tile 化成为可能
5. 具体演算追踪(m, d, o) 三个记账变量
6. Online Softmax → Flash Attention
7. 一句话总结
8. 与量化的关联
```

---

## 一、Softmax 的数学定义

给定长度为 N 的向量 x：

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

在 Transformer 里出现在两处：
- **Attention**：`softmax(QK^T / √d_k)`，作用在 [S, S] 矩阵的最后一维
- **LM Head 输出**：`softmax(logits)`，作用在词表维度 [V]

看似简单，AI Infra 视角却是**性能、精度、稳定性**三方面的核心战场。

---

## 二、AI Infra 视角的三大难点

### 难点 1:数值溢出（Numerical Overflow）

FP16 能表示的最大值 ≈ 65504，`exp(12) ≈ 163000` 已经溢出。

Attention 里 `QK^T / √d_k` 的值在长序列、outlier token 场景下经常达到 10+，直接 exp 就崩。

**Safe Softmax**：利用 softmax 的平移不变性，先减去最大值：

```
m = max(x)
softmax(x_i) = exp(x_i − m) / Σ_j exp(x_j − m)
```

减完后所有 exp 输入都 ≤ 0，输出都在 [0, 1]，永不溢出。**这是 PyTorch / CUDA 的标准做法，不是可选优化，是必须做**。

### 难点 2:内存带宽瓶颈（Memory-Bound）

朴素 safe softmax 是 **3-pass** 算法：

```
Pass 1: 遍历 x → 算 max(x)               [读 N 个数]
Pass 2: 遍历 x → 算 Σ exp(x − m)         [再读 N 个数]
Pass 3: 遍历 x → 输出 exp(x − m) / sum   [再读 N + 写 N]

访存量 ≈ 4N，计算量 ≈ 3N flop
算术强度 ≈ 0.75 flop/byte
```

参考 A100 的 roofline:算力 312 TFLOPS，带宽 2 TB/s，**算术强度平衡点 ≈ 150 flop/byte**。

**0.75 ≪ 150** → Softmax 是**极度 memory-bound**，GPU 算力用不到 1%，性能完全由带宽决定。

### 难点 3:Attention 里 softmax 卡在中间不能省略

Attention 的三步：

```
Step 1: S = Q · K^T / √d_k    [B, h, N, N]  ← 中间矩阵超大
Step 2: P = softmax(S)         [B, h, N, N]  ← 必须先读完整个 S 才能算 max
Step 3: O = P · V              [B, h, N, d_k]
```

痛点：
- 中间矩阵 S 形状 [B, h, N, N]，**N=32K 时单头一份就 4GB**（FP16）
- Softmax 需要在最后一维求 max 和 sum，是**跨维度的 reduction**，天然不能简单切块
- 每一步都要把大矩阵在 HBM ↔ SRAM 之间搬运

在 Flash Attention 之前，这就是**长上下文推理的核心瓶颈**：不是算不动，是**存不下、搬不完**。

---

## 三、GPU 内存模型速览：为什么 O(N²) 访存要命

要真正理解「HBM 访问 O(N²)」的含义，必须先明白 GPU 内存的两级结构。

```
┌────────────────────────────────────────┐
│  HBM (显存)                            │  大  ≈ 80 GB
│  存放:模型权重、KV cache、中间矩阵     │  慢  ≈ 2 TB/s
└──────────────┬─────────────────────────┘
               │ 数据搬运（这里才是瓶颈！）
               ▼
┌────────────────────────────────────────┐
│  SRAM (片上共享内存 + 寄存器)          │  小  每 SM 约 100~200 KB
│  CUDA kernel 真正的计算发生在这里      │  快  ≈ 19 TB/s（快 10 倍）
└────────────────────────────────────────┘
```

**关键事实**：大部分深度学习算子的瓶颈不是"算不动"，而是**数据在 HBM 和 SRAM 之间搬来搬去的时间**。

### 传统 Attention 的 HBM 读写路径

以 N = 序列长度、d = head_dim 为例：

```
Step 1: S = Q · K^T / √d_k
  ├─ 从 HBM 读 Q                    大小 O(N·d)
  ├─ 从 HBM 读 K                    大小 O(N·d)
  ├─ 在 SRAM 算矩阵乘
  └─ 把 S 写回 HBM                  ★ O(N²)  ← 灾难来了

Step 2: P = softmax(S)
  ├─ 从 HBM 读 S                    ★ O(N²)  ← 又读一遍
  └─ 把 P 写回 HBM                  ★ O(N²)  ← 又写一遍

Step 3: O = P · V
  ├─ 从 HBM 读 P                    ★ O(N²)  ← 第三遍读
  ├─ 从 HBM 读 V                    大小 O(N·d)
  └─ 把 O 写回 HBM                  大小 O(N·d)

总 HBM 访问 ≈ O(N²)
```

### 数字有多恐怖

N = 32K, FP16 时：

```
32768 × 32768 × 2 bytes = 2 GB      ← 单头单批次的 S
32 heads × batch = 几十 GB          ← 光中间矩阵就装不下
```

**「HBM 访问 O(N²)」的准确含义**：
> 传统 Attention 每层每头需要把 O(N²) 大小的中间矩阵在 HBM 里**写一次、再读一次、再写一次**，搬运数据的总字节数正比于 N²。N 从 2K 涨到 32K，搬运量涨 256 倍——这才是长上下文卡不住的根源。

---

## 四、Online Softmax：补偿因子如何让 tile 化成为可能

传统 softmax 要求先扫一遍算全局 max，再扫一遍算分母——这两步的**跨全局 reduction** 就是 tile 化的敌人。

### 一个关键疑问

Flash Attention 要按 tile 增量处理：
- Tile 1 处理完，用了当时的局部 max = 5
- Tile 2 处理时，发现新的 max = 10

**Tile 1 的所有 exp(x − 5) 岂不是全错了？难道要回去把 Tile 1 重读一遍，用 max=10 重算？**

答案是：**不需要重读**。用一个「补偿因子」就能修正。

### 关键数学:exp 的可分解性

回忆高中数学：

```
exp(a − c) = exp(a − b) · exp(b − c)
```

代入到 softmax 里：

```
用 max=5 算的:      exp(x − 5)
用 max=10 算的:     exp(x − 10) = exp(x − 5) · exp(5 − 10)
                                = exp(x − 5) · exp(−5)
```

**核心洞察**：
> 想把「用旧 max 算的结果」修正成「用新 max 算的结果」，**只需要乘一个常数 α = exp(old_max − new_max)**。这个常数**对所有旧值都相同**！

所以不用重读所有旧值分别修正，只需要修正**累积的分母和分子**这两个数就够了——这就是 Online Softmax。

### 三个记账变量 (m, d, o)

Online Softmax 用三个滚动状态记录"目前累积到什么状态"：

| 变量 | 类型 | 物理含义 |
|------|------|---------|
| **m** | 标量 | 目前见过的最大值（数值稳定用） |
| **d** | 标量 | 目前的**分母**:`Σ exp(x − m)` |
| **o** | 向量 [head_dim] | 目前的**分子加权和**:`Σ exp(x − m) · V` |

最终 attention 输出 = **o / d**。

### 增量更新公式

来一个新 tile，分数记作 `s_tile`，值记作 `V_tile`：

```
m_new = max(m, max(s_tile))
α     = exp(m − m_new)              ← 补偿因子，把旧状态"追认"到新参考系

p     = exp(s_tile − m_new)         ← 新 tile 的 exp 项
d_new = d · α + sum(p)              ← 分母:旧的乘 α，加上新的
o_new = o · α + p @ V_tile          ← 分子:旧的乘 α，加上新的

更新: m ← m_new, d ← d_new, o ← o_new
```

**核心 trick**：`d · α` 和 `o · α` 一次乘法就完成了对整个旧累积的修正——因为 α 对所有旧值都相同，可以直接提取公因子。

**每个 tile 只被访问一次**，从头到尾**流式前向处理，绝无回读**。

---

## 五、具体演算：从头到尾追踪一次 Online Softmax

用一个可以手算的小例子把整个流程走一遍。

**设定**：分数 = `[3, 5, 10, 8]`，值 = `[V_0, V_1, V_2, V_3]`（每个 V_i 是向量）。切成 2 个 tile：Tile 1 = `[3, 5]`，Tile 2 = `[10, 8]`。

### 初始化

```
m = −∞
d = 0
o = 0
```

### 处理 Tile 1 = [3, 5]

```
局部 max = 5
m_new = max(−∞, 5) = 5
α = exp(−∞ − 5) = 0       ← 旧状态本来就是空的

p = [exp(3−5), exp(5−5)] = [0.135, 1]
d_new = 0 · 0 + (0.135 + 1) = 1.135
o_new = 0 · 0 + (0.135·V_0 + 1·V_1) = 0.135·V_0 + V_1

状态: m=5, d=1.135, o=0.135·V_0 + V_1
```

**此时 o / d 就是"只看前两个 token 的 attention 结果"**：

```
o/d = (0.135·V_0 + V_1) / 1.135 = softmax([3, 5]) · [V_0, V_1]  ✓
```

### 处理 Tile 2 = [10, 8]

```
局部 max = 10
m_new = max(5, 10) = 10
α = exp(5 − 10) = 0.0067   ← 补偿因子

修正旧状态（一次乘法搞定）:
    d_old · α = 1.135 · 0.0067 = 0.0076
    o_old · α = (0.135·V_0 + V_1) · 0.0067 = 0.0009·V_0 + 0.0067·V_1

加入 Tile 2 的贡献:
    p = [exp(10−10), exp(8−10)] = [1, 0.135]
    sum(p) = 1.135
    p @ V_tile = 1·V_2 + 0.135·V_3

新状态:
    d = 0.0076 + 1.135 = 1.143
    o = 0.0009·V_0 + 0.0067·V_1 + V_2 + 0.135·V_3
```

### 最终输出

```
output = o / d
       = (0.0009·V_0 + 0.0067·V_1 + V_2 + 0.135·V_3) / 1.143
```

### 对照一次性算的结果

```
全局 max = 10
softmax([3, 5, 10, 8]) 权重:
    exp(3−10)/1.143  = 0.0008
    exp(5−10)/1.143  = 0.0058
    exp(10−10)/1.143 = 0.875
    exp(8−10)/1.143  = 0.118

output = 0.0008·V_0 + 0.0058·V_1 + 0.875·V_2 + 0.118·V_3
```

**完全一致** ✅（除舍入误差外）。

### 直觉：像"参考系平移"

把 (m, d, o) 想象成"在某个参考点下的账本"：

```
Tile 1 处理完:  参考点 m=5,账本 (d, o) 都是"相对于 5"的表示
Tile 2 发现 max=10:  参考点要挪到 10
    "所有旧账目一起向下平移 5 个单位"
    → 整体乘以 α = exp(−5) 完成参考系换算
    → 不需要翻旧账本
```

**Online Softmax 不是"重算"，是"换算"**。所有旧结果通过一次乘法一次性追认到新的 max 下，这是 exp 函数天生的可分解性带来的礼物。

---

## 六、Online Softmax → Flash Attention

Flash Attention 就是把上面的 (m, d, o) 记账机制**塞进一个 CUDA kernel**，让所有事情都在 SRAM 里完成，中间矩阵永远不落 HBM。

### 完整伪代码

```python
def flash_attention_row(q, K, V):
    """处理 Q 的一行,K/V 按行切成 tiles"""
    m = -inf                    # 最大值
    d = 0.0                     # 分母
    o = zeros(head_dim)         # 分子加权和
    
    for K_tile, V_tile in tiles(K, V):
        # 1. 算这个 tile 的分数（在 SRAM 里，永不写回 HBM）
        s = q @ K_tile.T                    # [tile_size]
        
        # 2. 找局部 max
        m_tile = max(s)
        m_new = max(m, m_tile)
        
        # 3. 补偿因子
        alpha = exp(m - m_new)
        
        # 4. 新 tile 的 exp 项
        p = exp(s - m_new)                  # [tile_size]
        
        # 5. 增量更新 (d, o) —— 核心两行
        d = d * alpha + sum(p)
        o = o * alpha + p @ V_tile
        
        # 6. 更新 m
        m = m_new
    
    return o / d                            # 最后一步归一化
```

### 数据流对比

```
传统 Attention:  Q,K → S(HBM) → P(HBM) → O
                       ↑ N²       ↑ N²         ← 中间矩阵反复读写

Flash Attention: Q,K,V → [tile 内完成一切,只维护 (m,d,o)] → O
                          ↑
                    S 和 P 只以 tile 形式在 SRAM 中短暂存在,永不进 HBM
```

### 效果对比

| 指标 | 传统 Attention | Flash Attention |
|------|--------------|----------------|
| HBM 访问量 | O(N²) | O(N) |
| 中间矩阵显存 | O(N²) | O(N) |
| N=32K 单头 S 大小 | 4 GB（HBM 里） | 几十 KB（SRAM 里） |
| 长上下文（128K） | OOM | 正常运行 |
| 端到端加速 | 1× | 2~4× |

**实际访存比**（LLaMA-2-7B, batch=1, single head, FP16）：

| N | 传统 HBM 访问 (≈ N²) | Flash Attention (≈ N·d) | 比值 |
|---|--------------------|------------------------|------|
| 2048   | 24 MB   | 1 MB   | 24×   |
| 8192   | 384 MB  | 4 MB   | 96×   |
| 32768  | 6.1 GB  | 16 MB  | **380×** |
| 131072 | 98 GB   | 64 MB  | **1500×** |

**推论**：
- N=2K 时优化不明显（24 MB 也没什么）
- **N=32K 时提速 380×**，且能装进显存
- N=128K 时不用 Flash Attention，光中间矩阵就占 98 GB，单卡直接爆

**GPT-4、Claude、LLaMA-3 能开 128K/200K 上下文，不是它们的显卡更大，是它们全部使用了 Flash Attention 系列 kernel。没有 Flash Attention，就没有长上下文时代。**

---

## 七、一句话总结

> **Softmax 数学上只是归一化，但在 Infra 层面是数值稳定性的雷区、内存带宽的瓶颈、Attention 长上下文的关键卡点。Online Softmax 的核心 trick 是用一个补偿因子 α = exp(m_old − m_new) 把旧累积状态「一次性追认」到新的 max 参考系下，从而实现 tile-wise 增量计算；Flash Attention 在此基础上把三步 Attention 塞进一个 kernel，中间矩阵永不出 HBM，搬运量从 O(N²) 降到 O(N)——这是过去几年最重要的推理 infra 突破。**

---

## 八、与量化的关联 🛠️

- **Softmax 内部必须保 FP32 / FP16 累加**:`exp` 对输入误差指数放大，是量化禁区
- **(m, d, o) 三个记账变量必须高精度**:α = exp(m − m_new) 的微小误差会经过多个 tile 累积，掉精度很快
- **QK^T 的量化误差进 softmax 前会被放大**：所以 K 通常保 INT8，V 可以更激进（INT4）
- **Flash Attention kernel 内嵌 softmax**：AngelSlim / SmoothQuant / AWQ 都是**量化 Q/K/V/O 投影权重**，attention 核心运算（QK^T、softmax、@V）保精度

---

## 参考

- [Online Softmax to Flash Attention — and Why it Matters (Medium)](https://medium.com/data-science-collective/online-softmax-to-flash-attention-and-why-it-matters-9d676e7c50a8) —— 从 tiled matmul → 3-pass safe softmax → online softmax → Flash Attention 的完整推导脉络，本笔记的主要参考来源
- [Flash Attention 官方仓库 (Tri Dao)](https://github.com/Dao-AILab/flash-attention)
- [Online normalizer calculation for softmax (原始论文, Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)

## 相关

- [Pre-Norm Block 逐步拆解](./pre-norm-block.md) —— Softmax 所在的 Attention 上下文
- [RoPE 旋转位置编码](./rope.md) —— 与 Flash Attention 完美解耦的位置编码方案
- [SwiGLU FFN 图解](./swiglu-ffn.md) —— Block 内的另一半
