---
sidebar_position: 7
title: 稀疏 × 量化
---

# 🧩 稀疏 × 量化组合

## 🎯 面试速记版

> **BF16 稀疏 = 两个正交优化叠加**（快 + 省内存）
> **量化模型稀疏 = 两个纠缠的优化**——量化的数值噪声会放大稀疏 attention 的 mask 决策误差、softmax 溢出、精度累积等问题
>
> **6 大核心差异**：
> 1. 数值精度：BF16 稳，FP8/INT8 要 per-tile dequant + FP32 累加
> 2. Softmax 溢出：低比特下 max 分布跳变加剧
> 3. Mask 决策：动态稀疏的 top-K 依赖 score 精度，FP8 下可能选错
> 4. KV Cache：量化 KV + 稀疏 gather 的 scale 元数据对齐复杂
> 5. Kernel 融合：多了 dequant/requant 步骤，代码量翻 2-3 倍
> 6. 精度损失叠加：不是相加，是**超线性**放大
>
> **工业最佳实践**：**FP8 + Block Sparse**（DeepSeek NSA 路线），block ≥ 64，KV 量化 group 对齐 block。**不建议 INT4/INT8 + Element Sparse**。

## 1. 两者的定位对比

| 维度 | 稀疏 Attention | 量化 |
|------|--------------|------|
| **优化目标** | 减少计算量（跳过 pair） | 减少每次计算的比特数 |
| **主战场** | 时间 + 显存 | 权重 / 激活 / KV cache |
| **精度损失来源** | 丢弃 low-score pair | 数值近似 |
| **对硬件依赖** | tensor core 友好度 | tensor core 精度支持 |

**关键**：两者理论正交，但**数值精度是共享资源**——量化用掉一部分精度余量，稀疏也要吃一部分，叠加容易撞穿精度墙。

## 2. 六大核心差异

### 差异 1：数值精度与 softmax 稳定性

| 精度 | 动态范围 | 稀疏场景风险 |
|------|---------|-------------|
| BF16 | $\pm 3.4 \times 10^{38}$ | 基本无风险 |
| FP8 (E4M3) | $\pm 448$ | $QK^T$ 稍大就溢出，需 per-tile scale |
| INT8 | $[-128, 127]$ | 稀疏后 max 分布跳变加剧 |

**BF16 完全不用担心**；量化模型必须在 kernel 里**做 per-tile 反量化到 FP32 后再算 softmax**。

### 差异 2：Softmax 实现路径

**BF16 稀疏**（FA-style）：加载 → matmul → online softmax → 写回，就 4 步。

**FP8 稀疏**（FA3 / NSA）：加载 → **dequant Q** → **dequant K** → matmul → **rescale** → online softmax → **quant** → matmul(V) → **dequant output** → 写回，多了 3-4 步 dequant/requant。

### 差异 3：Sparse Mask 决策的精度依赖

**静态稀疏**（Longformer 等）：mask 定死，量化不影响。

**动态稀疏**（NSA / Reformer）：mask 由 score 决定 → 量化后 score 精度差 → top-K 可能选错。

例：真实分数 `[0.51, 0.50, 0.49]` → 量化后 `[0.50, 0.51, 0.49]`，top-1 选错。

**工业解法**：
- **NSA**：mask index 用 BF16 计算，attention 主体用 FP8
- 或用 FP32 gate 网络做 mask 决策

### 差异 4：KV Cache 量化 × 稀疏

**BF16 KV + 稀疏**：直接 gather 稀疏块，简单。

**量化 KV + 稀疏**：每个 block 有独立 scale → 元数据管理复杂 → dequant 开销占比上升。

**关键**：**KV 量化的 group size 要和 sparse block size 对齐**（比如都用 64），否则一个稀疏块横跨多个量化 group，scale 处理变噩梦。

### 差异 5：Kernel 融合成本

- **BF16 稀疏 kernel**：~500-1000 行 Triton
- **FP8 稀疏 kernel**：~1500-3000 行，调优难度飙升

普通团队几乎写不出高性能 FP8 + Sparse kernel，只能用 DeepSeek 等团队的开源实现。

### 差异 6：精度损失叠加（超线性）

**独立损失**（大致）：
- BF16 稀疏（50%）：< 0.5%
- FP8 dense：< 1%
- **FP8 + 稀疏**：**1.5-2.5%**（不是 1.5%）

**为什么超线性**：稀疏 top-K 决策依赖激活精度 → 激活量化后 mask 抖动 → 稀疏本身也变差。

**解法**：**渐进式训练**——先 BF16 稀疏，再 QAT 到 FP8。

## 3. 工业方案对比

| 方案 | 精度 | 稀疏 | 代表 |
|------|------|------|------|
| BF16 + FA | BF16 | 无 | 基线 |
| BF16 + Block Sparse | BF16 | 静态 block | Longformer / Mistral SWA |
| FP8 Dense FA3 | FP8 | 无 | LLaMA-3 FP8 推理 |
| **FP8 + Block Sparse** 🌟 | FP8 | 静态/动态 block | **DeepSeek-V3.2 NSA** |
| INT4 + Block Sparse | INT4 | 静态 block | 私有部署，精度损失大 |
| ❌ INT8 + Element Sparse | INT8 | element | 学术论文，不推荐 |

## 4. 三个坑

- **分布 shift**：量化后 score 分布"变扁"，top-K 阈值需要重新校准
- **Block 太小 quant 失效**：block < 64 时 scale 统计不稳。**block ≥ 64 是硬性要求**
- **QAT + 稀疏训练**：梯度经过 sparse mask 和 quant STE 两层近似，需分阶段训练

## 5. 加速收益速览

```
                    相对 BF16 dense
BF16 Dense ──────────────  1×
BF16 + Block Sparse ─────  1.8×
FP8 Dense (H100) ────────  1.6×
FP8 + Block Sparse 🌟 ───  2.8×   ← 甜点
FP8 + Dynamic Block ─────  3.5×
INT4 + Block Sparse ─────  4.0×（精度损失大）
```

**FP8 + Block Sparse 是当前 SOTA 甜点**：约 2.5-3× 加速，精度损失 < 2%。

## 6. 关键结论

- **BF16 稀疏**：两个优化独立，损失可控
- **量化 + 稀疏**：精度纠缠，必须**统一设计**（scale 对齐、mask 决策路径分离）
- **只做 block-level**：element-level 稀疏 + 量化在 GPU 上必然拉胯
- **首选 FP8**：不要尝试 INT4 + Sparse，除非能接受 3-5% 精度损失

## 相关文档

- 前一篇：[是否需要定制算子](/docs/inference-engines/optimization/sparse-attention/custom-kernels)
- 返回：[稀疏 Attention 总览](/docs/inference-engines/optimization/sparse-attention)
- 交叉：[量化基础](/docs/quantization/basics) / [PTQ 方法](/docs/quantization/ptq)
