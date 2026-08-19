---
sidebar_position: 1
title: STE 直通估计器
description: 解决量化操作不可导的核心技巧，"前向用量化，反向假装没量化"
tags: [quantization, qat, ste, straight-through-estimator]
---

# STE 直通估计器

STE（**Straight-Through Estimator**）是 QAT 的核心操作，它的作用是**解决量化操作不可导**这个基础问题。

## 为什么需要 STE？

反向传播（Backpropagation）更新参数依赖于**导数（梯度）**。

- **浮点模型**：$y = f(x)$，函数顺滑，导数处处存在，反向传播畅通无阻
- **量化模型**：$y = \mathrm{Round}(f(x))$

当你对数据进行**取整（Round）**或**截断**操作时，函数变成了阶梯状：

- 在阶梯的**平地上**，导数为 0
- 在**台阶跳跃处**，导数无穷大（或无定义）

**核心问题**：反向传播时，梯度会直接"断流"，模型无法学习。

## STE 的核心原理

一句话：**"前向用量化，反向假装没量化"**。

它像一个临时的"翻译官"：

- **正向 `forward`**：严格执行离散操作（量化、采样、硬决策等）
- **反向 `backward`**：忽略离散操作的梯度（因其为 0 或无定义），**直接将后层传来的梯度 `grad_output` 作为当前层的梯度 `grad_input`**，实现"梯度直通"

这就是名字"Straight-Through"的由来。

## PyTorch 实现示例

以二值化（binarization）为例，把 STE 写成一个自定义 autograd Function：

```python
import torch
from torch.autograd import Function


class BinarizeSTE(Function):
    @staticmethod
    def forward(ctx, x):
        # 正向传播：二值化，输出 ±1（离散操作）
        return torch.sign(x)  # x>0 → 1, x<0 → -1, x=0 → 0（可微调为 1）

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：梯度直通（忽略 sign 操作，直接传递梯度）
        # 可选：限制梯度范围（如 clip 到 [-1, 1]，增强稳定性）
        grad_input = grad_output.clone()
        return grad_input.clamp_(-1, 1)  # 梯度截断，避免梯度爆炸


# 使用示例
x = torch.tensor([-0.5, 2.0, -3.0], requires_grad=True)
binarized_x = BinarizeSTE.apply(x)   # 正向输出：tensor([-1.,  1., -1.])
loss = binarized_x.sum()
loss.backward()                       # 反向：x.grad = [1., 1., 1.]（梯度直通）
print(x.grad)                         # 输出：tensor([1., 1., 1.])
```

**关键点**：
- `forward` 里做了不可导的 `sign` 操作
- `backward` 里直接把 `grad_output` 传出去，绕过了 `sign` 的导数

## 常见的 STE 变体

STE 的核心思路很简单，但为了训练稳定，实际中会做一些改进：

### 1. 梯度截断（Clipping）

```python
def backward(ctx, grad_output):
    return grad_output.clamp_(-1, 1)
```

避免梯度爆炸，特别是在二值/三值量化这种极端场景。

### 2. Saturating STE（饱和 STE）

如果原始值超出量化范围，梯度置 0：

```python
def backward(ctx, grad_output):
    x, = ctx.saved_tensors
    grad_input = grad_output.clone()
    grad_input[x.abs() > 1] = 0  # 超出 [-1, 1] 的位置梯度置 0
    return grad_input
```

这样量化范围外的值不会被"拽回来"，训练更稳定。

### 3. 可学习的量化步长（LSQ）

LSQ（Learned Step Size Quantization）把 scale 本身当作可学习参数，用 STE 处理 scale 的梯度：

```python
class LSQQuantize(Function):
    @staticmethod
    def forward(ctx, x, scale, q_min, q_max):
        ctx.save_for_backward(x, scale)
        ctx.q_min, ctx.q_max = q_min, q_max
        q = torch.clamp(torch.round(x / scale), q_min, q_max)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        # 对 x：STE 梯度直通，但饱和区域置 0
        grad_x = grad_output.clone()
        grad_x[(x / scale < ctx.q_min) | (x / scale > ctx.q_max)] = 0
        # 对 scale：可学习，用近似梯度
        grad_scale = ...  # 见 LSQ 论文推导
        return grad_x, grad_scale, None, None
```

## QAT 完整工作流程

STE 是核心机制，QAT 的完整流程是这样的：

![QAT 完整工作流程](./assets/qat-workflow.png)

### 1）准备阶段

- 使用**量化模拟封装层**替换敏感层（如卷积、全连接、激活函数）
- PyTorch 实现：通过 `prepare_qat` 或 `prepare_qat_fx` 接口完成

### 2）训练阶段

- **前向传播**时对权重和激活值进行"伪量化"（模拟 INT8/INT4 的取整和截断操作）
- **反向传播**采用 STE，让梯度计算忽略量化操作的导数影响
- 模型学习到"如何在被量化的情况下依然给出好答案"

### 3）转换阶段

- 训练完成后，通过 `convert` 或 `convert_fx` 将伪量化模块替换为**真实量化算子**
- 最终生成可直接执行高效 INT8/INT4 推理的部署模型

## STE 的局限性

STE 是**有偏估计**——它假装量化函数的梯度是 1（或近似为 1），但真实梯度并不是。这带来一些实际问题：

- **训练前期收敛慢**：梯度方向和真实分布不完全吻合
- **对超参数敏感**：lr、weight_decay、饱和策略都影响很大
- **极低比特下失效**：INT2/binary 场景，STE 的偏差累积严重，需要更复杂的方法（如 DoReFa、BiReal Net）

现代大模型 QAT 基本都在 STE 的基础上叠了很多改进——LSQ 的可学习 scale、Softmax-based soft quantization、量化蒸馏（QAD）等等。

---

## 相关文档

- [QAT 总览](/docs/quantization/qat)
- [PTQ 掉点排查](/docs/quantization/ptq/debug-quantization-loss) — 什么时候必须上 QAT
- [QAD vs OPD](/docs/quantization/practice-notes/qad-vs-opd) — QAT 的现代替代方案
