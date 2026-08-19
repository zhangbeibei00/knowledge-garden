---
sidebar_position: 1
title: 量化本质与映射
description: 从 scale/zero_point 出发理解量化的数学基础，区分线性与非线性量化
tags: [quantization, basics, scale, zero-point]
---

# 量化本质与映射

## 一句话：量化就是一次仿射映射

量化的本质，就是把一个连续的浮点数值域映射到一个离散的整数值域：

$$
q = \mathrm{round}\!\left(\frac{x}{\text{scale}}\right) + \text{zero\_point}
$$

反量化：

$$
x \approx \text{scale} \times (q - \text{zero\_point})
$$

两个参数：
- **scale**：浮点→整数的比例尺，决定"一格代表多大的浮点范围"
- **zero_point**：偏移量，让浮点 0 能精确对应到某个整数（非对称量化用）

## 对称量化 vs 非对称量化

| 类型 | zero_point | 适用分布 | 常见场景 |
|------|-----------|---------|---------|
| **对称量化** | `= 0` | 数值以 0 为中心（如权重） | 权重量化、INT8 对称 |
| **非对称量化** | `≠ 0` | 分布偏移（如 ReLU 后激活） | 激活量化、INT8 非对称 |

对称量化的好处是硬件实现简单（没有 zero_point 偏移），坏处是当分布不对称时会浪费 bit 位（比如 ReLU 输出全正，还用一半 bit 表示负数就是浪费）。

## INT8 的映射示例

INT8 有 256 个可表示值（-128~127 或 0~255）。以对称量化为例：

```python
# 假设权重范围是 [-2.0, 2.0]，INT8 对称量化范围 [-127, 127]
scale = max(abs(min_val), abs(max_val)) / 127  # = 2.0 / 127
zero_point = 0

# 量化
q = round(x / scale)      # x=1.5 → q = round(95.25) = 95
q = clip(q, -127, 127)

# 反量化
x_hat = q * scale         # 95 * (2/127) ≈ 1.496，损失 0.004
```

## 线性量化 vs 非线性量化

### 线性量化（Uniform Quantization）
- 量化步长**均匀**，就是上面公式的做法
- 优点：硬件友好、计算快
- 缺点：对分布不均匀的数据（如激活值长尾）不友好

### 非线性量化（Non-Uniform Quantization）
- 量化步长**不均匀**，通常在数据密集的区域步长小，稀疏区域步长大
- 典型代表：**浮点数本身就是非线性量化**（FP 的步长中间密两边稀疏）
- 也包括对数量化、K-Means 聚类量化等

![FP 步长分布：中间密，两边稀](../low-precision-formats/assets/fp-step-distribution.png)

FP 格式的这个特性天然适合权重和激活的分布（大部分值集中在 0 附近，少量离群点分布在远端），这也是为什么 FP8/FP4 在大模型量化里越来越流行。

## 量化误差从哪来

- **round 操作**：连续值取整必然引入 rounding error
- **clip 操作**：超出量化范围的值被截断
- **scale 精度**：scale 本身也是浮点数，存储和计算都有精度损失
- **零点偏移**：非对称量化时 zero_point 也可能引入误差

误差累积到模型输出，就体现为掉点。所以后面所有的 PTQ/QAT 方法，本质都是在想办法压制这几类误差。

---

## 相关文档

- 下一步：[量化粒度](./granularity) — 决定 scale 是每个 tensor 一个，还是每列 / 每组一个
- [激活量化](./activation-quantization) — 激活值的 scale 是提前算好还是实时算
- [INT8 详解](/docs/quantization/low-precision-formats/int8) — INT8 的具体实现
