---
sidebar_position: 4
title: INT8
description: 经典的 8-bit 整数量化，PTQ 时代的老将
tags: [quantization, int8, ptq]
---

# INT8

> 参考：[CSDN — INT8 量化原理](https://blog.csdn.net/Nichlson/article/details/121085747)

INT8 是量化领域最经典、最"老实"的格式——纯整数，均匀步长，8 位覆盖 -128~127。CV 时代几乎所有 PTQ 方案都是 INT8，直到 LLM 时代才逐渐让位给 FP8 / INT4。

## PTQ 原理

INT8 PTQ 的核心公式极其简单：

$$
q = \mathrm{round}\!\left(\frac{x}{\text{scale}}\right) + \text{zero\_point}
$$

反量化：

$$
x = \text{scale} \times (q - \text{zero\_point})
$$

![INT8 PTQ 原理](./assets/int8-ptq.png)

## INT8 的两种模式

### 对称量化

- `zero_point = 0`，范围 `[-127, 127]`（`-128` 常保留不用）
- 硬件实现最简单，没有偏移量
- 适合**权重量化**（权重分布通常以 0 为中心）

```python
scale = max(abs(x)) / 127
q = round(x / scale)
q = clip(q, -127, 127)
```

### 非对称量化

- `zero_point ≠ 0`，范围 `[0, 255]`
- 能更好利用 8 位表示范围
- 适合**激活量化**（ReLU 后全正，或分布明显偏移）

```python
scale = (max(x) - min(x)) / 255
zero_point = round(-min(x) / scale)
q = round(x / scale) + zero_point
q = clip(q, 0, 255)
```

## 为什么大模型不用 INT8 而选 FP8

INT8 在 CV 上一直很稳，但到了 LLM 就吃力。核心原因：

1. **激活值长尾分布**：Transformer 的激活值经常有离群点，INT8 的均匀步长处理不了
2. **动态范围窄**：INT8 有效范围就 ±127，跨数量级的激活值会被压得死死的
3. **无法优雅覆盖训练场景**：INT8 训练非常难做（梯度是浮点，需要频繁反量化）

FP8 的指数位天然覆盖大范围、非线性步长贴合数据分布，成了 LLM 时代的新宠。

## INT8 依然活跃的场景

虽然被 FP8/INT4 抢了不少风头，但 INT8 在以下场景仍然主流：

- **边缘部署**：手机、嵌入式设备的 NPU 大多只支持 INT8
- **CV 模型**：ResNet、YOLO 等的 PTQ 部署，INT8 精度损失可控
- **SmoothQuant**：LLM 的 INT8 W+A 量化经典方案，通过平滑激活离群点让 INT8 也能用
- **INT8 权重 + FP16 激活**：一种保守的 LLM 部署方案（如早期 GPTQ 变体）

## INT8 vs FP8 快速对比

| 维度 | INT8 | FP8 (E4M3) |
|------|------|-----------|
| **表示能力** | 256 个均匀值 | 256 个非均匀值 |
| **数值范围** | ±127 | ±448 |
| **步长** | 均匀 | 中间密、两边稀 |
| **对离群点** | 差 | 好 |
| **硬件支持** | 全平台通吃 | H100+/Blackwell |
| **典型场景** | CV 部署、边缘设备 | LLM 训练推理 |

## 相关 PTQ 方法

INT8 场景下的常见 PTQ 算法：

- [MinMax 校准](/docs/quantization/ptq/calibration-algorithms#minmax) — 最简单
- [Percentile 校准](/docs/quantization/ptq/calibration-algorithms#percentile-百分位裁剪) — 抗离群点
- [KL 散度校准](/docs/quantization/ptq/calibration-algorithms#kl-散度校准kl-divergence) — 最小化信息损失
- [MSE 校准](/docs/quantization/ptq/calibration-algorithms#mse-校准) — 最小化数值误差

---

## 相关文档

- [FP8 详解](./fp8) — 现代大模型的 8-bit 主流
- [PTQ 校准算法](/docs/quantization/ptq/calibration-algorithms) — INT8 具体怎么校准
- [量化本质](/docs/quantization/basics/quantization-mapping) — INT8 是线性均匀量化的代表
