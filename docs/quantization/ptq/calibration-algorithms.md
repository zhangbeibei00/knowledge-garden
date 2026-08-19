---
sidebar_position: 1
title: PTQ 校准算法
description: MinMax / Percentile / KL / MSE 四种主流校准方法的原理与实现对比
tags: [quantization, ptq, calibration, minmax, kl-divergence, mse]
---

# PTQ 校准算法

> **校准算法**：PTQ 的核心，本质是**用校准数据统计张量分布，找到最优的量化缩放因子**，平衡精度和加速效果。

## MinMax

**核心思路**：MinMax 是最简单、最直接的量化校准算法，"用数据的全局最小值和最大值，定出量化的边界"。

```python
# 缩放系数：决定浮点数和整数的映射比例
scale = (max_val - min_val) / (2**8 - 1)  # = (max_val - min_val) / 255

# 零点：把浮点数最小值映射到整数 0 的偏移量
zero_point = -min_val / scale

# 量化（浮点数 → 整数）
q_data = round(data / scale + zero_point)

# 反量化（整数 → 浮点数）
dq_data = (q_data - zero_point) * scale
```

**优点**：实现最简单，一遍数据统计就出结果  
**缺点**：极端易被离群点带偏——只要有一个 outlier，整个 scale 就被拉大，绝大多数正常值挤在很窄的量化格里

## Percentile 百分位裁剪

**核心思路**：MinMax 会被离群点带偏，Percentile 只取分布中某一百分位（如 99.9%、99.99%）的数值作为量化上限，忽略极端值。

例：统计激活值的 99.9% 分位数作为 max，min 仍取最小值，再计算 `scale = (max - min) / (2^bit - 1)`。

```python
def percentile_calibration(data, percentile=99.9):
    """Percentile 校准：用指定分位数替代 max，避免离群点"""
    min_val = np.min(data)
    # 计算百分位的 max（忽略极端值）
    max_val = np.percentile(data, percentile)
    scale = (max_val - min_val) / 255.0
    zero_point = -min_val / scale
    return scale, zero_point, min_val, max_val
```

![Percentile 示意](./assets/percentile.gif)

**优点**：抗离群点  
**缺点**：需要选一个合理的百分位（99% / 99.9% / 99.99%），选错了要么切太多要么切太少

## KL 散度校准（KL-Divergence）

**核心思路**：衡量**量化后分布**与**原始浮点分布**的 KL 散度（差异），找到使 KL 散度最小的 scale，本质是"最小化量化带来的信息损失"。

### 步骤

1. 用校准数据统计浮点激活值的直方图
2. 遍历不同的 scale 候选值，计算量化后直方图与原始直方图的 KL 散度
3. 选择 KL 散度最小的 scale 作为最优值

**优点**：从**信息论**角度找最优，对分布敏感场景效果好（NVIDIA TensorRT 的默认 PTQ 校准算法）  
**缺点**：需要遍历候选值，比 MinMax/Percentile 慢；对直方图 bin 数敏感

## MSE 校准

**核心思路**：最小化**量化反量化后的数据**与**原始浮点数据**的均方误差（MSE），找到最优 scale。

$$
\text{MSE} = \frac{1}{N} \sum_i (x_i - \hat{x}_i)^2
$$

其中 $\hat{x}_i$ 是量化后再反量化的近似值。最小化 MSE 的 scale 即为最优。

**优点**：直接优化数值误差，直觉最强  
**缺点**：数值误差最小 ≠ 模型精度最好（分布形状可能被扭曲）

## 四种校准算法的简单实现

下面这份代码可以直接跑，对比四种算法在同一份"正态分布 + 离群点"数据上的表现。

```python
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 生成模拟校准数据（带离群点的激活值） =====================
def generate_calibration_data():
    """生成模拟的激活值数据：正态分布 + 少量离群点"""
    # 主体：正态分布（均值 0，标准差 2）
    main_data = np.random.normal(loc=0, scale=2, size=20000)
    # 离群点：随机加 10 个大值（模拟极端值）
    outliers = np.random.uniform(low=10, high=50, size=10)
    return np.concatenate([main_data, outliers])

calib_data = generate_calibration_data()
plt.hist(calib_data, bins=100, alpha=0.7, label='Activation Distribution')
plt.title('模拟激活值分布（含离群点）')
plt.xlabel('激活值')
plt.ylabel('频次')
plt.legend()
plt.show()

# ===================== 2. 通用量化/反量化函数（所有算法共用） =====================
def quantize_dequantize(data, scale, zero_point=0, bit=8):
    """
    量化 + 反量化（验证校准算法的精度）
    :param data: 原始浮点数据
    :param scale: 校准算法算出的缩放系数
    :param zero_point: 零点（非对称量化用，对称量化=0）
    :param bit: 量化比特数（INT8=8）
    :return: 反量化后的数据（模拟量化后的精度损失）
    """
    q_min, q_max = 0, 2**bit - 1  # INT8 非对称范围 0~255
    q_data = np.round(data / scale + zero_point)
    q_data = np.clip(q_data, q_min, q_max)
    dq_data = (q_data - zero_point) * scale
    return dq_data

# ===================== 3. 四种校准算法实现 =====================
def minmax_calibration(data):
    """MinMax 校准：用全局最小/最大值算 scale"""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        max_val = min_val + 1e-6
    scale = (max_val - min_val) / 255.0
    zero_point = np.clip(-min_val / scale, 0, 255)
    return scale, zero_point, min_val, max_val

def percentile_calibration(data, percentile=99.9):
    """Percentile 校准：用指定分位数替代 max，避免离群点"""
    min_val = np.min(data)
    max_val = np.percentile(data, percentile)
    if max_val == min_val:
        max_val = min_val + 1e-6
    scale = (max_val - min_val) / 255.0
    zero_point = np.clip(-min_val / scale, 0, 255)
    return scale, zero_point, min_val, max_val

def kl_divergence_calibration(data, num_bins=2048, bit=8):
    """KL 散度校准：找使量化后分布与原分布差异最小的 scale"""
    q_max = 2**bit - 1
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        max_val = min_val + 1e-6

    # 统计原始数据的直方图（分布 P）
    bins = np.linspace(min_val, max_val, num_bins + 1)
    hist, _ = np.histogram(data, bins=bins)
    hist_sum = np.sum(hist)
    if hist_sum == 0:
        return minmax_calibration(data)
    P = hist / hist_sum
    epsilon = 1e-10

    # 生成候选 max 值（99%~99.99% 分位数）
    candidate_max_list = np.percentile(data, np.arange(99.0, 99.99, 0.01))
    candidate_max_list = [c for c in candidate_max_list if c > min_val]
    if not candidate_max_list:
        return minmax_calibration(data)

    best_kl = float('inf')
    best_scale, best_zero_point, best_candidate_max = None, None, None

    for candidate_max in candidate_max_list:
        scale = (candidate_max - min_val) / q_max
        zero_point = np.clip(-min_val / scale, 0, q_max)

        # 量化数据
        q_data = np.round(data / scale + zero_point)
        q_data = np.clip(q_data, 0, q_max)

        # 统计量化后分布 Q（映射回原始 bins）
        q_hist = np.zeros_like(hist)
        for i in range(num_bins):
            mask = (data >= bins[i]) & (data < bins[i+1])
            if np.sum(mask) == 0:
                continue
            q_hist[i] = np.sum(q_data[mask])

        q_hist_sum = np.sum(q_hist)
        Q = q_hist / q_hist_sum if q_hist_sum > 0 else np.zeros_like(P)

        # 计算 KL 散度 D(P||Q)
        kl = np.sum(P * np.log((P + epsilon) / (Q + epsilon)))

        if kl < best_kl:
            best_kl = kl
            best_scale = scale
            best_zero_point = zero_point
            best_candidate_max = candidate_max

    if best_scale is None:
        return minmax_calibration(data)
    return best_scale, best_zero_point, min_val, best_candidate_max

def mse_calibration(data, bit=8):
    """MSE 校准：找使量化反量化后 MSE 最小的 scale"""
    q_max = 2**bit - 1
    min_val = np.min(data)

    candidate_max_list = np.percentile(data, np.arange(99.0, 99.99, 0.01))
    candidate_max_list = [c for c in candidate_max_list if c > min_val]
    if not candidate_max_list:
        return minmax_calibration(data)

    best_mse = float('inf')
    best_scale, best_zero_point, best_candidate_max = None, None, None

    for candidate_max in candidate_max_list:
        scale = (candidate_max - min_val) / q_max
        zero_point = np.clip(-min_val / scale, 0, q_max)
        dq_data = quantize_dequantize(data, scale, zero_point, bit)
        mse = np.mean((data - dq_data) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_scale = scale
            best_zero_point = zero_point
            best_candidate_max = candidate_max

    if best_scale is None:
        return minmax_calibration(data)
    return best_scale, best_zero_point, min_val, best_candidate_max

# ===================== 4. 运行所有算法并对比 =====================
if __name__ == "__main__":
    calib_data = generate_calibration_data()
    print(f"校准数据统计：均值={np.mean(calib_data):.2f}，"
          f"最小值={np.min(calib_data):.2f}，最大值={np.max(calib_data):.2f}")

    algos = {
        "MinMax": minmax_calibration(calib_data),
        "Percentile(99.9%)": percentile_calibration(calib_data, 99.9),
        "KL Divergence": kl_divergence_calibration(calib_data),
        "MSE": mse_calibration(calib_data),
    }

    print("\n===== 四种校准算法结果对比 =====")
    for name, (scale, zp, _, _) in algos.items():
        dq = quantize_dequantize(calib_data, scale, zp)
        mse = np.mean((calib_data - dq) ** 2)
        print(f"{name:20s}：scale={scale:.6f}, MSE={mse:.6f}")
```

### 典型输出对比

![四种校准算法结果对比](./assets/calibration-comparison.png)

**观察结论**：
- **MinMax** 被 outlier 拉大 scale，正常值区间的 MSE 反而大
- **Percentile** 通过切掉极端值，正常区间精度显著改善
- **KL/MSE** 会自动搜索最优截断点，通常比手动定百分位更稳
- 但 **KL/MSE 计算开销大**，实际部署常权衡时间成本

## 校准算法选型建议

| 场景 | 推荐算法 | 理由 |
|------|---------|------|
| **快速原型** | MinMax | 一遍数据搞定 |
| **CV 模型 PTQ** | Percentile 或 KL | 抗离群点 |
| **NVIDIA TensorRT** | KL（默认） | 官方主推 |
| **激活量化** | Percentile / KL | 激活的离群点是常态 |
| **权重量化** | MinMax 或 MSE | 权重分布相对干净 |
| **大模型（LLM）** | 走 GPTQ/AWQ 等进阶方法 | 单纯校准算法不够用 |

## 大模型场景的补充

对 LLM，单纯的校准算法（MinMax/KL/MSE）**已经不够**。主流方案：

- **AWQ**：activation-aware，识别重要通道并保护，其余走 per-group INT4
- **GPTQ**：在 per-group 基础上用 Hessian 补偿，迭代式减少每一步的量化误差
- **SmoothQuant**：把激活的离群点通过数学变换"迁移"到权重上，再走常规校准

这些方法在校准算法之外，还引入了**权重调整**或**分布重构**，本质上是校准算法的进化版。

---

## 相关文档

- [PTQ 总览](/docs/quantization/ptq)
- [量化掉点排查](./debug-quantization-loss) — 校准结果不对怎么办
- [量化本质](/docs/quantization/basics/quantization-mapping) — scale/zero_point 的数学基础
