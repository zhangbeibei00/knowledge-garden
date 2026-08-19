---
sidebar_position: 0
title: 基础理论总览
---

# 📖 基础理论

> 量化模型的核心是：把高精度权重（如 FP32）映射到低精度（如 INT8/INT4）。

这个子模块把量化最基础的几件事讲清楚：数值映射怎么做、粒度怎么选、激活量化的动/静态方案怎么定。

## 内容地图

| 文档 | 关键词 | 定位 |
|------|--------|------|
| [量化本质与映射](/docs/quantization/basics/quantization-mapping) | `scale` / `zero_point` / 线性 vs 非线性 | 量化的数学基础 |
| [量化粒度](/docs/quantization/basics/granularity) | per-tensor / per-channel / per-group | 精度 vs 效率的第一权衡 |
| [激活量化：动态 vs 静态](/docs/quantization/basics/activation-quantization) | 校准 / 实时统计 | 部署时激活值怎么处理 |

## 学习顺序建议

1. 先看 **量化本质与映射**，把 `scale/zero_point` 的推导跑通
2. 再看 **量化粒度**，理解为什么大模型都在做 per-group
3. 最后看 **激活量化**，区分 W-only 和 W+A 场景

搞完这一节，就可以看 [PTQ 方法](/docs/quantization/ptq) 和 [低精度格式](/docs/quantization/low-precision-formats) 了。
