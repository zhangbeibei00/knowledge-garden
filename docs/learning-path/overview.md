---
sidebar_position: 1
title: 学习路径总览
---

# 🧭 学习路径：量化 × RL × 推理

> 我的知识花园聚焦 AI Infra 中三个高度耦合的主题。本文给出学习地图与建议路径。

## 为什么是这三个方向

大模型时代，**训练 → 后训练 → 部署** 的完整链条上，最有工程含量的三个交叉点：

```
              ┌────────────────┐
              │   预训练模型    │
              └────────┬───────┘
                       │
              ┌────────▼────────┐
     🎯       │   RL 后训练     │       ← 我的主攻方向 1
              │  (PPO/DPO/...)  │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
     📐       │    模型量化     │       ← 我的主攻方向 2
              │  (PTQ/QAT/FP8)  │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
     🚀       │    推理引擎     │       ← 我的主攻方向 3
              │  (vLLM/SGLang)  │
              └─────────────────┘
```

三者的耦合关系：
- **量化 × 推理**：量化后的模型能否跑得又快又准，取决于推理引擎的 kernel 支持
- **RL × 量化**：RL-QAT / QAD 等新方法在训练循环中融入量化
- **RL × 推理**：RLHF 训练中的 rollout 阶段本质上就是一次大规模推理

## 推荐学习路径

### Level 1：地基（1~2 周）
- [ ] Transformer 架构（重点：KV Cache、Attention 变体）
- [ ] PyTorch 基础（自动微分、Module、分布式）
- [ ] GPU 硬件基础（SM、显存层级、Tensor Core）
- [ ] 分布式通信原语（AllReduce、AllGather）

### Level 2：单点深入（4~8 周，选一个方向）

**量化路线**：
1. 基础量化理论（对称/非对称、per-tensor/channel/group）
2. GPTQ / AWQ 论文精读
3. FP8 / NVFP4 硬件格式
4. 动手：用 llmcompressor / AutoGPTQ 跑一个量化任务

**RL 后训练路线**：
1. PPO 算法（Advantage、Clip、KL）
2. DPO / RLHF 训练流程
3. GRPO / DAPO 前沿方法
4. 动手：用 verl 跑一个 GRPO 训练

**推理引擎路线**：
1. Continuous Batching + PagedAttention（vLLM 核心）
2. Speculative Decoding
3. PD 分离架构
4. 动手：读 vLLM 关键代码（scheduler + attention）

### Level 3：跨方向融合（长期）
- 量化模型在 vLLM 上的 kernel 支持链路
- RLHF 训练中如何用 vLLM 加速 rollout
- QAD / RL-QAT 等训练时量化方法

## 阅读顺序建议

如果你和我背景类似（工程为主，需要补理论）：

1. 先看[基础知识](/docs/fundamentals)里对应的地基章节
2. 然后进[模型量化](/docs/quantization)或[RL 后训练](/docs/rl-post-training)（你更感兴趣的那个）
3. 每读完一个主题，去[推理引擎](/docs/inference-engines)里对照工程实现
4. 高质量论文进[论文精读](/docs/paper-reading)

---

*本页会持续迭代。有建议欢迎 issue 或 PR。*
