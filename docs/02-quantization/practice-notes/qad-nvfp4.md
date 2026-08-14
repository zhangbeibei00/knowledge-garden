---
sidebar_position: 1
title: QAD - NVFP4 量化感知蒸馏
description: NVIDIA 提出的量化感知蒸馏方法，解决 NVFP4 量化后的精度恢复问题
tags: [quantization, distillation, nvfp4, qat, qad]
format: md
---

:::info 🌳 Evergreen · 论文精读
NVIDIA 2026-03 · arXiv:2601.20088v3
:::

# 面向 NVFP4 推理精度恢复的量化感知蒸馏 (QAD)

**Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery**

**作者：** Meng Xin, Sweta Priyadarshi, Jingyu Xin, Bilal Kartal, Aditya Vavre, Asma Kuriparambil Thekkumpate, Zijia Chen, Ameya Sunil Mahabaleshwarkar, Ido Shahaf, Akhiad Bercovich, Kinjal Patel, Suguna Varshini Velury, Chenjie Luo, Zhiyu Cheng, Jenny Chen, Chen-Han Yu, Wei Ping, Oleg Rybakov, Nima Tajbakhsh, Oluwatobi Olabiyi, Dusan Stosic, Di Wu, Song Han, Eric Chung, Sharath Turuvekere Sreenivas, Bryan Catanzaro, Yoshi Suhara, Tijmen Blankevoort, Huizi Mao

**联系人：** huizim@nvidia.com
**发表日期：** 2026-03-04（arXiv:2601.20088v3）
**版权：** © 2026 NVIDIA. All rights reserved.

---

## 摘要

本技术报告介绍了**量化感知蒸馏（Quantization-Aware Distillation, QAD）**以及在恢复 NVFP4 量化后的大语言模型（LLM）和视觉语言模型（VLM）精度方面的最佳实践。QAD 使用 KL 散度损失，将全精度的教师模型蒸馏为量化的学生模型。虽然将蒸馏应用于量化模型并非新想法，但我们观察到 QAD 对当今的 LLM 具有以下关键优势：

1. **对经过多阶段后训练流水线（包括 SFT、强化学习 RL、模型合并）训练的模型表现出显著的有效性和稳定性**——而传统的量化感知训练（QAT）在这些场景下面临工程复杂性和训练不稳定性的问题；
2. **对数据质量和覆盖度具有鲁棒性**——无需完整训练数据即可恢复精度。

我们在多个后训练模型上评估了 QAD，包括：**AceReason Nemotron**、**Nemotron 3 Nano**、**Nemotron Nano V2**、**Nemotron Nano V2 VL (VLM)** 和 **Llama Nemotron Super v1**，结果显示这些模型均可稳定恢复至接近 BF16 的精度。

**NVFP4 权重检查点：** Nemotron 3 Nano 30B-A3B、Nemotron Nano 9B V2、Nemotron Nano 12B V2 VL、Llama 3.1 Nemotron Nano VL 8B

**QAD 代码：** Megatron-LM 版本、NeMo 版本、HuggingFace Transformers 版本

---

## 1. 引言

大语言模型（LLM）的迅猛发展带来了对更高效数值格式的需求，以降低训练和推理阶段的计算成本、内存需求和能耗。8 位浮点格式（FP8 和 MXFP8）已成为加速 LLM 训练的流行数据类型（Micikevicius et al., 2022；DeepSeek-AI, 2024）。窄精度硬件的最新进展（NVIDIA, 2024）将 4 位浮点（FP4）推向下一个合乎逻辑的阶段（Chmiel et al., 2025；Chen et al., 2025b；Liu et al., 2023a；Rouhani et al., 2023），相比 FP8，它带来了 **2~3 倍的算力提升**和**约 50% 的内存占用降低**。

**NVFP4** 采用更小的 block 大小（16 vs. MXFP4 的 32）、FP8 缩放因子（E4M3）用于细粒度缩放，以及第二级 FP32 缩放以扩展动态范围，在许多模型上都表现出优于 INT4 和 MXFP4 的精度（Egiazarian et al., 2025）。对于超大 LLM，使用**训练后量化（PTQ）**的 NVFP4 在多种基准上表现良好。然而，对于小模型，PTQ 带来的精度下降往往不可忽视。

业界有多项工作尝试将量化融入训练过程。NVFP4 量化训练（NVIDIA, 2025）在预训练任务上已展现出良好的收敛性；但其主要目标是训练加速，训练出来的模型仍在 BF16 下评估。

**量化感知训练（QAT）**（Jacob et al., 2018）是一种恢复推理精度的有效训练方法，自 CNN 时代以来一直有良好的效果。许多 QAT 方法复用原始高精度模型的训练流水线和目标函数，但对于现代 LLM 而言，这种做法面临重大挑战：

1. **训练流水线复杂：** 现代 LLM 通常经历多阶段的后训练（Bakouch et al., 2025；Bercovich et al., 2025），例如 SFT、RL、模型合并，导致难以复现原始训练流程。
2. **数据可用性和质量：** 开源模型的原始训练数据可能不公开，而公开数据集的质量通常较差。

本技术报告在多个后训练 LLM 上评估了 QAD 用于 NVFP4 推理精度恢复的效果。**QAD 使用原始全精度模型作为教师，通过 KL 散度损失（而非任务专用目标）训练量化后的学生模型。** 我们的主要发现如下：

1. **QAD 比 QAT 更好地将量化模型对齐到高精度模型。**
2. **QAD 对多阶段后训练流水线（包括 SFT & RL）的模型有效，具备显著的稳定性。**
3. **QAD 对不完整的数据覆盖具有鲁棒性，即使仅有部分领域数据也能恢复精度，实现跨领域知识迁移。**

**报告组织：** 第 2 节介绍 NVFP4 与量化方法的背景；第 3 节介绍 QAD 方法和训练设置，并通过 LLM 和 VLM 上的全面结果展示其关键特性；第 4 节详细分析设计选择和训练数据的影响。

---

## 2. 背景与相关工作

### 2.1 NVFP4 格式与训练后量化

**NVFP4 格式。** NVFP4 是一种 4 位浮点格式，专为现代 GPU 架构上的高效训练和推理而设计（NVIDIA, 2025；Alvarez et al., 2025）。相比 FP8，NVFP4 提供 **2-3× 的算力吞吐**和 **约 1.8× 的内存缩减**。NVFP4 在 MXFP4（Rouhani et al., 2023）的基础上进行了扩展：

- **更小的 block 大小**（由 32 缩为 16）
- **两级缩放**（每块 E4M3 缩放 + 每张量 FP32 缩放）

更小的 block 大小实现了对数据分布更局部化的适配，E4M3 缩放提供非二次幂的缩放因子以降低量化误差，第二级 FP32 缩放则扩展了整体动态范围（Alvarez et al., 2025）。

**训练后量化（PTQ）。** PTQ 是一种简单且低成本的方法，无需训练（Nagel et al., 2021；Wu et al., 2020；Vanhoucke et al., 2011）。它通过**校准（calibration）**过程，用一小部分代表性数据（校准集）确定量化参数（例如缩放因子和零点）。使用 PTQ 时，训练和推理完全解耦，对无法访问原始训练流水线或没有微调计算资源的实践者非常有吸引力。

最简单的 PTQ 方法是 **max 校准**，它将校准数据中的最大绝对值映射到量化格式的最大可表示值。尽管简单，max 校准在许多情况下效果出奇地好。更复杂的校准方法包括：

- **基于 MSE 的校准**（Jacob et al., 2018）
- **KL 散度最小化**（Migacz, 2017）
- **可学习的裁剪阈值**（Choi et al., 2018）

针对 LLM 的 PTQ 高级方法包括：

- **AdaRound**（Nagel et al., 2020）和 **BRECQ**（Li et al., 2021）：优化权重舍入和块级重构
- **ZeroQuant**（Yao et al., 2022）和 **GPTQ**（Frantar et al., 2023）：逐层校准与高效近似
- **AWQ**（Lin et al., 2023）：保留敏感信息
- **OmniQuant**（Shao et al., 2024a）：通过梯度下降优化量化参数
- **基于变换的 PTQ**：应用可逆变换（旋转、仿射变换）抑制异常值，例如 SmoothQuant、QuaRot、QuIP#、SpinQuant、FlatQuant
- **低秩近似方法**：SVDQuant（Li et al., 2024）和 EoRA（Liu et al., 2025b）

PTQ 对大多数 LLM 的 FP8 量化效果很好。对于 NVFP4，PTQ 在**非常大的模型**上也表现良好（见附录 C）。但 PTQ 通常在**小模型和敏感任务上表现不佳**。最新研究表明，常见的 PTQ 算法往往无法在基线 NVFP4 性能上取得进一步提升，因为**小 block 大小会削弱传统异常值抑制技术的效果**（Egiazarian et al., 2025）。

### 2.2 量化感知训练（QAT）

量化感知训练（QAT）（Jacob et al., 2018）用于在模型完成全精度训练后恢复其推理精度。与**原生量化训练**（DeepSeek-AI, 2024；NVIDIA, 2025）不同——后者将权重、激活和梯度全部量化以加速前向和反向传播——**QAT 仅量化权重和激活**，因此只加速前向传播。**梯度保留高精度**以确保收敛稳定。

QAT 使用与原始全精度模型相同的损失函数（例如语言建模的交叉熵）微调量化模型，理想情况下使用相同的训练数据。现代 LLM 通常经历包含 SFT 和 RL 阶段的多阶段后训练流水线。对 SFT 阶段应用 QAT 很直接；对 RL 阶段，等效做法是**量化感知 RL（QARL）**，即量化 actor 的前向传播和 rollout 生成。但 QARL 尚未被充分探索——现有量化 RL 工作主要关注**加速训练吞吐**（Huang et al., 2025；Liu et al., 2025a），而非事后的推理精度恢复。这促使我们研究 QAD 作为 RL 训练模型的替代方案。

### 2.3 面向量化的知识蒸馏

知识蒸馏（KD）（Hinton et al., 2015）通过让学生模型学习教师模型的软标签（如概率分布）来传递知识，分类任务通常使用 KL 散度损失。理论工作表明，软标签能以更低的方差更好地估计真实类别概率（Menon et al., 2020），并且匹配教师分布可提供隐式正则化以加速收敛（Phuong and Lampert, 2021）。

知识蒸馏与量化的结合在 CNN 上已被广泛研究：

- **Mishra and Marr (2017)：** 证明低精度网络可通过全精度教师蒸馏显著改善。
- **Polino et al. (2018)：** 提出量化蒸馏，将蒸馏损失直接融入权重量化网络的训练。
- **Kim et al. (2019)：** 提出三阶段训练流程以减轻 KD 对量化模型的强正则化效应。

对于 LLM：

- **Liu et al. (2023b)：** 证明基于模型生成数据的**无数据蒸馏**非常有效，可实现无需原始训练数据的量化感知训练。
- **Kim et al. (2023)：** 提出 **token-scaled logit distillation**，根据教师置信度重加权每 token 的 KL 散度以防过拟合。
- **Du et al. (2024)：** 提出 **BitDistiller**，针对 sub-4-bit LLM，将非对称量化与融合前向和反向 KL 散度的自蒸馏目标结合。

先前的 QAD 工作主要针对整数量化。**本报告证明：简单的 KL 散度蒸馏对 NVFP4 推理精度恢复非常有效**，特别是对经过复杂多阶段后训练的 LLM——在这些场景下复现原始训练流程不切实际，而标准 QAT 存在破坏模型能力的风险。

---

## 3. 量化感知蒸馏（QAD）

### 3.1 方法概述

QAD 使用原始高精度模型作为教师来训练量化模型（学生），使用**蒸馏损失**。它与标准 QAT 的区别在于损失函数（如图 1 所示）：

- **QAT：** 使用与原始模型训练流水线相同的任务专用损失，例如语言建模的 next-token 交叉熵。
- **QAD：** 使用高精度教师与量化学生之间的 **KL 散度**。

形式化地，对于输入 $x$ 和词表 $V$，设 $p_{teacher}(y|x)$ 为全精度教师模型的输出分布，$p_{student}(y|x)$ 为量化学生模型的输出分布。QAD 损失定义为：

$$
\mathcal{L}_{QAD} = D_{KL}(p_{teacher} \| p_{student}) = \sum_{y \in V} p_{teacher}(y|x) \log \frac{p_{teacher}(y|x)}{p_{student}(y|x)}
$$

**表 1** 揭示了 QAD 与 QAT 的关键区别：**QAT 几乎达到与 BF16 基线相同的交叉熵损失**，看起来能力恢复很成功；**但 KL 散度告诉了不同的故事：QAD 的 KL 散度几乎为零，而 QAT 相对教师有显著散度**。这说明 QAT 虽然能匹配验证损失，但**显著改变了模型的输出分布，实际上像是额外的一次后训练**。相反，**QAD 忠实地保留了原始 BF16 模型的输出分布**。

**表 1：QAD 更好地将模型对齐到 BF16 基线**（Llama Nemotron Super V1，用其 SFT 数据集中约 0.3B tokens 训练，在 5,000 个留出样本约 8M tokens 上评估）

| 方法 | KL 散度（vs BF16） | 交叉熵（vs labels） |
|------|-----------------|-----------------|
| BF16 | 0 | 0.408 |
| QAT  | 0.311 | 0.408 |
| QAD  | 0.004 | 0.416 |

**图 1：QAT 与 QAD 的对比。** QAT 使用目标数据集上的 next-token 预测（交叉熵）进行训练，而 QAD 使用蒸馏损失（KL 散度），由全精度教师模型提供软目标。

### 3.2 QAD 用于后训练模型

对于**单阶段后训练**：我们在 Nemotron Nano 12B v2 VL（NVIDIA, 2025b）上测试后发现，QAT 可以达到与 QAD 相当的性能（见附录 A）。然而，现代 SOTA 大模型通常经历包含 SFT、RL、模型合并的复杂多阶段流水线（Bercovich et al., 2025；NVIDIA, 2025a），对这类流水线应用 QAT 具有挑战性：需要用量化前向重现每个阶段，并复现阶段间的模型合并流程。**更实用的方法是使用 SFT 数据或 RL 提示生成数据的混合，在单一阶段中运行 QAD 或 QAT。** 在这种简化设置下，我们证明 QAD 一致优于 QAT，无论原始流水线多复杂，都能达到接近 BF16 的精度。

**表 2：SFT-heavy 模型上的结果**（QAD 和 QAT 均使用模型的 SFT 数据混合训练——数学、代码、科学等）

| 模型 | 方法 | MATH500 | AIME25 | GPQA-D | IFEval-Instruction |
|------|------|---------|--------|--------|-------------------|
| **Llama Nemotron Super V1** | BF16 | 95.8 | 46.0 | 66.5 | 87.5 |
|  | NVFP4 PTQ | 91.4 | 32.3 | 62.1 | 86.9 |
|  | NVFP4 QAT | 94.3 | 41.5 | 63.3 | 87.2 |
|  | **NVFP4 QAD** | **94.6** | **45.6** | **64.5** | **87.8** |
| **Nemotron Nano V2** | BF16 | 97.8 | 71.1 | 64.0 | 90.3 |
|  | NVFP4 PTQ | 97.2 | 69.8 | 59.0 | 89.8 |
|  | NVFP4 QAT | 97.2 | 67.1 | 56.9 | 86.2 |
|  | **NVFP4 QAD** | **97.2** | **71.5** | **62.7** | **89.3** |

**表 3：RL-heavy 模型上的 QAD 结果。** 两个模型上，NVFP4 QAT 均破坏了 RL 模型的能力，而 QAD 成功恢复到接近 BF16 的性能——证明**蒸馏对 RL 模型是必要的**。

**(a) Nemotron 3 Nano²**

| 方法 | AA-LCR | AIME25 | GPQA-D | LiveCodeBench-v5 | SciCode |
|------|--------|--------|--------|------------------|---------|
| BF16 | 35.9 | 89.1 | 73.0 | 72.1 | 33.0 |
| NVFP4 PTQ | 31.3 | 85.0 | 71.6 | 68.9 | 30.5 |
| NVFP4 QAT | 24.8 | 83.3 | 66.0 | 62.0 | 25.8 |
| **NVFP4 QAD** | **34.3** | **87.9** | **72.7** | **68.9** | **32.3** |

**(b) AceReason Nemotron 1.1 7B**

| 方法 | AIME24 | AIME25 | LiveCodeBench-v6 |
|------|--------|--------|------------------|
| BF16 Baseline | 73.0 | 63.5 | 54.3 |
| NVFP4 PTQ | 69.4 | 58.7 | 52.0 |
| NVFP4 QAT | 62.1 | 46.1 | 45.9 |
| **NVFP4 QAD** | **71.7** | **62.0** | **53.3** |

> ² 由于使用了不同的评估工具和设置，基准数字与模型卡略有不同。

**SFT-heavy 模型。** 我们在两个经历多阶段后训练的 SFT-heavy 模型上评估 QAD：**Llama Nemotron Super V1 49B**（Bercovich et al., 2025）和 **Nemotron Nano 9B V2**（NVIDIA, 2025a）。表 2 显示，QAD 在两个模型的高难推理基准上均一致优于 QAT：

- **Llama Nemotron Super V1：** QAD 在 AIME25 上比 QAT 高 **+4.1%**，在 GPQA-D 上高 **+1.2%**，恢复到接近 BF16 的性能。
- **Nemotron Nano 9B V2：** QAD 达到接近 BF16 的性能，在 AIME25 上比 QAT 高 **+4.4%**，在 GPQA-D 上高 **+5.8%**。

**Nemotron Nano 12B v2 VL** 是在预训练后经历单一 SFT 阶段的 VLM 模型，结果见附录 A。

**RL-heavy 模型。** 我们在两个 RL-heavy 模型上评估 QAD：

- **Nemotron 3 Nano 30B-A3B**（NVIDIA, 2025）：混合 Mamba-Transformer，经过多阶段 RL 后训练；
- **AceReason Nemotron 1.1 7B**（Chen et al., 2025a；Liu et al., 2025c）：基于 Qwen2.5 的模型，通过 RL 专门优化了数学和代码能力。

对于 RL 训练模型，RL 训练数据通常只包含提示（模型在训练中生成响应）。但 RL 模型通常从**冷启动 SFT 阶段**初始化，该阶段教会基座模型用**链式思考推理**解决问题（DeepSeek-AI, 2025；Chen et al., 2025a；team, 2025b；Kimi Team, 2025；Ling Team, 2025）。冷启动 SFT 数据是 QAD 和 QAT 训练的实用选择。如果没有冷启动 SFT 数据，还可以使用 RL 提示的生成数据（见 4.1 节）。

在我们的实验中：Nemotron 3 Nano 用冷启动 SFT + RL 生成数据的混合训练；AceReason 仅用冷启动 SFT 数据训练。

**然而使用冷启动 SFT 数据（或包含它的混合数据）对 QAT 是根本性挑战：因为它可能破坏 RL 训练中学到的能力。** 表 3 显示：即使训练数据包含 RL 生成样本，QAT 相对 PTQ 也显著降低了所有基准上的性能。相反，**QAD 成功恢复到接近 BF16 的性能，因为它匹配教师的输出分布而非从数据分布重新学习。** 这证明**蒸馏对 RL 训练模型的精度恢复是必要的**。另一种方法是将 QAT 融入 RL 训练过程本身，这是一个活跃的研究方向。

这些结果凸显了 QAD 对 RL 模型的关键优势：**它避开了 RL 训练的复杂性和破坏已学能力的风险，只需要全精度教师模型。**

### 3.3 对不完整领域覆盖的鲁棒性

QAD 的一个关键优势是**对不完整数据覆盖的鲁棒性**——即使训练数据不覆盖模型的所有领域或能力，也能恢复精度。

**跨领域迁移（多领域模型）。** AceReason Nemotron 在数学和代码领域训练。表 4 显示，使用部分数据的 QAD（仅数学或仅代码）在所有基准上几乎与完整数据（数学+代码）匹配。**引人注目的是，仅用代码数据训练的 QAD 也能恢复强劲的数学性能**，展示了通过蒸馏实现的有效跨领域知识迁移。

**表 4：AceReason Nemotron 1.1 7B 上部分领域覆盖的 QAD 结果**

|  | AIME24 | AIME25 | LiveCodeBench-v6 |
|---|--------|--------|------------------|
| BF16 Baseline | 73.0 | 63.5 | 54.3 |
| NVFP4 PTQ | 69.4 | 58.7 | 52.0 |
| NVFP4 QAD（仅数学） | 71.0 | 61.7 | 53.1 |
| NVFP4 QAD（仅代码） | 71.0 | 62.0 | 53.3 |
| NVFP4 QAD（数学+代码） | 71.7 | 62.0 | 53.3 |

这些结果证明：**教师的输出分布隐式编码了所有领域和能力的知识**，即使输入数据来自有限领域。通过训练量化学生匹配这些分布，QAD 实现了跨领域知识迁移，使学生能在训练数据中未显式呈现的领域上近似教师行为。

### 3.4 训练与评估设置

**量化配置：**

- **Llama Nemotron Super V1 和 AceReason Nemotron：** 所有 GEMM 层量化为 NVFP4。
- **Nemotron Nano 9B V2**（混合 Mamba-Transformer，4 层 Transformer + 52 层 Mamba）：**选择性量化**——注意力层与首尾各 2 层保留 BF16 以获得更好的 PTQ 基线。
- **Nemotron 3 Nano**（MoE 混合 Mamba-Transformer，仅 6 层 self-attention）：6 层 self-attention 与其前置 Mamba-2 层保留 BF16，其余部分量化为 NVFP4，**KV-Cache 量化为 FP8**。

**超参数：**

- **学习率：** Llama Nemotron Super V1 和 Nemotron Nano 9B V2 用 1e-6；AceReason Nemotron 和 Nemotron 3 Nano 用 1e-5。
- **Softmax 温度：** 教师和学生均设 $T=1$，以精确匹配教师输出分布。
- **batch size 和序列长度：** 与原始后训练相似。

**数据量：** QAD 所需数据显著少于原始后训练。具体所需数据取决于模型大小和任务复杂度：

- Llama Nemotron Super V1 49B：**约 0.3B tokens**
- Nemotron Nano 9B V2：**约 6B tokens**
- Nemotron Nano 12B V2 VL：**约 0.5B tokens**
- Nemotron 3 Nano 30B-A3B：**约 2.5B tokens**
- AceReason Nemotron 7B：**约 0.8B tokens**

**评估：** 每个实验评估验证损失最低的 top 10 checkpoint，选择在评估基准上平均表现最好的。

- **Llama Nemotron Super V1、Nemotron Nano 9B V2、AceReason Nemotron：** AIME 2024/2025 各 48 次采样、LiveCodeBench 12 次、GPQA-D 20 次、IFEval 5 次；温度 $T=0.6$、top-p $=0.95$。
- **Nemotron 3 Nano：** AIME 2025 16 次、LiveCodeBench-v5 8 次、GPQA-D 8 次、AA-LCR 与 SciCode 各 5 次；SciCode 报告子任务精度；温度 $T=1.0$、top-p $=1.0$。

---

## 4. 消融研究

本节针对训练数据质量和学习率敏感性进行详细的消融研究。

### 4.1 训练数据质量

QAD 一个重要的实用问题是：对数据质量和来源的敏感性。我们在 AceReason Nemotron 上进行消融研究以理解不同数据源的影响（Nemotron 3 Nano 观察到类似的鲁棒性，详见附录 B）。

**表 5：训练数据对 AceReason Nemotron 1.1 7B 的 QAD 影响**

测试了五种数据源：
1. **SFT 数据：** 训练时使用的原始冷启动 SFT 数据
2. **RL 提示生成：** BF16 从 RL 提示生成的样本
3. **RL 提示生成（仅正确）：** 过滤后仅包含正确解
4. **BOS token 生成：** 用单个初始 token 生成的数据（Liu et al., 2023b）
5. **随机 token：** 完全随机的 token 序列

| 训练数据 | AIME24 | AIME25 | LiveCodeBench-v6 |
|---------|--------|--------|------------------|
| BF16 Baseline | 73.0 | 63.5 | 54.3 |
| NVFP4 PTQ | 69.4 | 58.7 | 52.0 |
| SFT 数据 | 71.7 | 62.0 | 53.3 |
| RL 提示生成 | 71.9 | 61.3 | 52.6 |
| RL 提示生成（仅正确） | 70.5 | 61.6 | 52.3 |
| BOS token 生成 | 70.1 | 60.9 | 52.4 |
| 随机 token | 68.6 | 60.0 | 51.7 |

冷启动 SFT 数据和 BF16 从 RL 提示生成的数据都能达到接近 BF16 的性能，表明**合成数据对 QAD 高度有效**。有趣的是，**使用全部生成样本（含错误的）比只用正确样本表现更好**，说明错误生成也为蒸馏提供了有用信息。**即使用完全随机 token，QAD 也保持了与 PTQ 基线相当的性能且不破坏模型**，展现出显著的稳定性。

Nemotron 3 Nano 在不同数据源下也观察到类似鲁棒性（详见附录 B）。这些结果证明 QAD **对数据源和质量具有显著鲁棒性**。

### 4.2 学习率

QAD 需要仔细选择学习率，最优范围因原始训练方式而异。**总体建议学习率介于 1e-5 到 1e-6 之间**。

**对于 SFT 训练的模型：** 使用等于或低于原始后训练学习率的值效果最好。

**表 6：AceReason Nemotron 1.1 7B（RL-heavy）与 Nemotron Nano 9B V2（SFT-heavy）的学习率敏感性**

对 AceReason，最优 LR（1e-5）明显高于标准 RL 学习率；对 Nano V2 9B，超过原始 SFT LR（1e-6）会降级或导致训练不稳定。

| 模型 | 学习率 | AIME24 | AIME 25 | LiveCodeBench |
|------|--------|--------|---------|---------------|
| **AceReason Nemotron 1.1 7B** | 1e-6 | 70.8 | 61.0 | 52.6 |
|  | 5e-6 | 71.0 | 60.9 | 53.2 |
|  | **1e-5** | **71.7** | **62.0** | **53.3** |
|  | 1e-4 | 72.4 | 61.8 | 53.0 |
| **Nemotron Nano 9B V2** | **1e-6** | 80.4 | **71.5** | **67.8** |
|  | 5e-6 | 80.0 | 71.0 | 66.8 |
|  | 1e-5 | 80.8 | 69.4 | 67.4 |
|  | 1e-4 | 78.8 | 65.2 | 64.0 |

**表 7：Nemotron Nano 12B v2 VL（SFT 训练）的学习率敏感性**

该模型在 LR=2e-6 时达到最优——**比其原始 SFT LR（2e-5）低 10 倍**。

| 学习率 | AI2D | ChartQA | DocVQA | InfoVQA | OCRBench | TextVQA |
|--------|------|---------|--------|---------|----------|---------|
| 1e-4 | 67.0 | 76.0 | 75.0 | 47.6 | 685 | 70.6 |
| 2e-5 | 85.3 | 87.6 | 91.6 | 72.2 | 820 | 82.8 |
| **2e-6** | **87.1** | **89.7** | **94.0** | **78.9** | **857** | **84.7** |

Nemotron Nano 9B V2 在 1e-6 时（与原始 SFT 学习率一致）达到最优，Nemotron Nano 12B v2 VL 在 2e-6 时表现最好。**更高的学习率会降级性能甚至发散**，可能是因为这些模型后训练后已在 SFT 数据分布上充分收敛。

**对于 RL 训练的模型：** 情况明显不同。由于最后的 RL 阶段将模型推离冷启动 SFT 数据分布，**QAD 从更大的学习率中受益**。表 6 显示，对 AceReason Nemotron，最优 QAD 学习率是 **1e-5**，远高于典型的 RL 学习率（约 1e-6）（Shao et al., 2024b；Luo et al., 2025；Chen et al., 2025a）。

### 4.3 其他设计选择

**KL 散度 vs. MSE。** QAD 使用 KL 散度作为蒸馏损失——这是匹配概率分布的标准选择。虽然理论上可以使用 logits 上的 MSE 等其他距离度量，但表 8 显示 **KL 散度在各基准上一致优于 MSE**。这可能是因为 KL 散度更适合衡量分布差异，并能为概率匹配提供更好的梯度。

**表 8：不同模型上 KL 散度 vs. MSE**

| 模型 | 损失 | GPQA-D | AIME24 | AIME25 | LiveCodeBench |
|------|------|--------|--------|--------|---------------|
| **AceReason Nemotron 1.1 7B** | KL-Div | / | 71.7 | 62.0 | 53.3 |
|  | MSE | / | 71.7 | 60.1 | 52.4 |
| **Nemotron Nano 9B V2** | KL-Div | 62.7 | 80.4 | 71.5 | 67.8 |
|  | MSE | 60.3 | 80.0 | 71.5 | 66.7 |

**使用更大的教师。** 与传统知识蒸馏（大教师传递知识给小学生）不同，QAD 使用原始 BF16 模型作为教师以恢复其精确分布。**使用同系列的更大教师也是可行的**，因为它们通常用相似数据和技术训练。我们在 Nemotron Nano 9B V2 上测试了两个教师：原始 9B BF16 和更大的 12B BF16。

**表 9：Nemotron Nano 9B V2 NVFP4 用不同教师蒸馏的结果**

| 教师 | AIME24 | AIME25 | LiveCodeBench |
|------|--------|--------|---------------|
| **9B BF16** | **80.4** | **71.5** | **67.8** |
| 12B BF16 | 80.2 | 69.8 | 66.7 |

结果显示 **9B 教师优于 12B 教师**。一个潜在原因是：**适配不同分布需要比典型 QAD 更多的训练数据**。为高效恢复精度，我们仍建议使用原始模型作为教师。

---

## 5. 结论

本技术报告提出**量化感知蒸馏（QAD）**，作为恢复 LLM 和 VLM 在 NVFP4 量化后推理精度的实用有效方法。通过在 Nemotron Nano、Nemotron Nano VL、Llama Nemotron Super 和 AceReason Nemotron 上的实验，我们证明：**QAD 可靠地将 NVFP4 模型带回接近 BF16 的精度**，包括那些经过复杂 SFT、RL、模型合并流水线训练、复现原始训练流程对标准 QAT 不切实际的模型。

我们的消融研究进一步证明：**QAD 对数据覆盖和质量鲁棒**——可以利用部分领域或合成数据，甚至用随机 token 训练也能保持稳定。加上相比原始后训练的**较低数据与计算需求**，这些特性使 QAD 成为 PTQ 不足时 NVFP4 精度恢复的**实用默认方案**。

NVFP4 QAD 检查点和代码已在摘要中链接提供，方便实践者在真实部署中采用这些技术。

---

## 附录 A：Llama Nemotron VL 结果

Nemotron Nano 12B v2 VL 是在预训练后经历单一 SFT 阶段的 VLM 模型。**与其他后训练 LLM 不同，QAT 对该模型达到与 QAD 相当的精度**——我们认为这归因于简单的训练流水线以及 PTQ 带来的小精度下降。

**表 10：Llama Nemotron Nano 12B v2 VL 在不同方法下的精度对比**

数据集：AI2D（Kembhavi et al., 2016）、ChartQA（Masry et al., 2022）、DocVQA（Mathew et al., 2021b）、InfoVQA（Mathew et al., 2021a）、OCRBench（Liu et al., 2024a）、TextVQA（Singh et al., 2019）

| 方法 | AI2D | ChartQA | DocVQA | InfoVQA | OCRBench | TextVQA |
|------|------|---------|--------|---------|----------|---------|
| Baseline | 87.3 | 89.7 | 94.3 | 79.3 | 855 | 85.2 |
| PTQ | 86.8 | 89.6 | 93.8 | 78.2 | 850 | 84.8 |
| QAT | 86.5 | 89.8 | 93.7 | 78.3 | 848 | 84.8 |
| QAD | 86.7 | 89.4 | 93.9 | 78.4 | 858 | 85.2 |

---

## 附录 B：Nemotron 3 Nano 数据质量消融

对 Nemotron 3 Nano，我们测试了三种数据源：(1) 仅冷启动 SFT 数据，(2) 仅 BF16 从 RL 提示生成的数据，(3) 两者的混合。**表 11 显示三种数据源表现相似，SFT+RL 混合略好**。

**表 11：训练数据对 Nemotron 3 Nano 30B-A3B QAD 的影响**

| 训练数据 | AA-LCR | AIME25 | GPQA-D | LiveCodeBench-v5 | SciCode |
|---------|--------|--------|--------|------------------|---------|
| BF16 Baseline | 35.9 | 89.1 | 73.0 | 72.1 | 33.0 |
| NVFP4 PTQ | 31.3 | 85.0 | 71.6 | 68.9 | 30.5 |
| SFT 数据 | 32.6 | 86.0 | 72.7 | 70.0 | 31.7 |
| RL 提示生成 | 34.0 | 82.7 | 73.9 | 70.4 | 33.1 |
| SFT+RL 生成混合 | 34.3 | 87.9 | 72.7 | 68.9 | 32.3 |

---

## 附录 C：大模型的 PTQ

一个经验发现是：**更大的 LLM 对量化更鲁棒**。表 12 各给出一个 NVIDIA 训练模型和社区模型的示例。更多结果和量化检查点可在 HuggingFace - NVIDIA 集合中找到。

**表 12：大模型（数百亿参数）上的 PTQ 结果**

大模型对 NVFP4 PTQ 具有鲁棒性，无需任何微调即可达到接近原始的精度。

| 模型 | 方法 | MATH500 | AIME24 | GPQA-D | GSM8K |
|------|------|---------|--------|--------|-------|
| **Llama Nemotron Ultra V1 (253B)** | BF16 | 96.6 | 75.0 | 75.7 | / |
|  | NVFP4 PTQ | 96.2 | 76.0 | 74.8 | / |
| **DeepSeek R1 (671B)** | Official FP8 | 95.4 | 80.0 | 69.7 | 96.3 |
|  | NVFP4 PTQ | 94.2 | 80.0 | 69.2 | 96.1 |

---

## 附录 D：QAT/QAD 与原生量化训练的对比

**原生量化训练**源自混合精度训练（FP16/BF16），二者共同目标是降低训练计算成本。原生量化训练主要用于**预训练**——此时训练是**计算受限**的（因为 batch size 可以任意增大）。核心思想是量化三个 GEMM：**前向传播 (Fprop)、权重梯度 (Wgrad)、数据梯度 (Dgrad)**。为在低精度下执行这些 GEMM，三者所有输入（激活、权重、输出梯度）都必须量化。例如 DeepSeek V3 的 FP8 训练（DeepSeek-AI, 2024）以及最近的 NVFP4/MXFP4 预训练（NVIDIA, 2025；Chmiel et al., 2025；Chen et al., 2025b；Rouhani et al., 2023）。这些方法专注于降低从头开始预训练前沿模型的成本。

**QAT/QAD** 的量化目标类似——均量化权重和/或激活，梯度保持高精度。因此**反向传播中的两个 GEMM（Wgrad 与 Dgrad）无法用低精度计算**。图 2 说明了 QAT 与量化训练的关键区别。

**图 2：QAT 与原生量化训练对比。** QAT 仅量化前向传播用于推理恢复，原生量化训练量化全部三个 GEMM（Fprop、Wgrad、Dgrad）以降低训练成本。**QAD 的计算图与 QAT 类似**。

---

## 参考文献（节选）

主要参考文献包括：

- **Alvarez et al. (2025)：** Introducing NVFP4 for efficient and accurate low-precision inference. NVIDIA Developer Blog.
- **Ashkboos et al. (2024)：** QuaRot: Outlier-free 4-bit inference in rotated LLMs. arXiv:2404.00456.
- **Bercovich et al. (2025)：** Llama-Nemotron: Efficient reasoning models. arXiv:2505.00949.
- **Chen et al. (2025a)：** AceReason-Nemotron: Advancing math and code reasoning through RL. arXiv:2505.16400.
- **DeepSeek-AI (2024)：** DeepSeek-V3 technical report. arXiv:2412.19437.
- **DeepSeek-AI (2025)：** DeepSeek-R1: Incentivizing reasoning capability in LLMs via RL. arXiv:2501.12948.
- **Du et al. (2024)：** BitDistiller: Unleashing the potential of sub-4-bit LLMs via self-distillation. arXiv:2402.10631.
- **Egiazarian et al. (2025)：** Bridging the gap between promise and performance for microscaling FP4 quantization. arXiv:2509.23202.
- **Frantar et al. (2023)：** GPTQ: Accurate post-training quantization for generative pre-trained transformers. arXiv:2210.17323.
- **Hinton et al. (2015)：** Distilling the knowledge in a neural network. arXiv:1503.02531.
- **Jacob et al. (2018)：** Quantization and training of neural networks for efficient integer-arithmetic-only inference. CVPR.
- **Lin et al. (2023)：** AWQ: Activation-aware weight quantization for LLM compression and acceleration. arXiv:2306.00978.
- **Liu et al. (2023b)：** LLM-QAT: Data-free quantization aware training for LLMs. arXiv:2305.17888.
- **Micikevicius et al. (2022)：** FP8 formats for deep learning. arXiv:2209.05433.
- **NVIDIA (2025)：** Pretraining large language models with NVFP4. arXiv:2509.25149.
- **NVIDIA (2025a)：** NVIDIA Nemotron Nano 2: An accurate and efficient hybrid Mamba-Transformer reasoning model. arXiv:2508.14444.
- **NVIDIA (2025b)：** NVIDIA Nemotron Nano V2 VL. arXiv:2511.03929.
- **Rouhani et al. (2023)：** Microscaling data formats for deep learning. arXiv:2310.10537.
- **Xiao et al. (2023)：** SmoothQuant: Accurate and efficient post-training quantization for LLMs. ICML.
- **Yao et al. (2022)：** ZeroQuant: Efficient and affordable post-training quantization for large-scale transformers. NeurIPS.

（完整参考文献列表见原论文第 11-15 页）

---

**翻译说明：**
- 本翻译基于原文 arXiv:2601.20088v3（2026-03-04 版本）
- 保留关键技术术语英文原文（如 QAT、QAD、NVFP4、GEMM、KL 散度、Softmax、logits 等）
- 表格数据完全保留原始数值
- 公式使用 LaTeX 语法呈现
- 关键结论用**粗体**突出

---

*翻译时间：2026-07-21 · 由开发虾 🛠️ 生成*
