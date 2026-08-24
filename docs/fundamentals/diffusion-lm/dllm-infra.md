---
sidebar_position: 4
title: dLLM 的 AI Infra 改进方向
description: 从 AI Infra 视角看 dLLM 的挑战与机会 —— KV Cache 为何不适用、Attention kernel 怎么改、量化的新战场、serving 优化路线、以及蚂蚁/Mercury 已经做了什么
tags: [dllm, llada, infra, kv-cache, kernel, quantization, serving, flash-attention]
---

# dLLM 的 AI Infra 改进方向

主流大模型推理框架（vLLM、SGLang、TensorRT-LLM、AngelSlim）都是为**自回归**设计的。dLLM/LLaDA 这套新范式对 Infra 提出了完全不同的挑战：**KV Cache 不好用了，Attention 需要重写，量化路线要重设计**。

这既是**痛点**，也是**机会** —— 对 Infra 工程师来说，这是一片没被完全开发的绿地。

这篇按下面这条脉络展开：

```
1. 痛点全景: 为什么现有 infra 直接搬不能用
2. KV Cache 的困境与三条突围路
3. Attention Kernel: 从 Causal Flash 到 Bidirectional Flash
4. 量化: 新的主战场在哪
5. Serving 优化: NFE / 混合调度 / 投机解码
6. AR-to-Diffusion: 权重迁移带来的量化红利
7. 蚂蚁 / Mercury / iLLaDA 已经做了什么
8. 一份可执行的 dLLM Infra 改进 checklist
```

---

## 一、痛点全景：为什么现有 infra 直接搬不能用

先把 AR 和 dLLM 在 infra 关键指标上的差异摆开：

| Infra 指标 | AR (LLaMA/Qwen) | dLLM (LLaDA) | 结论 |
|-----------|----------------|--------------|------|
| **KV Cache 复用** | ✅ 完美 | ⚠️ 需改造 | Attention 每步重算 |
| **Attention Mask** | Causal (下三角) | Bidirectional (无) | Kernel 要重编 |
| **单次前向复杂度** | O(L²) | O(L²) | 相同 |
| **总推理次数** | L 次（一 token 一次） | N 次（N 是超参） | dLLM 可远小于 L |
| **一次前向吞吐** | 1 token | L / N token | dLLM 每步更多 |
| **端到端延迟** | L × single_step | N × full_forward | 取决于 N |
| **首 token 延迟 (TTFT)** | 极低（只算 prompt） | 高（要跑完 N 步） | ⚠️ dLLM 弱项 |
| **流式输出** | ✅ 天然 | ❌ 需 Semi-AR 拆 block | ⚠️ dLLM 弱项 |
| **显存占用** | KV Cache 主导 | 激活主导 | 内存结构完全不同 |

**核心结论**：**dLLM 不是"更快的 AR"，而是"另一套 trade-off"** —— 在长序列 + 编辑场景强，在低延迟流式弱。infra 优化要因材施教。

---

## 二、KV Cache 的困境与三条突围路

### 2.1 为什么 KV Cache 天然不适用

自回归下，第 i 步生成时 K/V 只依赖前 i-1 个 token —— **前面的 K/V 不会变**，所以可以缓存下来，每步只算新 token 的 K/V。这是 vLLM PagedAttention、SGLang 等所有推理框架的地基。

dLLM 下：
- 双向 attention，**每个位置的 K/V 都依赖整个序列**
- 每一步去噪，**全序列的 token 都可能变**（remask 会改变 x_t）
- **前一步的 K/V 在下一步已经失效**

朴素做法：**每步重算所有 K/V** → 显存和算力浪费严重。

### 2.2 突围路一：GQA + Tied Embedding（iLLaDA 的方案）

**iLLaDA** 通过两个改造让 KV Cache **部分复用**：

- **GQA (Grouped-Query Attention)**：多个 query head 共享一组 K/V head，直接把 KV Cache 大小压缩 4-8×
- **Tied Embedding**：共享输入 embedding 和输出 LM head，参数量减少，显存留出空间给 Cache

**关键洞察**：虽然 dLLM 每步都要重算 K/V，但**如果 mask 位置分布稳定**（比如已经填了的 token 不再变），**未 mask 位置的 K/V 可以在多步之间复用**。

iLLaDA 论文明确说：
> GQA + Tied Embedding 的工程优化组合，支持 **KV-Cache 机制适配**并减少参数量

这就打开了 dLLM 上的 KV Cache 之门。

### 2.3 突围路二：Block Diffusion + KV Cache

**Block Diffusion**（Arriola et al., ICLR 2025）走 **半自回归 + 块内扩散**路线：

```
序列切分:  [Block 1] [Block 2] [Block 3] [Block 4]
              ↓         ↓         ↓         ↓
          块内并行     块内并行    块内并行   块内并行
              ↓         ↓         ↓         ↓
          左到右 AR    左到右 AR   左到右 AR

Block 之间: 传统 AR KV Cache 有效!
Block 内部: dLLM 迭代填空
```

**这是当前生产级 dLLM 最主流的路线** —— 兼顾了两者优势：
- 块间保留 KV Cache
- 块内并行加速
- 支持流式输出（每 block 完成一次吐一次）

### 2.4 突围路三：稀疏活跃位置计算

关键观察：**每步 remask 后，只有一部分位置是 [MASK]**（比如从 200 个 M 变成 196 个 M）。

**理论上**：**只需要重算 [MASK] 位置的 K/V**（已经确定的位置的 K/V 不需要变）。

**实际做法**：
- 类似稀疏 Attention，只对 mask 位置计算 output
- 未 mask 位置只是被"读"，不需要重算 output
- 这需要**专门的稀疏 kernel**，比 Flash Attention 还复杂

**SGLang 有 RFC 集成 Block dLLM 框架**（iwiki 提到），也在朝这个方向走。

---

## 三、Attention Kernel：从 Causal Flash 到 Bidirectional Flash

### 3.1 Flash Attention 在 dLLM 上要怎么改

之前花园讲过 [Flash Attention](../transformer/softmax-online-flash.md) 是 causal 版本。dLLM 用双向 attention，几个具体改动：

**Causal 版 Flash Attention 关键代码（Triton 伪代码）**：

```python
@triton.jit
def flash_attn_causal(Q, K, V, ...):
    for start_n in range(0, N, BLOCK_N):
        # ⚠️ Causal mask 关键: 只处理 start_n <= 当前 Q 的位置
        if start_n > current_q_pos:
            break                          # ← Causal 版可以早停

        k, v = load_block(K, V, start_n)
        qk = dot(q, k.T)
        qk = apply_causal_mask(qk)          # ← 加下三角 mask
        # ... online softmax
```

**Bidirectional 版**：

```python
@triton.jit
def flash_attn_bidirectional(Q, K, V, ...):
    for start_n in range(0, N, BLOCK_N):
        # ✅ 双向: 不需要早停,也不需要 causal mask
        k, v = load_block(K, V, start_n)
        qk = dot(q, k.T)                    # ← 直接算
        # ... online softmax
```

**看起来更简单**！但工程上：
- 需要**重新编译**一份 kernel
- **计算量是 causal 的 2×**（causal 只算下三角，双向要算全矩阵）
- 但**内存访问模式完全一样**，Flash Attention 的 tile 化 + online softmax 全部可用

### 3.2 dLLM 的 Attention 计算量分析

假设 L=2048，A100（312 TFLOPS FP16）：

| 场景 | Attention FLOPs | 单步耗时 |
|------|----------------|---------|
| AR causal（每步只算新 K/V） | O(L) | ~0.1ms |
| dLLM 双向（每步全序列） | O(L²) | ~26ms |

**这就是为什么 dLLM 的 N 必须够小** —— 如果 N=200，那 200 步双向 attention = 5.2s，远慢于 AR 的 200 × 0.1ms = 20ms。

**优化空间**：
- Bidirectional Flash Attention（现成的库如 `flash-attn` 里有 non-causal 模式）
- 稀疏 kernel 只算活跃位置
- 混合精度（FP8）

### 3.3 现成的库能不能直接用

**能用一半**：
- `flash-attn` 官方库：`causal=False` 就是双向版
- `xformers.ops.memory_efficient_attention`：默认无 mask 就是双向
- Triton 官方 tutorial：把 causal check 去掉就是双向

**不足**：
- 没有针对 dLLM 的**稀疏 mask 优化**（部分位置是 [MASK] 时的加速）
- 没有针对**多步迭代**的 K/V 部分复用

**Infra 工程机会**：把 Flash Attention v2/v3 fork 一份，写一个 **dLLM 专用版本**，加上稀疏活跃位置支持。

---

## 四、量化：新的主战场在哪

你目前在做的量化工具链，如果要跟上 dLLM，需要重新审视几个问题：

### 4.1 量化目标的变化

| 组件 | AR 下的地位 | dLLM 下的地位 |
|------|------------|--------------|
| **KV Cache 量化** | ⭐⭐⭐ 最重要 | ⭐ 需要重设计 |
| **权重量化 (W4A16 / W8A8)** | ⭐⭐ | ⭐⭐⭐ 更重要（每步都用） |
| **激活量化** | ⭐⭐ | ⭐⭐⭐ 每步激活量级大 |
| **Q/K/V 投影量化** | ⭐⭐ | ⭐⭐⭐ 双向 attention 更敏感 |

**核心结论**：**dLLM 下量化的主战场从 KV Cache 转向权重 + 激活**。

### 4.2 量化误差累积效应

AR 的量化误差累积是**线性**的（生成 L 个 token，量化误差累积 L 次）。

dLLM 的量化误差是**"跨 NFE 累积"** —— 每步 N 都要跑一次完整前向，量化误差会**在多次去噪之间放大**。

**推论**：
- dLLM 对量化位宽可能要比 AR **更保守**（比如同样任务 AR 能用 INT4，dLLM 可能要 INT8）
- 需要研究**多步稳定性**：随着 N 步迭代，量化误差如何演化

### 4.3 校准数据要重设计

主流量化算法（GPTQ、AWQ、SmoothQuant）的校准逻辑：

```python
for calib_sample in calib_data:
    # AR: 直接跑一次 forward
    logits = model(calib_sample)
    collect_activation_stats()
```

**dLLM 上的问题**：
- **单次 forward 只是完整推理的 1/N**
- 需要**跑 N 步迭代的完整推理**，采集每步的激活分布
- 每步的 x_t 分布不同（前几步 mask 多，后几步 mask 少）
- **需要 mask 感知的校准**

**这是一个开放问题**，AngelSlim 类工具需要**新的 dLLM adapter**。

### 4.4 具体可以做的量化研究

- **W4A16 for LLaDA**：把 GPTQ 适配到掩码扩散前向
- **INT8 双向 Attention Kernel**：Q/K/V 都用 INT8，累加用 FP16
- **多步稳定性分析**：INT4 vs INT8 在 N=10/50/100 步下的质量曲线
- **AR-to-Diffusion 量化迁移**：把已量化的 LLaMA 权重加载进 LLaDA 继续训练

---

## 五、Serving 优化：NFE / 混合调度 / 投机解码

### 5.1 关键指标：NFE (Number of Function Evaluations)

对 dLLM 来说，**NFE = 采样步数 N** 是最核心的效率指标。所有优化都围绕**降低 NFE 且不损质量**。

三条路：

**A. 蒸馏（Distillation）**
- Teacher: N=100 步的高质量模型
- Student: N=10 步的快速模型
- 类比 Consistency Models 在图像扩散的做法

**B. Flow Map / Consistency**
- 学习"直接从噪声跳到目标"的映射
- FMLM（2026）声称一步质量 > 早期离散 dLLM 的 8 步

**C. 编辑式细化**
- ME-DLM（ICML 2026）：**1/8 步数 + HumanEval +11.6 分**
- 先出全序列草稿，再学最小编辑（replace/delete/insert）

### 5.2 混合调度：AR + dLLM 路由

生产实践中，**没必要用一种模型解决所有问题**。iwiki 的深度研究报告给出了明确建议：

| 场景 | 推荐 | 原因 |
|------|------|------|
| IDE 补全、实时对话、JSON 工具调用 | **AR** | TTFT 低，流式，格式严格 |
| 代码 patch、批量重构、文档改写 | **DLM 或 AR+DLM** | 局部编辑强项 |
| 受约束文案、多属性改写 | **DLM** | 中间状态引导 |
| 长文本、RAG 问答 | **AR 主 + DLM 局部重写** | 长度控制成本 |
| 多模态统一 | **DLM 有潜力** | 需专门验证 |

**Infra 侧要做**：
- **模型路由器**：根据请求类型选 AR / DLM
- **统一 serving 框架**：一套 API 后面挂两种模型
- **KV Cache 共享**：AR 的 prompt encoding 结果可复用给 DLM 起始状态

### 5.3 投机解码适配 dLLM

AR 的 speculative decoding：小模型草稿 → 大模型验证 → 一次接受多 token。

**SimSD**（2026 预印本）为 dLLM 设计了 reference tokens + 特殊 attention mask，让 draft token 能在目标模型单次前向中验证。作者报告 SDAR-family dLLM 上 **7.46× 解码吞吐**。

**Infra 意义**：投机解码这条路在 dLLM 上**已经可行**，值得进一步工程化。

---

## 六、AR-to-Diffusion：权重迁移带来的量化红利

**DiffuLLaMA、Dream 7B、LLaDA 2.0** 都用了 **AR 权重初始化 + 继续扩散预训练**的路线。这对量化工作有一个**巨大红利**：

```
已量化的 LLaMA-3-8B 权重
        ↓ (加载)
LLaDA base (初始化)
        ↓ (继续训练:掩码扩散目标)
LLaDA 8B 扩散版
        ↓ (你的量化流程)
LLaDA W4A16
```

**推论**：
- LLaMA 系列已经积累的**量化经验、校准数据集、GPTQ/AWQ 参数**可以**部分迁移**
- 量化流程不用从零重跑
- **AngelSlim 可以复用 90% 的现有 pipeline**，只改 forward 逻辑

这条路让 dLLM 的量化**门槛大幅降低**。

---

## 七、蚂蚁 / Mercury / iLLaDA 已经做了什么

回顾一下前沿的具体 infra 工作，作为你的**参考**：

### 7.1 LLaDA 2.0 / 2.1 / 2.2（蚂蚁）

- **MoE 架构**：100B 总参数，16B 活跃 —— 和 Mixtral / DeepSeek-V2 同路线
- **AR 权重初始化**：不从零训练，用预训练 AR checkpoint 改造
- **Token Editing (2.1)**：不只 remask，主动纠错低置信度 token，推理速度提到 892 TPS
- **序列编辑纠错 (2.2)**：面向 Agent 的进一步优化
- **128K 上下文**：证明 dLLM 也能做长上下文

### 7.2 Mercury（Inception Labs）

- **1000+ TPS** on H100，**5-10× 快于同规模 AR**
- **自适应步数调度**：简单补全 2-4 步，复杂推理 8-10 步
- **半自回归混合**：长文本 block-wise，块内并行
- **蒸馏加速**：Teacher-Student 得到端侧模型

### 7.3 iLLaDA

- **GQA + Tied Embedding** 支持 KV Cache 适配
- **12T tokens 预训练**
- **统一 SFT 格式**（不再只 mask response）
- **置信度评分机制**（多选题评估）

### 7.4 SGLang RFC

- 有 PR / RFC 集成 **Block dLLM 框架**
- 一旦合并，**vLLM 之外的开源 dLLM serving 落地**

---

## 八、一份可执行的 dLLM Infra 改进 Checklist

如果你要在 dLLM 方向做 infra 工作，我推荐这份优先级：

### 阶段 1 · 基础适配（1-2 周）

- [ ] Fork 一份 Flash Attention，编译 **bidirectional 版本**（去掉 causal mask）
- [ ] 跑通 LLaDA-8B 的 baseline 推理（HuggingFace transformers 支持）
- [ ] Benchmark: N=10 / 50 / 100 步下的 TTFT、TPOT、总延迟、显存
- [ ] 对比同规模 AR (LLaMA-3-8B) 在同一 prompt 下的性能

### 阶段 2 · KV Cache 探索（2-3 周）

- [ ] 实现**朴素 KV 全量重算**版本（baseline）
- [ ] 实现**Block Diffusion 版本**（块间 KV Cache）
- [ ] 尝试**GQA-based 部分复用**（未 mask 位置 K/V 缓存）
- [ ] 对比三种方案的显存 vs 速度 trade-off

### 阶段 3 · 量化研究（3-4 周）

- [ ] 把 GPTQ 适配到 LLaDA forward（mask-aware calibration）
- [ ] W4A16 baseline 测试，看 N 步稳定性
- [ ] 测**AR 量化权重直接加载到 LLaDA**能否 work
- [ ] AWQ / SmoothQuant 同样迁移一次

### 阶段 4 · Kernel 优化（4-6 周）

- [ ] 写 **dLLM 专用 Bidirectional Flash Attention**
- [ ] 加**稀疏活跃位置**支持（只算 mask 位置的 output）
- [ ] Triton 版本原型 → CUDA 版本产品化

### 阶段 5 · Serving 集成（长期）

- [ ] 提 PR 到 SGLang / vLLM 支持 dLLM
- [ ] 实现 AR + dLLM **混合路由**
- [ ] 投机解码适配（SimSD 复现）

---

## 九、和 AngelSlim / 你现有工作的接口

如果你在做 AngelSlim 或类似量化工具，dLLM 的接入点：

```
AngelSlim 现有 pipeline (AR 版):
    校准数据 → forward → 采集激活 stats → GPTQ/AWQ → 量化权重

dLLM 适配版:
    校准数据 → 迭代 forward (N 步) → 采集每步激活 stats → mask-aware GPTQ → 量化权重
              └─ 新增 ─┘              └─── 新增 ───┘        └─ 新增 ─┘
```

**主要改动点**：
1. `AutoModelForCausalLM` → `AutoModelForMaskedDiffusion`（HuggingFace 已支持）
2. 前向调用要传 `x_t` 和 `t`，不是 input_ids
3. 校准数据要跑 N 步而不是 1 次
4. 激活统计要按"每步"或"每 mask 率区间"分开

**估算工作量**：给现有 AR 量化 pipeline 加 dLLM 支持，约 **1-2 周**（如果基础扎实）。

---

## 十、一句话总结

> **dLLM 对 Infra 提出的挑战不是"更难"，而是"不一样"**。KV Cache 天然优势没了，但换来了新的**并行度**和**编辑能力**。量化的主战场从 KV Cache 转向权重和激活，kernel 需要从 causal 版重编到双向版，serving 要学会 AR + dLLM 混合路由。**AR 权重可迁移**这条捷径大幅降低了 dLLM 的量化研究门槛 —— 这是 Infra 工程师入场的最佳时机。

---

## 十一、参考

1. Nie et al. *iLLaDA: Improved Large Language Diffusion Models.* 2026. [arXiv 2606.25331](https://arxiv.org/abs/2606.25331)
2. Inclusion AI. *LLaDA 2.0: Scaling Up Diffusion Language Models to 100B.* 2025. [arXiv 2512.15745](https://arxiv.org/abs/2512.15745)
3. Cui et al. *SimSD: Speculative Decoding for Diffusion Language Models.* 2026 preprint. [arXiv 2606.02544](https://arxiv.org/abs/2606.02544)
4. Arriola et al. *Block Diffusion.* ICLR 2025. [openreview.net/forum?id=tyEyYT267x](https://openreview.net/forum?id=tyEyYT267x)
5. Ren et al. *ME-DLM: Edit-Based Refinement for Parallel Masked Diffusion Language Models.* ICML 2026. [arXiv 2605.09603](https://arxiv.org/abs/2605.09603)
6. Gong et al. *DiffuLLaMA: Scaling Diffusion Language Models via Adaptation from Autoregressive Models.* ICLR 2025. [arXiv 2410.17891](https://arxiv.org/abs/2410.17891)
7. Inception Labs. *Mercury.* 2025.02. 商用 dLLM
8. 内网 iwiki: [DLM 深度研究报告](https://iwiki.woa.com/p/4026624940)
9. 内网 iwiki: [dLLM 革命 · 从架构突破到商用化落地](https://iwiki.woa.com/p/4025438157)
10. 内网 iwiki: [iLLaDA 深度解读](https://iwiki.woa.com/p/4023224165)

---

## 下一步

- 复习方法：[LLaDA 方法解析](./llada-method.md)
- 基础背景：[扩散模型基础](./diffusion-basics.md)
- 相关花园：[Softmax · Flash Attention](../transformer/softmax-online-flash.md)（Flash Attention 的双向版就是这里的核心）
- 相关花园：[量化基础](../../quantization/basics/index.md)（如果你在做量化工具链）
