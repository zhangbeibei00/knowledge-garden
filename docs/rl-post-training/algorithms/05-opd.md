---
sidebar_position: 5
slug: 05-opd
title: OPD - On-Policy Distillation 精读
description: Thinking Machines 2025.10 提出的 on-policy distillation 后训练范式，兼顾 on-policy 相关性与 dense 奖励信号
tags: [distillation, post-training, on-policy, reverse-kl, rlhf]
format: md
---

:::tip 🌳 Evergreen · 后训练笔记
Thinking Machines Lab, 2025.10 · [原文链接](https://thinkingmachines.ai/blog/on-policy-distillation/)
:::

# OPD (On-Policy Distillation) 精读

> 让"小模型"继承"大模型"能力的一种新范式：on-policy 采样 + dense reward，比 RL 便宜 10 倍，比 SFT 抗错误累积。

---

## 一句话结论

**OPD = 让学生自己 rollout（on-policy）+ 教师逐 token 打分（reverse KL）**：既避免 SFT 分布偏移带来的错误累积，又比 RL 的稀疏 reward 效率高得多，是**小模型后训练的性价比之王**。

---

## 一、为什么会有 OPD？先看后训练的两条老路

现代 LLM 训练三阶段：

- **Pre-training**：通用能力（语言、常识、推理雏形）
- **Mid-training**：注入领域知识（代码、医学、企业文档）
- **Post-training**：塑造定向输出行为（指令跟随、数学推理、多轮对话）

Post-training 是"能不能用"的分水岭，成本最高、最难收敛。主流两条路各有硬伤：

### Path A：SFT / 离线蒸馏（off-policy + dense reward）

用**教师模型**（更大、更强）生成的轨迹，学生模仿。

- ✅ 每个 token 都有 label（**dense**），学得快
- ❌ 学生学的是"教师习惯待的状态"，一旦自己走岔了，就再也回不来（**exposure bias / compounding error**）
- ❌ 只学到教师的**风格与自信**，不一定学到**事实准确性**（[The False Promise of Imitating Proprietary LLMs](https://arxiv.org/abs/2305.15717)）

**下棋类比**：像看象棋大师下棋 —— 你看到的都是高手棋局，但新手根本走不到那种局面里。

### Path B：RL（on-policy + sparse reward）

学生自己 rollout，靠 reward 模型（或 rule-based verifier）打分更新。

- ✅ 学的是**学生自己会走到的状态**，训练分布 = 部署分布
- ❌ 一整个 rollout 只给一个 reward，**梯度稀疏**
- ❌ 学生学到"这次错了"但不知道错在**哪一步**（是运算错了还是策略错了）

Thinking Machines 的一个数字：**每个 episode 只传递固定 bits，与 token 数无关** → 长序列任务效率极差。

**下棋类比**：像自己下棋没教练，赢了输了才知道，但不知道**哪一步是关键**。

### Path C（OPD）：把两者的优点抢过来

| 方法 | Sampling | Reward |
|------|----------|--------|
| SFT | off-policy | **dense** ✅ |
| RL | **on-policy** ✅ | sparse |
| **OPD** | **on-policy** ✅ | **dense** ✅ |

**下棋类比**：教练站在你身边，看你每一步都点评"这步好、这步臭、这步大昏招"—— 就是 chess.com 的引擎分析。

---

## 二、算法核心：Reverse KL 的每 token 版本

OPD 的 loss 就一条公式，非常干净：

$$
\text{KL}\big(\pi_\theta \,\|\, \pi_\text{teacher}\big) = \mathbb{E}_{x \sim \pi_\theta} \Big[ \log \pi_\theta(x_{t+1} | x_{1..t}) - \log \pi_\text{teacher}(x_{t+1} | x_{1..t}) \Big]
$$

拆开来看：

| 符号 | 含义 |
|------|------|
| $\pi_\theta$ | **学生**当前策略（正在训练的小模型） |
| $\pi_\text{teacher}$ | **教师**策略（更大、更强的模型，冻结） |
| $x \sim \pi_\theta$ | 轨迹**由学生自己采样**（这就是 on-policy 的含义） |
| $x_{1..t}$ | 到当前位置为止的前文 |
| $x_{t+1}$ | 下一个 token |

**逐 token** 的 reverse KL 意味着：**学生在每个位置的下个 token，都要跟教师的分布对齐**。

### 为什么用 reverse KL 而不是 forward KL？

参考你已有的 [QAD vs OPD 笔记](/docs/quantization/practice-notes/qad-vs-opd) 里的完整推导，这里只列 OPD 视角的三个理由：

1. **Mode-seeking（聚焦一峰）**：学生容量比教师小，硬要覆盖教师所有 mode（forward KL 干的事）会稀释概率，导致啥都会一点、啥都不精。reverse KL 让学生**集中概率在教师最认可的那个 mode**，做深做透
2. **Unhackable**：低 KL ⇔ 教师视角下"这个行为高概率" —— **没法钻空子**（RL 的 reward model 常见"钻分"问题在这不存在）
3. **降低 exposure bias**：因为轨迹来自学生自己，训练分布 = 部署分布

### 一个直观例子（来自原文）

题目：*"一杯冰块放进热平底锅，最后剩几个？"*（SimpleBench 的物理常识题，正确答案 B. 0，因为冰块会融化）

学生 Qwen3-4B 把它当纯数学题算错了。教师 Qwen3-235B 逐 token 打分：

- 那些**引入错误推理方向的开头 token**（"forking tokens"）→ 高 reverse KL → 强惩罚
- 后面顺着错误方向走的 token → 因为条件概率高，KL 反而不高，不冤枉学生

📌 **关键洞察**：OPD 的 dense reward 天然能定位"错在哪一步"，这是 RL 的 sparse reward 永远做不到的。

---

## 三、Discount Factor 用 0：只优化下一个 token

原文实验：discount factor $\gamma > 0$ 更"数学正确"，但**实践中并没变好**。

所以 OPD 直接选 $\gamma = 0$：

$$
\text{loss}_t = -\log \pi_\theta(x_{t+1}) + \log \pi_\text{teacher}(x_{t+1})
$$

**每个位置只优化下个 token，不考虑后续影响** —— 简单，还工作得很好。

这个决策带来的巨大好处：**不需要 rollout 走完就能算 loss** → 可以用**短 rollout / partial rollout** 训练 → 大幅省 compute。

---

## 四、伪代码（对应 Tinker 实现）

原文实现在 [tinker-cookbook/recipes/distillation](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation)。核心逻辑：

```python
# 1. 初始化 teacher（只需要采样，不需要反向）
teacher_client = SamplingClient(teacher_model)

# 2. 学生 rollout（跟 RL 一模一样）
trajectories = student.sample(prompts)              # x ~ π_θ
student_logprobs = student.compute_logprobs(trajectories)  # log π_θ(x)

# 3. 查教师的 logprob（教师前向一次即可）
teacher_logprobs = teacher_client.compute_logprobs(trajectories)  # log π_teacher(x)

# 4. 逐 token reverse KL
per_token_kl = student_logprobs - teacher_logprobs

# 5. 塞进 RL 的 importance-sampling loss（advantage = -KL）
advantages = -per_token_kl
loss = rl_importance_sampling_loss(trajectories, advantages)  # 复用 RL 训练脚本

loss.backward()
```

⚡ 亮点：**在带 KL 正则化的 RL 脚本上，OPD 只需一行改动 —— 把 KL 正则的 regularizer model 换成 teacher 即可**。

工程角度这个"少即是多"的实现，是它能快速被业界采用的关键原因。

---

## 五、实验结果（三条主线）

### 实验 1：数学推理（Qwen3-4B 学 Qwen3-235B）

| 方法 | AIME 24 | 训练成本 |
|------|---------|----------|
| RL from scratch | 44.7 | 1× 基准 |
| SFT (off-policy) | 39.1 | 更便宜但差 5 分 |
| **OPD** | **45.3** | **~1/10 RL 成本** |

📌 **接近 RL 效果，训练成本降到 ~10%**。这就是 "on-policy distillation 让训练成本直降 10 倍" 的来源。

### 实验 2：持续学习助理（避免灾难性遗忘）

给已有 chat 模型注入新领域知识（例如企业内部知识库）：

- **纯 SFT 新领域**：新知识学到了，但**指令跟随能力显著下降**（灾难性遗忘）
- **OPD（教师用原始 chat 模型）**：新知识学到 + 老能力保住 ✅

### 实验 3：数据效率

OPD 用的 tokens 数**比 RL 少一个数量级**（[On-policy distillation 相关论文](https://arxiv.org/abs/2306.13649) 的原始结论也印证了这一点）。

---

## 六、变体与扩展

### 6.1 GOLD（Generalized OPD）— 跨 tokenizer 支持

**问题**：原始 OPD 要求学生和教师**共享同一个 tokenizer**（因为要对齐每个 token 的 logprob）。这在实际中经常做不到 —— 比如学生是 Qwen3、教师是 GPT-4，tokenizer 完全不同。

**GOLD 解法**：
- 用 **sub-token / character 级别对齐** 替代精确 token 对齐
- 或者用 **序列级 IS 权重**近似
- 让 OPD 能跨模型家族用

（参考 [HuggingFace Blog: Unlocking On-Policy Distillation for Any Model Family](https://huggingface.co/blog/gold)）

### 6.2 与 RL 的正则化融合

原文提到一个自然扩展方向：

- **RL loss + OPD KL 作为正则项**：让学生既优化外部 reward，又保持接近教师 → 相当于 KL-regularized RL，只是 regularizer 从 reference model 换成了 teacher

hytuner 里的 `compute_policy_loss_icepop` 就长这个样子（advantage = teacher_logprob - student_logprob，然后套 PPO clip + IcePop mask）—— 参考 [IcePop 讨论](https://arxiv.org/abs/2510.18855)。

---

## 七、什么时候用 OPD？决策速查

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| 有强 teacher + 小学生 + 数据少 | ✅ OPD | 数据效率高，成本 1/10 RL |
| 想让小模型学**特定领域**（数学、代码、工具使用） | ✅ OPD | dense reward + on-policy |
| 无 teacher，但有可验证 reward（数学、代码） | ✅ RL (GRPO / PPO) | OPD 需要 teacher |
| 需要教师能力"以外"的能力（比如强 reasoning） | ⚠️ RL 更适合 | OPD 天花板 = teacher |
| 快速偏好对齐（对话风格） | ✅ DPO / SimPO | 更简单更稳 |
| 量化恢复精度 | ✅ QAD / QAT | forward KL + off-policy 才对 |

📌 **一条清晰的分界**：
- 追求"学到教师会的" → **OPD**
- 追求"学到教师也不会的" → **RL**
- 追求"BF16 版本学 FP4 版本" → **QAT / QAD**

---

## 八、与 QAT / QAD 的本质区别（一句话总结版）

参考完整对比 [QAD vs OPD](/docs/quantization/practice-notes/qad-vs-opd)：

| 维度 | QAD | OPD |
|------|-----|-----|
| **教师** | 同一模型的 BF16 版本 | 更大、更强的独立模型 |
| **目标** | 量化恢复（保真度） | 能力迁移（继承） |
| **数据** | Off-policy，随便什么都行 | On-policy，学生自己 rollout |
| **KL 方向** | Forward KL（mode-covering） | Reverse KL（mode-seeking） |
| **成功标准** | 学生 ≈ 教师 | 学生 → 教师（有 gap） |

**核心区别**：QAD 是"**复制品**"，OPD 是"**继承者**"。

---

## 九、局限性与注意事项

### 硬约束
1. **必须有强 teacher**：没有比学生强的模型，OPD 无从谈起
2. **Teacher 的能力就是学生的天花板**：想突破 teacher 得走 RL
3. **默认要求 tokenizer 一致**（否则得走 GOLD 等变体）
4. **Teacher 推理成本**：虽然只需前向一次，但如果 teacher 是 235B 级别，成本也很可观

### 常见坑
- **Reward hacking 变种**：学生可能学到教师的"表面 pattern"而不是深层能力 —— 需要看 downstream benchmark 而非只看 KL 曲线
- **训练数据 prompts 的选择很关键**：跟 RL 一样，如果 prompt 分布跟部署时差太远，效果会打折
- **教师 log_prob 数值精度**：如果 teacher 用 vLLM 采样，rollout 和 training 引擎的数值差异会污染 KL 估计 —— 这就是 [IcePop](https://arxiv.org/abs/2510.18855) 要解决的问题

---

## 十、参考资料

**一手材料**
- [On-Policy Distillation - Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/) · 2025.10.27（原文）
- [Tinker Cookbook 实现](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation)

**理论基础**
- [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) · Agarwal et al, 2023 · 早期 OPD 工作
- [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543) · Gu et al, 2023 · reverse KL 分析
- [DAGGER: A Reduction of Imitation Learning](https://arxiv.org/abs/1011.0686) · Ross et al, 2010 · 思想源头
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) · 2025 · 大规模实践

**扩展变体**
- [GOLD: Unlocking On-Policy Distillation for Any Model Family](https://huggingface.co/blog/gold) · 跨 tokenizer 版本
- [IcePop: An Effective Method for MoE Stability](https://arxiv.org/abs/2510.18855) · 2025.10 · 修正 rollout-training mismatch，可与 OPD 结合

**相关问题**
- [The False Promise of Imitating Proprietary LLMs](https://arxiv.org/abs/2305.15717) · Gudibande et al, 2023 · SFT 蒸馏的表面性问题
- [Beyond the 80/20 Rule: Forking Tokens](https://arxiv.org/abs/2506.01939) · 2025 · 关键 token 定位

---

## 附：一分钟速览（回头速查用）

- **是什么**：让学生自己 rollout，教师逐 token 打分（reverse KL）
- **成本**：≈ RL 的 1/10
- **loss**：`per_token_kl = student_logprob - teacher_logprob`（学生 detach 前算完再 detach 教师）
- **实现**：RL 脚本 + 一行改动（KL regularizer 换成 teacher）
- **γ = 0**：只优化下个 token，可以用 partial rollout
- **不能用**：没 teacher / 想突破 teacher / tokenizer 不同（除非用 GOLD）
