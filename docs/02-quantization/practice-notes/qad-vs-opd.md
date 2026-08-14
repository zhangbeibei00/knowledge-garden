---
sidebar_position: 2
title: QAD vs OPD - 两种蒸馏的本质区别
description: 对比 NVFP4 QAD 与 On-Policy Distillation，理解 forward/reverse KL 的差异
tags: [quantization, distillation, kl-divergence, on-policy]
format: md
---

:::tip 🌳 Evergreen · 对比笔记
两种蒸馏方法的定位、目标、KL 方向对比
:::

# QAD vs OPD：两种蒸馏的本质区别

> 精简对比 NVFP4 QAD（NVIDIA, 2026）与 On-Policy Distillation（Thinking Machines, 2025），并解释 forward/reverse KL 的核心差异。

---

## 一、两个方法的定位

**一句话总结**：QAD 和 OPD 都叫"蒸馏"、都用 KL 散度，但目标完全不同。

| 维度 | QAD (Quantization-Aware Distillation) | OPD (On-Policy Distillation) |
|-----|----------------------------------------|------------------------------|
| **要解决的问题** | BF16 → NVFP4 量化后精度掉了，想拉回来 | 大模型能力压缩到小模型里 |
| **教师身份** | **同一个模型** 的全精度版本 (BF16) | **更大、更强的另一个模型**（如 32B） |
| **学生身份** | 同一个模型的量化版本 (NVFP4) | 更小的模型（如 8B） |
| **参数量差异** | 完全相同 | 差 3-10 倍常见 |
| **成功标准** | 学生 ≈ 教师（天花板就是教师） | 学生 → 教师（天然有差距） |
| **数据来源** | Off-policy（外部固定数据） | On-policy（学生自己 rollout 生成） |
| **数据敏感度** | 极低（连随机 token 都能用） | 较高（需要合理 prompt） |
| **数据规模** | ~0.3B~6B tokens | 比 RL 少 ~10 倍 |
| **对比的对手** | vs QAT（量化感知训练） | vs RL、vs SFT |
| **主要贡献场景** | 模型压缩部署 | 后训练、能力迁移 |
| **计算开销** | 前向 × 2，无采样 | 前向 × 2 + 学生 rollout 采样 |

**核心区别**：QAD 是"**复制品**"（保真度问题），OPD 是"**继承者**"（技艺传承问题）。

**共同点**：
- 都用 KL 散度作为损失（但方向不同，见第三节）
- 都是软标签监督，比 SFT 硬标签信息量大
- 都比 RL 稳定（KL 是密集信号，每个 token 都有梯度）

---

## 二、数据流对比

**QAD 数据流**（静态、off-policy）：

```
[固定数据集] ──▶ [教师 forward]  ──▶ P_teacher
              └▶ [学生 forward]  ──▶ P_student
                                    └▶ KL(P_teacher || P_student) ──▶ 反向传播
```

**OPD 数据流**（动态、on-policy）：

```
[Prompt] ──▶ [学生 rollout 生成轨迹]
          ──▶ [教师对学生生成的 token 打分] ──▶ P_teacher
          ──▶ [学生自己对这些 token 的概率] ──▶ P_student
                                            └▶ KL(P_student || P_teacher) ──▶ 反向传播
                                                                        └▶ 更新学生 → 下一轮轨迹又变
```

OPD 的 **on-policy** 关键点：**数据分布随学生能力变化而变化**，训练什么、教师纠正什么，都跟着学生走。

---

## 三、损失函数：Forward KL vs Reverse KL

### 数学定义

KL 散度是**非对称**的：

$$
D_{KL}(P \| Q) = \sum_y P(y) \log \frac{P(y)}{Q(y)}
$$

关键看谁在前面（谁作为权重、谁在 log 分子上）：

| 叫法 | 公式 | 使用者 |
|-----|------|-------|
| **Forward KL** | $D_{KL}(P_{teacher} \| P_{student})$ | **QAD** ✅ |
| **Reverse KL** | $D_{KL}(P_{student} \| P_{teacher})$ | **OPD** ✅ |

> 🔍 **判断依据**：看代码里 KL 计算时哪个分布作权重（在 log 前面）、哪个是 `detach()`。PyTorch 的 `F.kl_div(input=log_Q, target=P)` 算的是 $D_{KL}(P \| Q)$，是 forward KL。

### QAD 代码写法（Forward KL）

```python
# QAD: forward KL, mode-covering
with torch.no_grad():
    p_teacher = F.softmax(teacher_logits, dim=-1)          # 权重来自教师

log_p_student = F.log_softmax(student_logits, dim=-1)
log_p_teacher = F.log_softmax(teacher_logits, dim=-1)

# sum over vocab: p_t * (log p_t - log p_s)
loss = (p_teacher * (log_p_teacher - log_p_student)).sum(-1).mean()
```

### OPD 代码写法（Reverse KL）

```python
# OPD: reverse KL, mode-seeking
with torch.no_grad():
    log_p_teacher = F.log_softmax(teacher_logits, dim=-1)  # 教师 detach

log_p_student = F.log_softmax(student_logits, dim=-1)
p_student = log_p_student.exp()                            # 权重来自学生

# sum over vocab: p_s * (log p_s - log p_t)
loss = (p_student * (log_p_student - log_p_teacher)).sum(-1).mean()
```

---

## 四、Mode-Covering vs Mode-Seeking（双峰图解释）

**这是理解 forward/reverse KL 差异的关键。**

假设教师分布是**双峰**——两个正确答案 A 和 B 都可以，各占 50%：

### Forward KL（QAD 用）：Mode-Covering / 全覆盖

**惩罚机制**：只要教师在某个 y 上有概率（$P_t(y) > 0$），学生必须也给它非零概率。否则 $\log(P_t/0) = +\infty$，损失爆炸。

→ 学生**被迫覆盖教师的所有峰**，就算次要的低概率区域也不能漏。

```
教师:  ▁▁▂▅█▅▂▁▁▂▅█▅▂▁▁      (双峰 A 和 B)
              ↑        ↑
              A        B
学生:  ▁▁▂▄▆▄▂▁▁▂▄▆▄▂▁▁      (也是双峰，摊得开，全覆盖)
```

### Reverse KL（OPD 用）：Mode-Seeking / 聚焦一峰

**惩罚机制**：只要学生在某个 y 上有概率，教师必须也给它非零概率。学生完全**可以把概率集中在一个峰上**，另一个峰不管——只要那个峰是教师认可的，就没惩罚。

→ 学生倾向于**"选一个正确答案往死里学"**，塌缩到某个 mode。

```
教师:  ▁▁▂▅█▅▂▁▁▂▅█▅▂▁▁      (双峰 A 和 B)
              ↑        ↑
              A        B
学生:  ▁▁▁▁▁▁▁▁▁▁▁▄█▄▁▁      (只学 B 这个峰，抛弃 A)
                       ↑
                       B
```

### 一道具体题的输出对比

同一道数学题，教师给出的答案分布：

```
教师:
  "42"                ██████████ 50%
  "forty-two"         ████████   40%
  "the answer is 42"  ███        10%
```

**Forward KL (QAD) 训出来的学生** → 保留所有表达方式：

```
  "42"                ████████   45%
  "forty-two"         ██████     35%
  "the answer is 42"  ████       15%
  其他                █           5%
```

**Reverse KL (OPD) 训出来的学生** → 集中攻坚一个答案：

```
  "42"                ████████████████ 90%
  "forty-two"         █                 3%
  "the answer is 42"  ▁                 1%
  其他                ▁                 6%
```

---

## 五、为什么两种选择都是对的？

| 任务性质 | 需求 | 选谁 | 为什么 |
|---------|------|------|-------|
| **QAD**（量化恢复） | 学生和教师**完全同能力**，一寸能力都不能丢 | Forward KL | mode-covering 逼学生覆盖教师所有分布，包括 hedging、跨领域能力（论文表 4：只用 code 数据也能保住 math） |
| **OPD**（能力迁移） | 学生**容量不足**，硬要覆盖大教师所有 mode 会稀释概率 | Reverse KL | mode-seeking 让小学生锁定教师最认可的 mode 做深做透，"做对一件事"比"什么都会一点"重要 |

**Reverse KL 与 RL 的关系**：数学上 reverse KL + on-policy 数据 ≈ 以 $(\log p_s - \log p_t)$ 为 reward 的 policy gradient。这也是 Thinking Machines 说 OPD "结合 RL 的 on-policy + 蒸馏的密集监督" 的原因。

**Forward KL 则更像 SFT 的软化版本**，和 RL 没有直接对应关系。

---

## 六、两个方向的完整属性对比

| 维度 | Forward KL (QAD) | Reverse KL (OPD) |
|-----|-----------------|------------------|
| **公式** | $D_{KL}(P_t \| P_s)$ | $D_{KL}(P_s \| P_t)$ |
| **权重来自** | 教师（固定） | 学生（在变） |
| **优化几何** | Mean-seeking / Mode-covering | Mode-seeking |
| **对多峰分布** | 学生也变多峰，摊开概率 | 学生塌缩到一个峰 |
| **零概率惩罚** | 学生在教师有概率处给 0 → 惩罚爆炸 | 学生在教师给 0 处有概率 → 惩罚爆炸 |
| **和 RL 的关系** | 更像 SFT 的软化版 | 更像 dense-reward 的 policy gradient |
| **副作用** | 无（本来就要求全覆盖） | 可能 mode collapse（塌缩到任意高概率峰，不保证最优） |
| **典型场景** | 精度恢复、量化、模型压缩 | 能力迁移、小模型对齐、on-policy 训练 |
| **代表工作** | QAD (本文), LLM-QAT | OPD (Thinking Machines), DPO 隐式 KL |

---

## 七、一句话记忆

- **Forward KL (QAD)** = "**教师覆盖到哪里，学生就必须到哪里**" → 全盘继承
- **Reverse KL (OPD)** = "**学生走到哪里，教师必须点头**" → 集中攻坚

选哪个，本质上就是问：**希望学生和教师"一样宽"还是"一样深"？**

- 一样宽（保住所有能力）→ Forward KL → QAD
- 一样深（把核心能力吃透）→ Reverse KL → OPD

---

## 参考

- **QAD**: NVIDIA, *Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery*, arXiv:2601.20088v3, 2026-03
- **OPD**: Thinking Machines Lab, *On-Policy Distillation*, 2025-10-27, [thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
