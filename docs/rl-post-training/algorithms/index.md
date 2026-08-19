---
sidebar_position: 0
title: 📖 索引
---

# RL 算法知识库：PPO / GRPO / DPO

> 按算法拆分为 4 个独立章节（PPO / GRPO / DPO / 横向对比），方便按需查阅与检索。

---

## 📖 一句话结论

**PPO** 是"稳妥的探索型选手"（4 模型、在线 RL、探索强但成本高），**GRPO** 是"推理任务的性价比之王"（3 模型、组内归一化、省 Critic 显存），**DPO** 是"极简的离线优化器"（2 模型、无 RL 循环、稳定但探索弱）。

三者递进关系：

```
PPO  →  GRPO (去 Critic)  →  DPO (去 RL 循环)
     逐步降低工程复杂度，但也逐步牺牲探索能力
```

---

## 📚 章节导航

| 章节 | 内容 |
|------|------|
| **[01 · PPO](/docs/rl-post-training/algorithms/01-ppo)** | 近端策略优化（原理、四模型、GAE、裁剪目标） |
| **[02 · GRPO](/docs/rl-post-training/algorithms/02-grpo)** | 组相对策略优化（组内归一化、目标函数逐项拆解、显存节省） |
| **[03 · DPO](/docs/rl-post-training/algorithms/03-dpo)** | 直接偏好优化（RLHF→DPO 完整推导、Bradley-Terry、隐式奖励） |
| **[04 · 横向对比](/docs/rl-post-training/algorithms/04-comparison)** | 三者对比 + 演进关系 + 最新变体 + 选型指南 + 公式速查 |

---

## 🎯 快速参考表（速览三算法差异）

| 维度 | PPO | GRPO | DPO |
|------|-----|------|-----|
| **全称** | Proximal Policy Optimization | Group Relative Policy Optimization | Direct Preference Optimization |
| **提出时间** | 2017 (OpenAI) | 2024 (DeepSeek) | 2023 (Stanford) |
| **所需模型数** | 4 (Policy + Ref + Reward + Critic) | 3 (Policy + Ref + Reward) | 2 (Policy + Ref) |
| **训练范式** | 在线 RL（On-policy） | 在线 RL（On-policy） | 离线学习 |
| **是否需要 RL 循环** | ✅ 是 | ✅ 是 | ❌ 否 |
| **是否需要奖励模型** | ✅ 是 | ✅ 是 | ❌ 否（隐式） |
| **是否需要 Critic** | ✅ 是 | ❌ 否 | ❌ 否 |
| **优势估计方式** | GAE（广义优势估计） | 组内归一化奖励 | 不需要（直接优化偏好） |
| **探索能力** | ⭐⭐⭐ 最强 | ⭐⭐ 强 | ⭐ 有限 |
| **训练稳定性** | ⭐⭐ 中（需调参） | ⭐⭐ 中 | ⭐⭐⭐ 最稳定 |
| **训练成本（显存）** | 最高（4 模型同显存） | 中等（省 Critic ~50%） | 最低（仅 2 模型） |
| **适合场景** | 通用对齐、复杂任务 | 推理任务（数学、代码） | 快速偏好对齐 |

---

## 🔀 选型速查

- **大规模推理训练（数学、代码竞赛）** → GRPO / Dr. GRPO / DAPO（已被 DeepSeek-R1 验证）
- **快速偏好对齐（对话风格、安全过滤）** → DPO / SimPO（简单、稳定、低成本）
- **复杂通用任务，需强探索能力** → PPO（探索最强但成本最高）
- **MoE 模型训练** → GSPO（序列级优化适配专家路由）
- **显存受限** → DPO > SimPO > GRPO > PPO

完整决策树见 [04 · 横向对比 § 选型指南](/docs/rl-post-training/algorithms/04-comparison#6-选型指南)。
