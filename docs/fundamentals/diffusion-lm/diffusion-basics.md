---
sidebar_position: 1
title: 扩散模型基础 · 从 DDPM 到离散扩散
description: 图像扩散的数学起点（DDPM/iDDPM/DDIM）、Score-based 视角、离散扩散怎么迁移到 token（D3PM/MDLM）、以及这条路线通向 LLaDA 的关键跳跃
tags: [diffusion, ddpm, iddpm, ddim, d3pm, mdlm, generative-model, foundation]
---

# 扩散模型基础 · 从 DDPM 到离散扩散

在讲 LLaDA 之前，先把**扩散模型这一整条家族树**理清楚。因为 LLaDA 不是凭空出现的 —— 它是扩散模型从**图像连续域**跨越到**语言离散域**的自然产物。

这篇按下面这条脉络展开：

```
1. 扩散模型的核心思想 —— 加噪与去噪
2. DDPM: 现代扩散模型的正式起点
3. iDDPM: 三大关键改进
4. DDIM: 非马尔可夫加速采样
5. Score-based / SDE 视角: 统一框架
6. 跨界到离散: D3PM 与 MDLM
7. 通向 LLaDA 的关键跳跃
```

---

## 一、核心思想：加噪 → 去噪

扩散模型的直觉极其简单：

```
训练:  真实图片 → 逐步加噪 → 高斯白噪声   (前向过程,固定)
                                  ↓
推理:  高斯白噪声 → 逐步去噪 → 真实图片   (反向过程,模型学习)
```

**训练学的是"去噪"这一步**，也就是给一张被加了 t 步噪声的图，让模型预测**这些噪声长什么样**。

关键洞察：
- **前向加噪是设计好的**（数学上给定的高斯扰动过程）
- **反向去噪是学习出来的**（Transformer 或 U-Net 拟合）
- **训练和推理是"反向对称"**（都跨越 T 步）

---

## 二、DDPM (2020) · 现代扩散模型的正式起点

**论文**: Denoising Diffusion Probabilistic Models, Ho et al., 2020, [arXiv 2006.11239](https://arxiv.org/abs/2006.11239)

### 2.1 前向过程

给定原始数据 $x_0$，定义 $T$ 步（比如 $T=1000$）的加噪链，每一步：

$$
x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \varepsilon,\quad \varepsilon \sim \mathcal{N}(0, I)
$$

其中 $\beta_t$ 是一个预先设计好的"噪声调度"（noise schedule），随 $t$ 增大而增大。

**巧妙的一步**：可以合并 $t$ 步得到闭式：

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon,\quad \bar\alpha_t = \prod_{i=1}^{t}(1 - \beta_i)
$$

**这就是说**：任意一步 $t$ 的加噪状态可以**一步到位从 $x_0$ 采样得到**，不需要真的迭代 $t$ 次。训练时省时省力。

### 2.2 反向过程（学习目标）

模型 $\varepsilon_\theta(x_t, t)$ 输入含噪图和时间步，**预测这一步的噪声 $\varepsilon$**：

$$
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{x_0,\, t,\, \varepsilon}\left[\, \lVert \varepsilon - \varepsilon_\theta(x_t, t) \rVert^2 \,\right]
$$

**训练目标就一行**：预测噪声的 MSE。极简。

**为什么预测噪声而不是 $x_0$**？数学上等价，但预测噪声在 loss 数值稳定性上更好（推导见 DDPM 附录）。

### 2.3 推理（采样）

从纯噪声 $x_T \sim \mathcal{N}(0, I)$ 起步，迭代 $T$ 步：

```python
for t in [T, T-1, ..., 1]:
    eps_pred = model(x_t, t)
    x_{t-1} = f(x_t, eps_pred, t) + noise    # 每步都要注入一点新噪声
return x_0
```

**问题**：$T=1000$ 步太慢，生成一张图要跑 1000 次模型前向。这就是后来 DDIM、Consistency Models 想解决的。

### 2.4 关键概念表

| 概念 | 含义 |
|------|------|
| $\beta_t$ (noise schedule) | 第 $t$ 步加多少噪声，通常线性从 $10^{-4}$ 到 $0.02$ |
| $\bar\alpha_t$ | 从 $0$ 到 $t$ 的累积保留系数 |
| $\varepsilon$ (epsilon) | 加入的高斯噪声，模型要预测的目标 |
| $T$ (总步数) | DDPM 原论文用 $1000$ |
| Timestep Embedding | 把 $t$ 编码为向量，通过 sinusoidal 位置编码，输入 U-Net |

---

## 三、iDDPM (2021) · 三个关键改进

**论文**: Improved Denoising Diffusion Probabilistic Models, Nichol & Dhariwal, 2021, [arXiv 2102.09672](https://arxiv.org/abs/2102.09672)

DDPM 效果好但有几个明显缺点，iDDPM 针对性地做了三个改进：

### 3.1 改进一：Cosine Noise Schedule

DDPM 的**线性 $\beta$ 调度**在小分辨率图（32×32、64×64）上有问题 —— 加噪太快，前几步就把信息毁掉了。

iDDPM 用 **cosine 调度**：

- DDPM 线性：$\beta_t$ 从 $10^{-4}$ 线性到 $0.02$
- iDDPM cosine：

$$
\bar\alpha_t = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right),\quad s \approx 0.008
$$

**效果**：
- 前期加噪更慢，中期加噪更均匀
- 小图上的 FID 显著改善
- 这一 trick **被后来几乎所有扩散模型继承**（Stable Diffusion、DiT 都用变种）

### 3.2 改进二：学习方差 Σ_θ

DDPM 里反向过程的方差是**固定**的（$\beta_t$ 或 $\tilde\beta_t$）。iDDPM 让模型**学习**一个介于两者之间的插值系数：

$$
\Sigma_\theta(x_t, t) = \exp\!\Big(\, v_\theta(x_t, t) \cdot \log \beta_t + \big(1 - v_\theta(x_t, t)\big) \cdot \log \tilde\beta_t \,\Big)
$$

其中 $v_\theta(x_t, t) \in [0, 1]$ 是模型输出的插值系数。

**这带来一个副作用**：模型现在**同时输出**均值和方差，训练目标变成：

$$
\mathcal{L}_{\text{iDDPM}} = \underbrace{\mathcal{L}_{\text{simple}}(\varepsilon)}_{\text{DDPM 原始 MSE}} + \lambda \cdot \underbrace{\mathcal{L}_{\text{VLB}}(\mu, \Sigma)}_{\text{变分下界}\text{(约束方差学习)}}
$$

**好处**：可以在**更少的采样步数**下（比如 50 步而不是 1000 步）保持高质量。

### 3.3 改进三：架构和训练细节

- U-Net 里加入 **attention layer**（在特定分辨率）
- **Importance sampling**：采样 t 时给不同时间步不同权重
- 训练更长时间，训练 loss 曲线更平滑

### 3.4 iDDPM 的历史意义

| 意义 | 说明 |
|------|------|
| **Cosine schedule 成为标配** | 后续几乎所有扩散模型都用它或它的变种 |
| **可学习方差成为标配** | Guided Diffusion、Stable Diffusion 都用 |
| **为 DDIM 铺路** | 步数可以显著减少 |
| **证明 diffusion beats GANs 的可行性** | 同期的 Guided Diffusion 就是基于 iDDPM |

---

## 四、DDIM (2020) · 非马尔可夫加速采样

**论文**: Denoising Diffusion Implicit Models, Song et al., 2020, [arXiv 2010.02502](https://arxiv.org/abs/2010.02502)

### 4.1 核心洞察

DDPM 的采样是**马尔可夫过程**（$x_t$ 只依赖 $x_{t+1}$），每步都要加新噪声，所以必须走完 $T$ 步。

DDIM 说：如果我不严格遵守马尔可夫链，而是**直接从 $x_t$ 跳到 $x_{t-k}$**（跳 $k$ 步），可以吗？

**结论**：**可以**。而且效果几乎不损失。

### 4.2 数学关键

DDIM 的采样公式：

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}} \cdot \underbrace{\hat{x}_0(x_t)}_{\text{预测的 } x_0} + \underbrace{\sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2} \cdot \varepsilon_\theta(x_t, t)}_{\text{沿"噪声方向"移动}} + \underbrace{\sigma_t \cdot z}_{\text{新采噪声（可为0）}}
$$

**关键**：当 $\sigma_t = 0$ 时，采样是完全确定性的（相同 $x_T$ 总生成相同 $x_0$）。

### 4.3 加速效果

| 采样方案 | 步数 | FID (ImageNet 64) |
|---------|------|------------------|
| DDPM | 1000 | ~3.5 |
| DDIM (η=0) | 50 | ~3.5 (几乎一样) |
| DDIM (η=0) | 20 | ~4.0 (略差) |
| DDIM (η=0) | 10 | ~5.0 |

**10-50 步就能达到 DDPM 1000 步的效果**，这是 20-100× 的加速。

### 4.4 DDIM 的思想遗产

- **确定性采样**：给同一个 x_T，总生成同一个 x_0 → 可以做 image editing、interpolation
- **分离"预测 x_0"和"移动到下一步"** → 后来 DPM-Solver、UniPC 都基于这个思想

---

## 五、Score-based / SDE 视角 · 统一框架

**论文**: Score-Based Generative Modeling through Stochastic Differential Equations, Song et al., 2020, [arXiv 2011.13456](https://arxiv.org/abs/2011.13456)

### 5.1 一个视角切换

Yang Song 证明了：**DDPM、NCSN、Score-based 是同一个东西的不同离散化**。

用**连续时间 SDE**（随机微分方程）描述前向加噪：

$$
dx = f(x, t)\, dt + g(t)\, dW
$$

其中 $W$ 是 Wiener 过程（布朗运动）。

反向过程也是一个 SDE：

$$
dx = \big[\, f(x, t) - g(t)^2 \cdot \underbrace{\nabla_x \log p_t(x)}_{\text{score function}} \,\big]\, dt + g(t)\, d\bar{W}
$$

**核心洞察**：模型学的其实是 **score function** —— 数据分布对数密度的梯度 $\nabla_x \log p_t(x)$。

预测噪声 $\varepsilon$ 等价于学 score，两者数学等价（在高斯噪声假设下）：

$$
\varepsilon_\theta(x_t, t) \propto -\nabla_x \log p_t(x_t)
$$

差一个 $-\sqrt{1 - \bar\alpha_t}$ 的系数。

### 5.2 意义

- 提供了**统一理论框架**
- 打通了扩散模型和 score matching（更早的一条研究线）
- **ODE 视角**：如果去掉噪声项，反向过程变成 ODE，可以用高阶数值求解器（DPM-Solver 等）加速

---

## 六、跨界到离散：D3PM 与 MDLM

到这里，扩散模型都在**连续域**（图像像素、embedding 空间）。要用到**语言 token**（离散空间）就要重新设计。

### 6.1 D3PM (2021) · 离散扩散奠基

**论文**: Structured Denoising Diffusion Models in Discrete State-Spaces, Austin et al., NeurIPS 2021, [arXiv 2107.03006](https://arxiv.org/abs/2107.03006)

D3PM 用**转移矩阵 $Q_t$** 定义离散加噪：

$$
q(x_t \mid x_{t-1}) = \text{Cat}(x_t;\; Q_t \cdot x_{t-1})
$$

其中 $\text{Cat}$ 是类别分布。$Q_t$ 有几种设计：

| 类型 | 转移矩阵 |
|------|---------|
| **Uniform** | 每个 token 有 $\beta_t$ 概率变成随机 token |
| **Gaussian on embedding** | 转移概率正比于 embedding 距离 |
| **Absorbing (吸收态)** | 每个 token 有 $\beta_t$ 概率变成特殊 `[MASK]` |

**吸收态转移**是最重要的一种 —— 它把 D3PM 和 **BERT 的 MLM 训练**联系起来了：

```
Absorbing D3PM:
    x_0 (原句)
    → 部分 token 被替换为 [MASK]
    → 更多 token 被替换为 [MASK]
    → 全部变成 [MASK]
    ↑
    这正是 MLM 训练的过程,只是把 mask 率做成了时间轴
```

### 6.2 MDLM (2024) · 化简为 principled MLM

**论文**: Simple and Effective Masked Diffusion Language Models, Sahoo et al., NeurIPS 2024, [arXiv 2406.07524](https://arxiv.org/abs/2406.07524)

MDLM 聚焦**吸收态 mask 扩散**，通过 Rao-Blackwellization 把复杂的扩散 loss 简化：

$$
q(x_t \mid x_0) = \text{Cat}\!\big(\, \alpha_t \cdot x_0 + (1 - \alpha_t) \cdot m \,\big)
$$

其中 $\alpha_t \cdot x_0$ 是原 token 分布，$(1 - \alpha_t) \cdot m$ 是 `[MASK]` 分布。训练损失：

$$
\mathcal{L}_{\text{MDLM}} = \mathbb{E}_{t,\, x_t}\!\left[\, \underbrace{\frac{\alpha_t'}{1 - \alpha_t}}_{\text{加权系数（从 }t\text{ 导出）}} \cdot \sum_i \underbrace{\mathbb{1}[x_t^i = \text{MASK}]}_{\text{只在 mask 位置}} \cdot \big(\!-\log p_\theta(x_0^i \mid x_t)\big) \,\right]
$$

**关键结论**：
- 扩散语言模型的训练 = **有原则的加权 masked cross-entropy**
- 每一步都是一次 MLM 前向 —— **训练成本和 BERT-MLM 同一量级**
- 保留了变分下界解释 —— **理论上是 principled 的**

**这就直接通向 LLaDA** —— LLaDA 的训练目标本质上就是 MDLM 目标。

---

## 七、通向 LLaDA 的关键跳跃

从 DDPM 到 LLaDA，一共跨越了几个关键节点：

```
DDPM (2020)         连续域扩散,预测噪声
    │
    │ Q1: 能不能加速?
    ▼
DDIM/iDDPM (2020-21)  非马尔可夫 + cosine + 学方差 + 少步采样
    │
    │ Q2: 能不能统一理论?
    ▼
Score SDE (2020)    连续时间 SDE,统一 score matching + DDPM
    │
    │ Q3: 能不能做条件生成?
    ▼
Classifier / CFG    引导采样,文生图爆发
    │
    │ Q4: 骨干能不能不用 U-Net?
    ▼
DiT (2022)          Transformer 骨干,与语言模型架构对齐 ⭐
    │
    │ Q5: 能不能做 token 而不是像素?
    ▼
D3PM (2021)         离散转移矩阵,吸收态 = BERT-MLM
    │
    │ Q6: loss 能不能简化?
    ▼
MDLM (2024)         principled MLM,理论 + 工程双收敛
    │
    │ Q7: 能不能 scale 到大模型?
    ▼
⭐ LLaDA (2025)      8B 从零训练,证明 dLLM 追平 LLaMA-3
    │
    ▼
LLaDA 2.0-2.2       蚂蚁,100B MoE,Token Editing,商用化
```

---

## 八、和知识花园其他板块的关联

- **[Transformer · Pre-Norm Block](../transformer/pre-norm-block.md)** — LLaDA 的去噪器就是 Transformer,只是把 causal mask 去掉
- **[RoPE · 一篇讲透](../transformer/rope.md)** — LLaDA 也用 RoPE,双向 attention 下的 RoPE 仍然有效
- **[Softmax · Flash Attention](../transformer/softmax-online-flash.md)** — Flash Attention 也能用在 dLLM,但 causal 版要改成 non-causal 版
- **[LLaDA 方法解析](./llada-method.md)** — 下一篇,详细讲 LLaDA 的训练和推理
- **[dLLM Infra 改进方向](./dllm-infra.md)** — 再下一篇,从 AI Infra 视角看 dLLM 的挑战和机会

---

## 九、参考

1. Ho, Jain & Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. [arXiv 2006.11239](https://arxiv.org/abs/2006.11239)
2. Nichol & Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021. [arXiv 2102.09672](https://arxiv.org/abs/2102.09672)
3. Song, Meng & Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021. [arXiv 2010.02502](https://arxiv.org/abs/2010.02502)
4. Song et al. *Score-Based Generative Modeling through SDEs.* ICLR 2021. [arXiv 2011.13456](https://arxiv.org/abs/2011.13456)
5. Austin et al. *Structured Denoising Diffusion Models in Discrete State-Spaces.* NeurIPS 2021. [arXiv 2107.03006](https://arxiv.org/abs/2107.03006)
6. Sahoo et al. *Simple and Effective Masked Diffusion Language Models.* NeurIPS 2024. [arXiv 2406.07524](https://arxiv.org/abs/2406.07524)
7. Peebles & Xie. *Scalable Diffusion Models with Transformers (DiT).* ICCV 2023. [arXiv 2212.09748](https://arxiv.org/abs/2212.09748)
8. Lilian Weng. *What are Diffusion Models?* [lilianweng.github.io](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
