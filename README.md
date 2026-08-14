# 🌱 开发虾的知识花园（Knowledge Garden）

> 一个 AI Infra 工程师的第二大脑 · 聚焦 **模型量化 × RL 后训练 × 推理引擎**

[![Docusaurus](https://img.shields.io/badge/Docusaurus-3.x-blue)](https://docusaurus.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 关于

这是一个基于 [Docusaurus](https://docusaurus.io/) 搭建的个人知识库。设计参考了 [AIInfraGuide](https://github.com/caomaolufei/AIInfraGuide) 与 Digital Garden 理念。

**内容成熟度**用三档标注：
- 🌱 **Seedling** — 想法与半成品
- 🌿 **Budding** — 部分成型
- 🌳 **Evergreen** — 反复打磨的成熟笔记

## 🗂️ 内容板块

- 🧭 学习路径
- 📐 模型量化（PTQ / QAT / FP8 / NVFP4）
- 🎯 RL 后训练（PPO / DPO / GRPO）
- ⚙️ 训练框架（Megatron / verl / DeepSpeed）
- 🚀 推理引擎（vLLM / SGLang）
- 🧠 基础知识（Transformer / GPU / 分布式）
- 📖 论文精读
- 🌱 想法花园

## 🚀 本地开发

```bash
# 装依赖
npm install

# 启动开发服务器（热更新）
npm start

# 构建静态站点
npm run build

# 本地预览构建结果
npm run serve
```

## 🌐 部署

### 方式一：Vercel（推荐，全自动）

1. 把仓库 push 到 GitHub
2. 登录 https://vercel.com/ 用 GitHub 账号
3. Import 你的 GitHub 仓库
4. Framework 选 **Docusaurus**，其他默认
5. 点 Deploy，一分钟后拿到 `xxx.vercel.app` 域名

**部署前先改配置**：编辑 `docusaurus.config.ts`，把 `your-username` 全部替换成你的 GitHub 用户名，`url` / `baseUrl` 改成实际部署地址。

### 方式二：GitHub Pages

```bash
# 1) 修改 docusaurus.config.ts 里的 organizationName / projectName / url / baseUrl
# 2) 执行部署
GIT_USER=<你的 GitHub 用户名> npm run deploy
```

详见 [Docusaurus 部署文档](https://docusaurus.io/docs/deployment)。

## ✍️ 写作规范

- 中文优先，术语首次出现给英文原文
- 每篇文章 frontmatter 里加 `tags`、`sidebar_position`
- 数学公式用 `$...$`（行内）或 `$$...$$`（块级），KaTeX 渲染
- 涉及数学公式的 md 文件在 frontmatter 加 `format: md`（避免 MDX 花括号冲突）
- 允许有 🌱 状态的种子文章，标注清楚就行

## 📄 License

MIT
