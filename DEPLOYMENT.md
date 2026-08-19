# 部署说明

本仓库使用 **GitHub Pages** 自动部署，源码在 `main` 分支，构建产物发布到 `gh-pages` 分支。

## 自动部署（当前方案）

每次 push 到 `main` 分支，GitHub Actions 会自动触发 `.github/workflows/deploy.yml`：

1. Checkout 代码
2. `npm ci` 装依赖
3. `npm run build` 构建静态站点（产物在 `build/`）
4. 通过 `actions/deploy-pages` 发布到 GitHub Pages

访问地址：<https://zhangbeibei00.github.io/knowledge-garden/>

## 首次启用 GitHub Pages

在 GitHub 仓库首次推送后，需要手动做一次：

1. 进入仓库 → **Settings → Pages**
2. Source 选 **GitHub Actions**（不是 branch）
3. 保存。之后 Actions 触发时会自动发布

## 本地开发 / 手动构建

```bash
npm install        # 装依赖
npm start          # 本地开发（热更新，http://localhost:3000/knowledge-garden/）
npm run build      # 构建静态站点
npm run serve      # 本地预览构建产物
```

## 常见坑位

### baseUrl
GitHub Pages 项目页部署到 `<user>.github.io/<repo>/`，所以 `docusaurus.config.ts` 里：
- `url: 'https://zhangbeibei00.github.io'`
- `baseUrl: '/knowledge-garden/'`

如果之后绑定自定义域名或换到根域名部署，需要把 `baseUrl` 改回 `/`。

### 中文分类 URL 编码
Docusaurus build 时会 warn 中文 slug 被 URL 编码。想要英文 URL 就在 `_category_.json` 里加 `link.slug: 'quantization'` 之类的英文别名。

### MDX 数学公式报错
含数学公式的 md 文件在 frontmatter 加 `format: md`，改用 CommonMark 解析避开 MDX 花括号冲突。

## 备用：Vercel 部署

如果要换 Vercel（会自动识别 Docusaurus，零配置）：

1. 登录 <https://vercel.com/>，Import 本仓库
2. Framework Preset 会自动识别为 Docusaurus
3. **同时** 把 `docusaurus.config.ts` 的 `baseUrl` 改成 `/`、`url` 改成 Vercel 域名
4. 点 Deploy

## 启用站内搜索（可选）

Docusaurus 默认不带搜索。可选：

- **Local Search**（免费、无需申请）：
  ```bash
  npm install --save @easyops-cn/docusaurus-search-local
  ```
  然后在 `docusaurus.config.ts` 的 `themes` 数组加：
  ```ts
  [
    require.resolve('@easyops-cn/docusaurus-search-local'),
    {hashed: true, language: ['en', 'zh']},
  ],
  ```
- **Algolia DocSearch**（免费，需申请）：<https://docsearch.algolia.com/apply>
