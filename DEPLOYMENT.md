# 部署到 GitHub + Vercel

## 一、推送到 GitHub

### 1. 创建 GitHub 仓库
在 GitHub 上新建一个仓库（例如 `knowledge-garden`），**不要**勾选 Add README / .gitignore / license（我们本地已经有了）。

### 2. 修改配置里的用户名
把 `docusaurus.config.ts` 里所有 `your-username` 替换成你的 GitHub 用户名：

```bash
cd knowledge-garden
sed -i 's/your-username/你的GitHub用户名/g' docusaurus.config.ts
```

如果要用 GitHub Pages，还需要改：
- `url`：改成 `https://你的用户名.github.io`
- `baseUrl`：仓库名前后加斜杠，如 `/knowledge-garden/`

如果只用 Vercel（不走 GitHub Pages）：
- `url`：改成你 Vercel 部署的地址，如 `https://knowledge-garden.vercel.app`
- `baseUrl`：改成 `/`

### 3. 提交并推送
```bash
cd knowledge-garden
git init -b main
git add .
git commit -m "chore: initial commit of knowledge garden"
git remote add origin git@github.com:你的用户名/knowledge-garden.git
git push -u origin main
```

## 二、部署到 Vercel（最简单）

1. 打开 https://vercel.com/ ，用 GitHub 账号登录
2. 点 **Add New → Project**
3. Import 你刚推的仓库
4. Framework Preset 会自动识别为 **Docusaurus**
5. 点 **Deploy**
6. 一分钟后拿到 `https://knowledge-garden-xxx.vercel.app`

**后续每次 push 到 main 分支，Vercel 会自动重新部署**。

## 三、（可选）绑定自定义域名

1. 在 Vercel 项目 → Settings → Domains
2. 添加你的域名（比如 `garden.你的域名.com`）
3. 按提示在域名服务商处加 CNAME 记录，指向 `cname.vercel-dns.com`

## 四、部署到 GitHub Pages（备用方案）

```bash
# 确保 docusaurus.config.ts 里 organizationName / projectName / url / baseUrl 都正确
GIT_USER=你的GitHub用户名 npm run deploy
```

访问 `https://你的用户名.github.io/knowledge-garden/`

## 五、常见坑位

### baseUrl 配置错误
如果部署后页面样式/图片全丢，通常是 `baseUrl` 写错了：
- Vercel（根域名部署）：`baseUrl: '/'`
- GitHub Pages（项目子路径）：`baseUrl: '/仓库名/'`

### 中文分类 URL 编码
Docusaurus build 时会 warn 中文 slug 被 URL 编码。功能正常但看起来长。想要英文 URL 就在 `_category_.json` 里加 `link.slug: 'quantization'` 之类的英文别名。

### MDX 数学公式报错
含数学公式的 md 文件在 frontmatter 里加：
```yaml
format: md
```
这样 Docusaurus 会用 CommonMark 而不是 MDX 解析，避免花括号冲突。

## 六、启用站内搜索（可选）

Docusaurus 默认不带搜索。有两个方案：

### 免费方案：Local Search
```bash
npm install --save @easyops-cn/docusaurus-search-local
```

在 `docusaurus.config.ts` 的 `themes` 数组加：
```ts
themes: [
  [
    require.resolve('@easyops-cn/docusaurus-search-local'),
    {hashed: true, language: ['en', 'zh']},
  ],
],
```

### 强大方案：Algolia DocSearch（免费，需申请）
访问 https://docsearch.algolia.com/apply 申请，通过后按官方文档配置。
