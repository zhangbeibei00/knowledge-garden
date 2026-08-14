import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: '开发虾的知识花园',
  tagline: '一个 AI Infra 工程师的第二大脑 · 量化 × RL × 推理',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  // ⚠️ 部署到 GitHub Pages 或 Vercel 前，请把下面 url / organizationName / projectName 改成你自己的
  url: 'https://your-username.github.io',
  baseUrl: '/knowledge-garden/',

  organizationName: 'your-username',
  projectName: 'knowledge-garden',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['zh-Hans'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: 'docs',
          editUrl: 'https://github.com/your-username/knowledge-garden/tree/main/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl: 'https://github.com/your-username/knowledge-garden/tree/main/',
          onInlineTags: 'warn',
          onInlineAuthors: 'ignore',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: '🌱 知识花园',
      logo: {
        alt: 'Knowledge Garden Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: '📚 知识库',
        },
        {to: '/blog', label: '📝 博客', position: 'left'},
        {
          href: 'https://github.com/your-username/knowledge-garden',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: '主要板块',
          items: [
            {label: '模型量化', to: '/docs/category/模型量化'},
            {label: 'RL 后训练', to: '/docs/category/rl-后训练'},
            {label: '推理引擎', to: '/docs/category/推理引擎'},
          ],
        },
        {
          title: '推荐资源',
          items: [
            {label: 'AIInfraGuide', href: 'https://caomaolufei.github.io/AIInfraGuide/'},
            {label: 'vLLM 官方文档', href: 'https://docs.vllm.ai/'},
            {label: 'Megatron-LM', href: 'https://github.com/NVIDIA/Megatron-LM'},
          ],
        },
        {
          title: '其他',
          items: [
            {label: '博客', to: '/blog'},
            {label: 'GitHub', href: 'https://github.com/your-username/knowledge-garden'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} 开发虾的知识花园 · Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'yaml', 'json'],
    },
    // 站内搜索：先用本地搜索，后续可换 Algolia
  } satisfies Preset.ThemeConfig,
};

export default config;

// KaTeX CSS
// @ts-ignore
config.stylesheets = [
  {
    href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css',
    type: 'text/css',
    integrity: 'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV',
    crossorigin: 'anonymous',
  },
];
