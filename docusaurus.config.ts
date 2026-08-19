import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: '知识花园',
  tagline: '量化 × RL × 推理',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  markdown: {
    mermaid: true,
  },

  themes: ['@docusaurus/theme-mermaid'],

  // 部署目标：https://zhangbeibei00.github.io/knowledge-garden/
  url: 'https://zhangbeibei00.github.io',
  baseUrl: '/knowledge-garden/',

  organizationName: 'zhangbeibei00',
  projectName: 'knowledge-garden',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

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
          editUrl: 'https://github.com/zhangbeibei00/knowledge-garden/tree/main/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl: 'https://github.com/zhangbeibei00/knowledge-garden/tree/main/',
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
          href: 'https://github.com/zhangbeibei00/knowledge-garden',
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
            {label: '模型量化', to: '/docs/quantization'},
            {label: 'RL 后训练', to: '/docs/rl-post-training'},
            {label: '推理引擎', to: '/docs/inference-engines'},
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
            {label: 'GitHub', href: 'https://github.com/zhangbeibei00/knowledge-garden'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Knowledge Garden · Built with Docusaurus.`,
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
