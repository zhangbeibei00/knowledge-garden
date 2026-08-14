import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  emoji: string;
  title: string;
  description: ReactNode;
  link: string;
  linkText: string;
};

const FeatureList: FeatureItem[] = [
  {
    emoji: '📐',
    title: '模型量化',
    description: (
      <>
        从 PTQ 到 QAT，从 FP8 到 NVFP4。系统梳理主流量化方法与低精度硬件格式的原理和工程实践。
      </>
    ),
    link: '/docs/category/模型量化',
    linkText: '进入量化专区 →',
  },
  {
    emoji: '🎯',
    title: 'RL 后训练',
    description: (
      <>
        PPO、DPO、GRPO 等主流 RLHF 算法笔记。以 AI Infra 工程师的视角理解 RL 训练 pipeline 的每一步。
      </>
    ),
    link: '/docs/category/rl-后训练',
    linkText: '进入 RL 专区 →',
  },
  {
    emoji: '🚀',
    title: '推理引擎',
    description: (
      <>
        vLLM、SGLang、TensorRT-LLM 的原理与源码笔记。理解 PagedAttention、Continuous Batching、PD 分离等核心技术。
      </>
    ),
    link: '/docs/category/推理引擎',
    linkText: '进入推理专区 →',
  },
];

function Feature({emoji, title, description, link, linkText}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center" style={{fontSize: '5rem', marginBottom: '1rem'}}>
        {emoji}
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
        <Link className="button button--outline button--primary" to={link}>
          {linkText}
        </Link>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
