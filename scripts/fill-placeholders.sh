#!/bin/bash
# 给空目录批量创建 index.md 占位页

BASE="/root/.openclaw/workspace/knowledge-garden/docs"

# 一级目录占位（除了 02-quantization 已经写过）
declare -A INDEX_MAP=(
  ["01-learning-path"]="🧭 学习路径|系统化的学习路径与知识地图"
  ["03-rl-post-training"]="🎯 RL 后训练|PPO/DPO/GRPO 等 RLHF 算法与训练 pipeline"
  ["04-training-frameworks"]="⚙️ 训练框架|Megatron / verl / DeepSpeed 主流训练框架"
  ["05-inference-engines"]="🚀 推理引擎|vLLM / SGLang / TensorRT-LLM 推理引擎"
  ["06-fundamentals"]="🧠 基础知识|Transformer / GPU-CUDA / 分布式训练基础"
  ["07-paper-reading"]="📖 论文精读|关键论文的读书笔记"
  ["08-garden"]="🌱 想法花园|随手记的想法、疑问、待深挖话题"
)

for dir in "${!INDEX_MAP[@]}"; do
  IFS='|' read -r title desc <<< "${INDEX_MAP[$dir]}"
  file="$BASE/$dir/index.md"
  if [ ! -f "$file" ]; then
    cat > "$file" <<EOF
---
sidebar_position: 0
title: $title
---

# $title

> $desc

🚧 这个板块还在建设中。回来看看吧！

## 已计划话题

（陆续更新）
EOF
    echo "created: $file"
  fi
done

# 二级子目录占位
find "$BASE" -mindepth 2 -type d | while read subdir; do
  # 跳过 practice-notes（已写过）
  if [[ "$subdir" == *"practice-notes"* ]]; then continue; fi
  index_file="$subdir/index.md"
  if [ ! -f "$index_file" ]; then
    # 从 _category_.json 里读 label
    label=$(cat "$subdir/_category_.json" 2>/dev/null | grep -oP '"label":\s*"\K[^"]+' | head -1)
    [ -z "$label" ] && label=$(basename "$subdir")
    cat > "$index_file" <<EOF
---
sidebar_position: 0
title: $label
---

# $label

🌱 这个子模块的笔记正在陆续填充中。
EOF
    echo "created: $index_file"
  fi
done
