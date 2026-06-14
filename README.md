# 大模型强化学习实战：用 GRPO 训练 Qwen2.5-0.5B 做数学推理

## 目录结构

```
D:\llm_project\
├─ 项目知识与规划.txt      原理与路径规划
├─ README.md              本文件
├─ scripts\
│  ├─ 01_download.py      下载模型+GSM8K(走hf-mirror镜像)
│  ├─ 02_verify.py        验证模型加载/推理/数据格式
│  ├─ 02b_smoketest.py    冒烟测试(3步跑通链路)
│  ├─ 03_train_grpo.py    核心GRPO训练脚本
│  ├─ 04_eval.py          评估对比(base/lora)
│  ├─ 05_infer.py         加载训练后模型做推理
│  └─ test_rewards.py     奖励函数单元测试
├─ models\                模型缓存
├─ data\gsm8k\            数据集
└─ outputs\
   ├─ grpo-qwen0.5b\      训练产出的LoRA权重
   ├─ metrics.tsv         逐步奖励记录
   └─ train_log.txt       训练日志
```

## 复现步骤

```bash
conda activate llmrl
cd D:\llm_project
python scripts/01_download.py        # 下载模型和数据
python scripts/03_train_grpo.py      # GRPO训练(120步,约2小时)
python scripts/04_eval.py base       # 评估原始模型
python scripts/04_eval.py lora       # 评估训练后模型
python scripts/05_infer.py "你的数学题"   # 用训练后模型答题
```
