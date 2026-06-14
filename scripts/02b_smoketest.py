# -*- coding: utf-8 -*-
"""冒烟测试：用最小配置跑通 GRPO 链路（3 步），确认不报错、不爆显存。"""
import os, sys
PROJ = r"D:\llm_project"
sys.path.insert(0, os.path.join(PROJ, "scripts"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# 复用正式脚本里的数据/奖励函数
import importlib.util
spec = importlib.util.spec_from_file_location("t", os.path.join(PROJ, "scripts", "03_train_grpo.py"))
t = importlib.util.module_from_spec(spec)
# 阻止它执行 main()
import builtins
spec.loader.exec_module(t)

MODEL_DIR = t.MODEL_DIR

def main():
    print("加载数据集（取前 32 条做冒烟测试）...")
    ds = t.build_dataset().select(range(32))

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM",
    )
    cfg = GRPOConfig(
        output_dir=os.path.join(PROJ, "outputs", "smoketest"),
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_prompt_length=256,
        max_completion_length=128,
        max_steps=3,
        logging_steps=1,
        save_strategy="no",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=False, fp16=True,
        beta=0.04, temperature=0.9,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=MODEL_DIR,
        processing_class=AutoTokenizer.from_pretrained(MODEL_DIR),
        reward_funcs=[t.reward_correct, t.reward_format, t.reward_partial_format],
        args=cfg, train_dataset=ds, peft_config=lora_cfg,
    )
    print("开始冒烟测试（3 步）...")
    trainer.train()
    print(f"\n冒烟测试通过！峰值显存: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

if __name__ == "__main__":
    main()
