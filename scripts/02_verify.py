"""快速验证：模型能加载+推理，数据能读+格式正确"""
import os
PROJ = r"D:\llm_project"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

MODEL_DIR = os.path.join(PROJ, "models", "Qwen2.5-0.5B-Instruct")

print("=== 1. 加载模型 ===")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16, device_map="cuda"
)
print(f"模型已加载到 GPU | 参数量约 {sum(p.numel() for p in model.parameters())/1e6:.0f}M")
print(f"显存占用: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("\n=== 2. 推理测试 ===")
q = "Natalia sold clips to 48 friends in April, and half as many in May. How many clips did she sell altogether?"
msgs = [{"role": "user", "content": q}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("模型回答:\n", resp[:400])

print("\n=== 3. 数据格式 ===")
df = pd.read_parquet(os.path.join(PROJ, "data", "gsm8k", "main", "train-00000-of-00001.parquet"))
print(f"训练集大小: {len(df)}")
print("第一条 question:", df.iloc[0]["question"][:100])
print("第一条 answer:", repr(df.iloc[0]["answer"][-60:]))
