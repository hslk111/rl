"""
阶段1：下载 Qwen2.5-0.5B-Instruct 模型 + GSM8K 数据集
全部走 hf-mirror.com 镜像，缓存到 D:\llm_project 下。
"""
import os

# === 关键：所有缓存指向 D 盘，走国内镜像 ===
PROJ = r"D:\llm_project"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(PROJ, "models", "hf_home")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Windows 上关掉以免报错

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_DIR = os.path.join(PROJ, "models", "Qwen2.5-0.5B-Instruct")

print(f"[1/2] 下载模型 {MODEL_ID} -> {MODEL_DIR}")
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_DIR,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "vocab*", "merges*"],
)
print("    模型下载完成")

# === GSM8K 数据集 ===
DATA_DIR = os.path.join(PROJ, "data", "gsm8k")
print(f"[2/2] 下载 GSM8K 数据集 -> {DATA_DIR}")
snapshot_download(
    repo_id="openai/gsm8k",
    repo_type="dataset",
    local_dir=DATA_DIR,
)
print("    数据集下载完成")
print("\n阶段1 完成。")
