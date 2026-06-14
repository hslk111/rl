# -*- coding: utf-8 -*-
"""
Stage 4: evaluate the base model or a LoRA adapter on GSM8K.

Usage:
  python scripts/04_eval.py base [n_eval]
  python scripts/04_eval.py lora_v1 [n_eval]
  python scripts/04_eval.py lora_v2 [n_eval]
  python scripts/04_eval.py lora latest [n_eval]
  python scripts/04_eval.py lora D:\llm_project\outputs\your-lora-dir [n_eval]

Examples:
  python scripts/04_eval.py lora_v1 200
  python scripts/04_eval.py lora_v2 500
  python scripts/04_eval.py base all
"""
import os
import re
import sys

PROJ = r"D:\llm_project"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.join(PROJ, "models", "Qwen2.5-0.5B-Instruct")
LORA_V1_DIR = os.path.join(PROJ, "outputs", "grpo-qwen0.5b")
LORA_V2_DIR = os.path.join(PROJ, "outputs", "lora_v2")
LATEST_LORA_PATH = os.path.join(PROJ, "outputs", "latest_lora_dir.txt")
TEST_FILE = os.path.join(PROJ, "data", "gsm8k", "main", "test-00000-of-00001.parquet")
DEFAULT_N_EVAL = 200

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a math question, "
    "and the Assistant solves it. The Assistant first thinks step by step in the mind, "
    "then provides the final numeric answer. The response MUST follow this format exactly:\n"
    "<reasoning>\n...step by step reasoning...\n</reasoning>\n<answer>\n...final number only...\n</answer>"
)


def clean_path(path):
    return path.strip().lstrip("\ufeff").strip('"')


def resolve_lora_dir(arg=None):
    if arg in {None, "default", "lora_v1"}:
        return LORA_V1_DIR
    if arg == "lora_v2":
        return LORA_V2_DIR
    if arg == "latest":
        if not os.path.exists(LATEST_LORA_PATH):
            raise FileNotFoundError(f"Latest LoRA pointer not found: {LATEST_LORA_PATH}")
        with open(LATEST_LORA_PATH, "r", encoding="utf-8-sig") as f:
            return clean_path(f.read())
    return clean_path(arg)


def parse_n_eval(value):
    if value is None:
        return DEFAULT_N_EVAL
    value = value.strip().lower()
    if value in {"all", "full"}:
        return None
    n_eval = int(value)
    if n_eval <= 0:
        raise ValueError("n_eval must be a positive integer, 'all', or 'full'")
    return n_eval


def parse_args(argv):
    if not argv:
        return "base", None, DEFAULT_N_EVAL

    mode = argv[0].lower()

    if mode == "base":
        return "base", None, parse_n_eval(argv[1] if len(argv) > 1 else None)

    if mode in {"lora_v1", "lora_v2"}:
        return "lora", mode, parse_n_eval(argv[1] if len(argv) > 1 else None)

    if mode == "lora":
        lora_arg = argv[1] if len(argv) > 1 else "lora_v1"
        n_eval_arg = argv[2] if len(argv) > 2 else None
        return "lora", lora_arg, parse_n_eval(n_eval_arg)

    raise ValueError("mode must be 'base', 'lora_v1', 'lora_v2', or 'lora'")


def extract_gt(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def extract_pred(text):
    # Decoupled extraction (matches 03_train_grpo.py): prefer the number inside
    # <answer>...</answer>; if absent, fall back to the last number in the text.
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    target = m.group(1) if m else text
    nums = re.findall(r"-?\d+\.?\d*", target.replace(",", ""))
    return nums[-1] if nums else None


def main(mode, lora_arg=None, n_eval=DEFAULT_N_EVAL):
    print(f"Eval mode: {mode}")
    if mode == "lora":
        print(f"LoRA selection: {lora_arg}")
    print(f"Eval count: {'all' if n_eval is None else n_eval}")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="cuda"
    )

    if mode == "lora":
        from peft import PeftModel

        lora_dir = resolve_lora_dir(lora_arg)
        if not os.path.exists(lora_dir):
            raise FileNotFoundError(f"LoRA directory not found: {lora_dir}")
        model = PeftModel.from_pretrained(model, lora_dir)
        model = model.merge_and_unload()
        print(f"Loaded LoRA weights: {lora_dir}")
    elif mode != "base":
        raise ValueError("mode must be 'base' or 'lora'")

    model.eval()

    df = pd.read_parquet(TEST_FILE)
    if n_eval is not None:
        df = df.head(n_eval)

    correct, fmt_ok = 0, 0
    for i, row in df.iterrows():
        current = i + 1
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id
            )
        resp = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_pred(resp)
        gt = extract_gt(row["answer"])
        try:
            if pred is not None and abs(float(pred) - float(gt)) < 1e-4:
                correct += 1
        except ValueError:
            if pred == gt:
                correct += 1
        if re.search(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", resp, re.DOTALL):
            fmt_ok += 1
        if current % 20 == 0:
            print(f"  {current}/{len(df)} | accuracy {correct / current:.1%} | format {fmt_ok / current:.1%}")

    print(f"\n=== {mode if mode == 'base' else lora_arg} results ===")
    print(f"Accuracy: {correct}/{len(df)} = {correct / len(df):.1%}")
    print(f"Format match: {fmt_ok}/{len(df)} = {fmt_ok / len(df):.1%}")


if __name__ == "__main__":
    mode, lora_arg, n_eval = parse_args(sys.argv[1:])
    main(mode, lora_arg, n_eval)
