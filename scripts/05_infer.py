# -*- coding: utf-8 -*-
"""
Stage 5: inference for the base model or a LoRA adapter.


"""
import os
import sys

PROJ = r"D:\llm_project"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.join(PROJ, "models", "Qwen2.5-0.5B-Instruct")
DEFAULT_LORA_DIR = os.path.join(PROJ, "outputs", "grpo-qwen0.5b")
LATEST_LORA_PATH = os.path.join(PROJ, "outputs", "latest_lora_dir.txt")
DEFAULT_QUESTION = (
    "Weng earns $12 an hour for babysitting. Yesterday, she babysat for 50 minutes. "
    "How much did she earn?"
)

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a math question, "
    "and the Assistant solves it. The Assistant first thinks step by step in the mind, "
    "then provides the final numeric answer. The response MUST follow this format exactly:\n"
    "<reasoning>\n...step by step reasoning...\n</reasoning>\n<answer>\n...final number only...\n</answer>"
)


def resolve_lora_dir(arg=None):
    if arg in {None, "default"}:
        return DEFAULT_LORA_DIR
    if arg == "latest":
        if not os.path.exists(LATEST_LORA_PATH):
            raise FileNotFoundError(f"Latest LoRA pointer not found: {LATEST_LORA_PATH}")
        with open(LATEST_LORA_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    # Short name like "lora_v2" -> D:\llm_project\outputs\lora_v2
    if not os.path.isabs(arg) and os.sep not in arg and "/" not in arg:
        candidate = os.path.join(PROJ, "outputs", arg)
        if os.path.isdir(candidate):
            return candidate
    return arg


def parse_args(argv):
    if not argv:
        return "lora", None, DEFAULT_QUESTION

    first = argv[0].lower()
    if first == "base":
        return "base", None, " ".join(argv[1:]).strip() or DEFAULT_QUESTION

    if first == "lora":
        if len(argv) >= 2 and (argv[1].lower() in {"latest", "default"}
                               or os.path.isdir(argv[1])
                               or os.path.isdir(os.path.join(PROJ, "outputs", argv[1]))):
            return "lora", argv[1], " ".join(argv[2:]).strip() or DEFAULT_QUESTION
        return "lora", None, " ".join(argv[1:]).strip() or DEFAULT_QUESTION

    # First arg is a bare LoRA directory path: "<lora_dir> <question>"
    if os.path.isdir(argv[0]):
        return "lora", argv[0], " ".join(argv[1:]).strip() or DEFAULT_QUESTION

    # First arg is a short LoRA name under outputs/, e.g. "lora_v2": "<name> <question>"
    if os.path.isdir(os.path.join(PROJ, "outputs", argv[0])):
        return "lora", argv[0], " ".join(argv[1:]).strip() or DEFAULT_QUESTION

    return "lora", None, " ".join(argv).strip()


def load_model(mode, lora_arg=None):
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="cuda"
    )

    if mode == "lora":
        lora_dir = resolve_lora_dir(lora_arg)
        if not os.path.exists(lora_dir):
            raise FileNotFoundError(f"LoRA directory not found: {lora_dir}")
        model = PeftModel.from_pretrained(model, lora_dir)
        print(f"[mode: lora] Loaded LoRA weights: {lora_dir}\n")
    else:
        print("[mode: base] Using the original base model only.\n")

    model.eval()
    return tok, model


def main(mode, lora_arg, question):
    tok, model = load_model(mode, lora_arg)

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    resp = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Question: {question}\n")
    print(f"Model response:\n{resp}")


if __name__ == "__main__":
    mode, lora_arg, question = parse_args(sys.argv[1:])
    main(mode, lora_arg, question)
