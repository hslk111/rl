# -*- coding: utf-8 -*-
"""
Stage 3: GRPO reinforcement learning for Qwen2.5-0.5B-Instruct on GSM8K.

This run writes a new adapter to outputs/lora_v3 and keeps prior adapters
untouched. v3 plan (fixing v2 over-training): correct reward as the dominant
signal (2.0), light full-format reward (0.3), no partial-tag reward, decoupled
answer extraction (format is no longer a prerequisite for the correctness
reward), lower sampling temperature (0.7) to narrow the train/eval distribution
gap, stronger KL (beta=0.06), longer completions (200) to avoid truncated
reasoning, 150 steps with checkpoints at 50/100/150.
"""
import os
import re
import sys

PROJ = r"D:\llm_project"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

MODEL_DIR = os.path.join(PROJ, "models", "Qwen2.5-0.5B-Instruct")
DATA_FILE = os.path.join(PROJ, "data", "gsm8k", "main", "train-00000-of-00001.parquet")
OUTPUTS_DIR = os.path.join(PROJ, "outputs")

RUN_NAME = "lora_v3"
OUT_DIR = os.path.join(OUTPUTS_DIR, RUN_NAME)
METRICS_PATH = os.path.join(OUTPUTS_DIR, "metrics_lora_v3.tsv")
MONITOR_PATH = os.path.join(OUTPUTS_DIR, "monitor_lora_v3.tsv")
TRAIN_LOG_PATH = os.path.join(OUTPUTS_DIR, "train_log_lora_v3.txt")
LATEST_LORA_PATH = os.path.join(OUTPUTS_DIR, "latest_lora_dir.txt")

MAX_STEPS = 150
MONITOR_WINDOW = 50
CORRECT_REWARD = 2.0
FORMAT_REWARD = 0.3
PARTIAL_TAG_REWARD = 0.0
LEARNING_RATE = 1e-5
BETA = 0.06
TEMPERATURE = 0.7
MAX_COMPLETION_LENGTH = 200

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a math question, "
    "and the Assistant solves it. The Assistant first thinks step by step in the mind, "
    "then provides the final numeric answer. The response MUST follow this format exactly:\n"
    "<reasoning>\n...step by step reasoning...\n</reasoning>\n<answer>\n...final number only...\n</answer>"
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class MetricLogger(TrainerCallback):
    def __init__(self, path):
        self.path = path
        with open(path, "w", encoding="utf-8") as f:
            f.write("step\treward\treward_correct\treward_format\treward_partial\tkl\tcompletion_len\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "reward" not in logs:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(
                "{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.5f}\t{:.1f}\n".format(
                    state.global_step,
                    logs.get("reward", 0.0),
                    logs.get("rewards/reward_correct", 0.0),
                    logs.get("rewards/reward_format", 0.0),
                    logs.get("rewards/reward_partial_format", 0.0),
                    logs.get("kl", 0.0),
                    logs.get("completion_length", 0.0),
                )
            )


class WindowMonitor(TrainerCallback):
    def __init__(self, path, window=50):
        self.path = path
        self.window = window
        self.rows = []
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "step\twindow\tavg_reward\tprev_avg_reward\tdelta_reward\t"
                "avg_correct\tprev_avg_correct\tdelta_correct\t"
                "avg_format_total\tprev_avg_format_total\tdelta_format_total\tavg_kl\tnote\n"
            )

    @staticmethod
    def _avg(rows, key):
        return sum(row[key] for row in rows) / len(rows)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "reward" not in logs:
            return

        self.rows.append(
            {
                "reward": logs.get("reward", 0.0),
                "correct": logs.get("rewards/reward_correct", 0.0),
                "format_total": logs.get("rewards/reward_format", 0.0)
                + logs.get("rewards/reward_partial_format", 0.0),
                "kl": logs.get("kl", 0.0),
            }
        )

        if state.global_step < self.window or state.global_step % self.window != 0:
            return

        current = self.rows[-self.window :]
        previous = self.rows[-2 * self.window : -self.window]
        avg_reward = self._avg(current, "reward")
        avg_correct = self._avg(current, "correct")
        avg_format_total = self._avg(current, "format_total")
        avg_kl = self._avg(current, "kl")

        if previous:
            prev_avg_reward = self._avg(previous, "reward")
            prev_avg_correct = self._avg(previous, "correct")
            prev_avg_format_total = self._avg(previous, "format_total")
        else:
            prev_avg_reward = 0.0
            prev_avg_correct = 0.0
            prev_avg_format_total = 0.0

        delta_reward = avg_reward - prev_avg_reward
        delta_correct = avg_correct - prev_avg_correct
        delta_format_total = avg_format_total - prev_avg_format_total
        note = "better" if delta_correct > 0 else "flat_or_worse"

        line = (
            f"{state.global_step}\t{self.window}\t{avg_reward:.4f}\t{prev_avg_reward:.4f}\t"
            f"{delta_reward:.4f}\t{avg_correct:.4f}\t{prev_avg_correct:.4f}\t"
            f"{delta_correct:.4f}\t{avg_format_total:.4f}\t{prev_avg_format_total:.4f}\t"
            f"{delta_format_total:.4f}\t{avg_kl:.5f}\t{note}\n"
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)

        print(
            f"[monitor] step {state.global_step}: "
            f"correct_avg={avg_correct:.4f} ({delta_correct:+.4f}), "
            f"reward_avg={avg_reward:.4f} ({delta_reward:+.4f}), "
            f"format_avg={avg_format_total:.4f} ({delta_format_total:+.4f}), "
            f"kl_avg={avg_kl:.5f}"
        )


def setup_run_logging():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    if os.path.exists(OUT_DIR):
        raise FileExistsError(f"Refusing to overwrite existing LoRA output directory: {OUT_DIR}")
    for path in [METRICS_PATH, MONITOR_PATH, TRAIN_LOG_PATH]:
        if os.path.exists(path):
            raise FileExistsError(f"Refusing to overwrite existing run file: {path}")

    log_file = open(TRAIN_LOG_PATH, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    return log_file


def extract_gt(answer_text):
    return answer_text.split("####")[-1].strip().replace(",", "")


def build_dataset():
    ds = load_dataset("parquet", data_files=DATA_FILE, split="train")

    def fmt(ex):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            "ground_truth": extract_gt(ex["answer"]),
        }

    return ds.map(fmt, remove_columns=ds.column_names)


def extract_pred(text):
    """Decoupled extraction: prefer the number inside <answer>...</answer>;
    if the tags are absent, fall back to the last number in the whole text.
    This separates answer-correctness from format compliance."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    target = m.group(1) if m else text
    nums = re.findall(r"-?\d+\.?\d*", target.replace(",", ""))
    return nums[-1] if nums else None


def reward_correct(completions, ground_truth, **kwargs):
    out = []
    for comp, gt in zip(completions, ground_truth):
        text = comp[0]["content"]
        pred = extract_pred(text)
        try:
            ok = pred is not None and abs(float(pred) - float(gt)) < 1e-4
        except ValueError:
            ok = pred == gt
        out.append(CORRECT_REWARD if ok else 0.0)
    return out


def reward_format(completions, **kwargs):
    pat = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    out = []
    for comp in completions:
        text = comp[0]["content"]
        out.append(FORMAT_REWARD if re.search(pat, text, re.DOTALL) else 0.0)
    return out


def reward_partial_format(completions, **kwargs):
    out = []
    for comp in completions:
        text = comp[0]["content"]
        score = 0.0
        for tag in ["<reasoning>", "</reasoning>", "<answer>", "</answer>"]:
            if tag in text:
                score += PARTIAL_TAG_REWARD
        out.append(score)
    return out


def main():
    log_file = setup_run_logging()
    try:
        print(f"Run name: {RUN_NAME}")
        print(f"Output dir: {OUT_DIR}")
        print(f"Metrics: {METRICS_PATH}")
        print(f"Monitor: {MONITOR_PATH}")
        print(f"Train log: {TRAIN_LOG_PATH}")
        print(
            "Reward weights: "
            f"correct={CORRECT_REWARD}, format={FORMAT_REWARD}, "
            f"partial_tag={PARTIAL_TAG_REWARD} (partial max={PARTIAL_TAG_REWARD * 4})"
        )
        print(
            f"Max steps: {MAX_STEPS}, monitor window: {MONITOR_WINDOW}, "
            f"learning_rate: {LEARNING_RATE}, beta: {BETA}"
        )

        print("Loading dataset...")
        train_ds = build_dataset()
        print(f"Training samples: {len(train_ds)}")

        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )

        cfg = GRPOConfig(
            output_dir=OUT_DIR,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=4,
            max_prompt_length=256,
            max_completion_length=MAX_COMPLETION_LENGTH,
            num_train_epochs=1,
            max_steps=MAX_STEPS,
            logging_steps=1,
            save_steps=50,
            save_total_limit=3,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=False,
            fp16=True,
            beta=BETA,
            temperature=TEMPERATURE,
            report_to="none",
        )

        trainer = GRPOTrainer(
            model=MODEL_DIR,
            processing_class=AutoTokenizer.from_pretrained(MODEL_DIR),
            reward_funcs=[reward_correct, reward_format, reward_partial_format],
            args=cfg,
            train_dataset=train_ds,
            peft_config=lora_cfg,
        )
        trainer.add_callback(MetricLogger(METRICS_PATH))
        trainer.add_callback(WindowMonitor(MONITOR_PATH, window=MONITOR_WINDOW))

        print("Starting GRPO training...")
        trainer.train()
        trainer.save_model(OUT_DIR)

        with open(LATEST_LORA_PATH, "w", encoding="utf-8") as f:
            f.write(OUT_DIR)

        print(f"Training complete. LoRA weights saved to: {OUT_DIR}")
        print(f"Latest LoRA pointer written to: {LATEST_LORA_PATH}")
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()


if __name__ == "__main__":
    main()
