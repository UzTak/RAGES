from __future__ import annotations

"""
QLoRA SFT training entrypoint for the reasoning model.

Input dataset format (JSON/JSONL):
  [
    {
      "input": {
        "x0": [...],
        "oec0_modified": [...],
        "koz_param": [...],
        "artms_scaling_1e3": [...],
        "intent_priority": [...]
      },
      "output": {
        "reasoning": "...",
        "tf": 123,
        "b_seq": [..]
      }
    },
    ...
  ]

Output artifacts:
  - LoRA adapter weights + tokenizer files in --output-dir
  - train args snapshot (train_args.json)
  - learning curve plot (loss_curve_reasoning.png)
"""

import argparse
import importlib.metadata as importlib_metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from packaging.version import Version

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

import os, sys
def find_root_path(path:str, word:str):
    parts = path.split(word, 1)
    return parts[0] + word if len(parts) > 1 else path 
root_folder = Path(__file__).resolve().parents[1]

from parameters import ANSWER_TAG, END_TAG, THINK_TAG
from utils import (
    align_special_tokens,
    apply_chat_prompt,
    contiguous_train_eval_split,
    format_reasoning_user_prompt,
    make_reasoning_answer_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT for reasoning trajectory model.")

    parser.add_argument(
        "--data-path",
        type=str,
        default="rpod/rages/reasoning_data/reasoning_dataset30k_v4.json",
    )
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", type=str, default="rpod/rages/reasoning_model/v4")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=1024)

    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--report-to", type=str, default="none")

    parser.add_argument("--target-modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--reasoning-weight", type=float, default=1.0)
    parser.add_argument("--answer-weight", type=float, default=5.0)

    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--compute-dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--live-curve", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--curve-update-steps", type=int, default=1)

    return parser.parse_args()


def load_raw_examples(path: Path) -> List[Dict[str, Any]]:
    """Load reasoning examples from .json or .jsonl into a list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected dataset JSON to be a list of {input, output} objects.")
    return data


def encode_example(
    ex: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_seq_len: int,
) -> Optional[Dict[str, Any]]:
    """
    Convert one raw sample into token-level training tensors.

    Labels are masked for prompt tokens (-100) so loss is only computed on the
    assistant output. We keep `answer_start` to upweight answer tokens later.
    """

    # Build prompt + assistant text with explicit reasoning/answer boundaries.
    user_prompt = format_reasoning_user_prompt(ex["input"])
    prompt_text = apply_chat_prompt(tokenizer, user_prompt)

    reasoning_text = f"{THINK_TAG}\n{ex['output']['reasoning'].strip()}\n"
    answer_text = f"{ANSWER_TAG}\n{make_reasoning_answer_payload(ex['output'])}\n{END_TAG}\n"
    eos = tokenizer.eos_token or ""
    assistant_text = reasoning_text + answer_text + eos
    full_text = prompt_text + assistant_text

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    reasoning_ids = tokenizer(reasoning_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    labels = full_ids.copy()
    prompt_len = len(prompt_ids)
    labels[:prompt_len] = [-100] * prompt_len

    answer_start = prompt_len + len(reasoning_ids)
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    if answer_start >= len(full_ids) - 1:
        return None
    if all(x == -100 for x in labels):
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "answer_start": int(answer_start),
    }


def prepare_dataset(
    raw_examples: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_seq_len: int,
) -> Dataset:
    """Tokenize all samples and drop unusable ones (e.g., fully truncated labels)."""
    encoded = [
        encode_example(ex, tokenizer=tokenizer, max_seq_len=max_seq_len)
        for ex in raw_examples
    ]
    processed = [item for item in encoded if item is not None]
    dropped = len(encoded) - len(processed)

    if not processed:
        raise ValueError("No valid samples after tokenization/truncation.")

    print(f"[data] kept={len(processed)} dropped={dropped}")
    return Dataset.from_list(processed)


@dataclass
class ReasoningCollator:
    """Pad variable-length tokenized samples and keep `answer_start` per sample."""
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        max_len = batch["input_ids"].shape[1]
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        answer_start = torch.zeros(len(features), dtype=torch.long)

        for i, f in enumerate(features):
            seq_len = len(f["labels"])
            labels[i, :seq_len] = torch.tensor(f["labels"], dtype=torch.long)
            answer_start[i] = min(int(f["answer_start"]), max(seq_len - 1, 0))

        batch["labels"] = labels
        batch["answer_start"] = answer_start
        return batch


class WeightedLossTrainer(Trainer):
    """Trainer with weighted token loss: answer tokens > reasoning tokens > prompt (ignored)."""
    def __init__(self, *args, reasoning_weight: float, answer_weight: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoning_weight = float(reasoning_weight)
        self.answer_weight = float(answer_weight)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        answer_start = inputs.pop("answer_start")
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        ).view_as(shift_labels)

        valid = shift_labels.ne(-100)
        pos = torch.arange(shift_labels.size(1), device=shift_labels.device).unsqueeze(0)
        answer_start = answer_start.to(shift_labels.device)
        answer_start_shift = torch.clamp(answer_start - 1, min=0).unsqueeze(1)

        weights = torch.full_like(token_loss, self.reasoning_weight)
        answer_weights = torch.full_like(token_loss, self.answer_weight)
        weights = torch.where(pos >= answer_start_shift, answer_weights, weights)

        valid_f = valid.to(token_loss.dtype)
        weighted_sum = (token_loss * weights * valid_f).sum()
        weight_denom = (weights * valid_f).sum().clamp_min(1e-8)
        loss = weighted_sum / weight_denom

        if return_outputs:
            return loss, outputs
        return loss


def resolve_compute_dtype(arg: str) -> torch.dtype:
    if arg == "bf16":
        return torch.bfloat16
    if arg == "fp16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def parse_target_modules(raw: str) -> List[str]:
    modules = [x.strip() for x in raw.split(",") if x.strip()]
    if not modules:
        raise ValueError("No valid target modules provided.")
    return modules


def build_model(args: argparse.Namespace):
    """
    Build 4-bit quantized base model + LoRA adapters.

    Includes a version guard for bitsandbytes/torch compatibility to fail fast
    with an actionable message.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. This training script expects GPU.")

    try:
        bnb_ver = Version(importlib_metadata.version("bitsandbytes"))
    except Exception as e:
        raise RuntimeError(
            "bitsandbytes is required for QLoRA but not available in this environment."
        ) from e
    torch_ver = Version(torch.__version__.split("+")[0])
    if bnb_ver >= Version("0.49.0") and torch_ver < Version("2.3.0"):
        raise RuntimeError(
            f"Incompatible versions: bitsandbytes=={bnb_ver} with torch=={torch.__version__}. "
            "bitsandbytes>=0.49 requires torch>=2.3. "
            "Fix: poetry run pip install 'bitsandbytes==0.43.3' --no-deps"
        )

    compute_dtype = resolve_compute_dtype(args.compute_dtype)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model_kwargs: Dict[str, Any] = {
        "quantization_config": bnb_cfg,
        "device_map": "auto",
        "trust_remote_code": args.trust_remote_code,
    }

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=parse_target_modules(args.target_modules),
    )
    model = get_peft_model(model, lora_cfg)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def save_loss_curve(log_history: List[Dict[str, Any]], out_dir: Path) -> None:
    """Plot train/eval loss vs step from Trainer log history."""
    train_steps: List[int] = []
    train_losses: List[float] = []
    eval_steps: List[int] = []
    eval_losses: List[float] = []

    for row in log_history:
        step = int(row.get("step", len(train_steps)))
        if "loss" in row and "eval_loss" not in row:
            train_steps.append(step)
            train_losses.append(float(row["loss"]))
        if "eval_loss" in row:
            eval_steps.append(step)
            eval_losses.append(float(row["eval_loss"]))

    if not train_losses and not eval_losses:
        print("[plot] no train/eval loss found in log history; skip curve plot.")
        return

    plt.figure(figsize=(8, 5))
    if train_losses:
        plt.plot(train_steps, train_losses, label="train")
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="eval")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("learning curve")
    # plt.grid(alpha=0.25)
    # plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "loss_curve_reasoning.png"
    plt.savefig(out_path, dpi=180)
    plt.close()

    # points = {
    #     "train": [{"step": s, "loss": l} for s, l in zip(train_steps, train_losses)],
    #     "eval": [{"step": s, "loss": l} for s, l in zip(eval_steps, eval_losses)],
    # }
    # with open(out_dir / "loss_curve_points_reasoning.json", "w", encoding="utf-8") as f:
    #     json.dump(points, f, indent=2)

    print(f"[plot] saved learning curve: {out_path}")


class LossCurveCallback(TrainerCallback):
    """Periodically refresh loss curve files while training is running."""
    def __init__(self, out_dir: Path, update_steps: int = 10):
        self.out_dir = out_dir
        self.update_steps = max(1, int(update_steps))
        self._last_step = -1

    def _maybe_save(self, state):
        step = int(getattr(state, "global_step", 0))
        if step <= 0 or step == self._last_step:
            return
        if step % self.update_steps != 0:
            return
        save_loss_curve(state.log_history, self.out_dir)
        self._last_step = step

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._maybe_save(state)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._maybe_save(state)
        return control


if __name__ == "__main__":
    # High-level flow:
    # 1) parse args and load/encode dataset
    # 2) build quantized LoRA model
    # 3) train with weighted token loss
    # 4) save adapter/tokenizer and training diagnostics

    args = parse_args()
    set_seed(args.seed)

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_examples = load_raw_examples(data_path)
    dataset = prepare_dataset(raw_examples, tokenizer=tokenizer, max_seq_len=args.max_seq_length)

    train_dataset, eval_dataset = contiguous_train_eval_split(dataset, args.val_ratio)

    print(f"[split] train={len(train_dataset)} val={0 if eval_dataset is None else len(eval_dataset)}")

    model = build_model(args)
    align_special_tokens(model, tokenizer)
    collator = ReasoningCollator(tokenizer=tokenizer)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = not bf16
    if args.compute_dtype == "bf16":
        bf16, fp16 = True, False
    elif args.compute_dtype == "fp16":
        bf16, fp16 = False, True

    has_eval = eval_dataset is not None

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        evaluation_strategy="steps" if has_eval else "no",
        save_strategy="steps",
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        report_to=args.report_to,
        seed=args.seed,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        reasoning_weight=args.reasoning_weight,
        answer_weight=args.answer_weight,
    )
    if args.live_curve:
        trainer.add_callback(LossCurveCallback(output_dir, update_steps=args.curve_update_steps))

    trainer.train()
    if has_eval:
        print(
            f"[best] checkpoint={trainer.state.best_model_checkpoint} "
            f"eval_loss={trainer.state.best_metric}"
        )
    save_loss_curve(trainer.state.log_history, output_dir)

    # Save adapter weights (LoRA) and tokenizer.
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with open(output_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[done] saved LoRA adapter to: {output_dir}")
