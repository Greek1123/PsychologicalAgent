from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


LOGGER = logging.getLogger("train_eval_behavior_patch_peft")


class ChatSftDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.items: list[dict[str, torch.Tensor]] = []
        for row in rows:
            messages = row.get("messages") or []
            if len(messages) < 2 or messages[-1].get("role") != "assistant":
                continue
            prompt_messages = messages[:-1]
            prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
            if not full_ids:
                continue

            labels = full_ids.copy()
            prompt_len = min(len(prompt_ids), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
            if all(label == -100 for label in labels):
                continue
            self.items.append(
                {
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def configure_cache(cache_root: Path) -> None:
    os.environ.setdefault("MODELSCOPE_CACHE", str(cache_root / "modelscope"))
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "huggingface" / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(cache_root / "huggingface" / "xet"))
    for key in ("MODELSCOPE_CACHE", "HF_HUB_CACHE", "HF_XET_CACHE"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny PEFT SFT patch for evaluation bad cases.")
    parser.add_argument("--base-model", default="D:/llm_cache/modelscope/models/Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--adapter",
        default="D:/psychologicalAgent/training/ms_swift/outputs/curated_behavior_clean_sft/v5-20260508-173940/checkpoint-352",
    )
    parser.add_argument(
        "--dataset",
        default="D:/psychologicalAgent/data/training/feedback_bad_cases/eval_behavior_sft_20260511_ms_swift.jsonl",
    )
    parser.add_argument("--out-dir", default="D:/psychologicalAgent/training/ms_swift/outputs/eval_behavior_patch_peft")
    parser.add_argument("--cache-root", default="D:/llm_cache")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    configure_cache(Path(args.cache_root))

    base_model = Path(args.base_model)
    adapter = Path(args.adapter)
    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not base_model.exists():
        raise FileNotFoundError(f"Base model not found: {base_model}")
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading base model: %s", base_model)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quant_config,
    )
    model = prepare_model_for_kbit_training(model)
    LOGGER.info("Loading trainable adapter: %s", adapter)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    model.config.use_cache = False

    rows = read_jsonl(dataset_path)
    train_dataset = ChatSftDataset(rows, tokenizer, args.max_length)
    if len(train_dataset) == 0:
        raise RuntimeError("No trainable records were built from the dataset.")
    LOGGER.info("Built train dataset with %s records", len(train_dataset))

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        optim="adamw_torch",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    trainer.save_model(str(out_dir / "checkpoint-final"))
    tokenizer.save_pretrained(str(out_dir / "checkpoint-final"))
    print(json.dumps({"output": str(out_dir / "checkpoint-final"), "records": len(train_dataset)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
