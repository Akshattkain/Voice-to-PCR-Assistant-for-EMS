"""Fine-tune T5-base with PEFT/LoRA for transcript -> PCR JSON extraction.

Usage:
    python -m training.fine_tune.train_t5 --data_dir training/data/processed --output_dir models/t5_pcr_v1
"""

import argparse
import json
from pathlib import Path

from training.fine_tune.config import TrainingConfig


def main(config: TrainingConfig):
    """Run fine-tuning."""
    # Deferred imports to avoid loading torch at app startup
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    from training.fine_tune.dataset import build_hf_dataset

    print(f"Loading base model: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model)

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"Loading dataset from {config.data_dir}")
    splits = build_hf_dataset(config.data_dir)

    if "train" not in splits:
        raise ValueError(f"No training data found in {config.data_dir}")

    # Tokenize
    def preprocess(examples):
        inputs = tokenizer(
            examples["source"],
            max_length=config.max_source_length,
            truncation=True,
            padding="max_length",
        )
        targets = tokenizer(
            examples["target"],
            max_length=config.max_target_length,
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    train_dataset = splits["train"].map(preprocess, batched=True)
    val_dataset = splits.get("validation")
    if val_dataset:
        val_dataset = val_dataset.map(preprocess, batched=True)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        fp16=config.fp16 and torch.cuda.is_available(),
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=config.eval_steps if val_dataset else None,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        load_best_model_at_end=bool(val_dataset),
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # Save model
    print(f"Saving model to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune T5 for PCR extraction")
    parser.add_argument("--data_dir", default="training/data/processed")
    parser.add_argument("--output_dir", default="models/t5_pcr_v1")
    parser.add_argument("--base_model", default="t5-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    main(config)
