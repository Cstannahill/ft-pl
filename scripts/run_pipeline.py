#!/usr/bin/env python3
"""End-to-end fine-tuning pipeline as described in guide.md.

This script orchestrates the stages:
1. Fine-tune a base model with LoRA/QLoRA using HuggingFace Transformers.
2. Merge adapters and export the full model in safetensors format.
3. Quantize the model using a Rust CLI (quantize-rs).
4. Package the quantized weights and tokenizer into a GGUF file using a Rust CLI (gguf-writer).

Each stage can be run individually or the entire pipeline can be executed
with default directory layout.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional

# Optional imports (not strictly required if stages are skipped)
try:
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception:
    # Training dependencies may not be installed in all environments.
    pass

WORKSPACE = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = WORKSPACE / "base_model"
DATA_DIR = WORKSPACE / "data"
FINETUNE_OUTPUT_DIR = WORKSPACE / "finetune_output"
EXPORTED_DIR = WORKSPACE / "exported_model"
QUANTIZED_DIR = WORKSPACE / "quantized"
GGUF_DIR = WORKSPACE / "gguf"


# --------------------------- Stage 1 ---------------------------------------

def run_finetuning(base_model: Path = BASE_MODEL_DIR,
                   data_dir: Path = DATA_DIR,
                   output_dir: Path = FINETUNE_OUTPUT_DIR,
                   num_train_epochs: int = 1) -> None:
    """Fine-tune `base_model` using LoRA/QLoRA."""

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=True,
        device_map="auto",
    )

    # Prepare model for QLoRA training
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset(str(data_dir))
    train_dataset = dataset["train"]

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(str(output_dir))


# --------------------------- Stage 2 ---------------------------------------

def export_merged_model(finetune_dir: Path = FINETUNE_OUTPUT_DIR,
                        output_dir: Path = EXPORTED_DIR) -> Path:
    """Merge LoRA adapters into the base model and export safetensors."""
    model = AutoModelForCausalLM.from_pretrained(
        finetune_dir,
        device_map="auto",
    )
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(finetune_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir / "model.safetensors"


# --------------------------- Stage 3 ---------------------------------------

def quantize_model(model_path: Path,
                   config_path: Path,
                   outfile: Path,
                   fmt: str = "Q4_0") -> None:
    """Invoke the Rust quantizer CLI."""
    cmd = [
        "quantize-rs",
        "--model", str(model_path),
        "--config", str(config_path),
        "--outfile", str(outfile),
        "--format", fmt,
    ]
    subprocess.run(cmd, check=True)


# --------------------------- Stage 4 ---------------------------------------

def write_gguf(quantized_path: Path,
               config_path: Path,
               tokenizer_path: Path,
               outfile: Path,
               name: Optional[str] = None) -> None:
    """Invoke the Rust GGUF writer CLI."""
    cmd = [
        "gguf-writer",
        "--model", str(quantized_path),
        "--config", str(config_path),
        "--tokenizer", str(tokenizer_path),
        "--outfile", str(outfile),
    ]
    if name:
        cmd.extend(["--name", name])
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run fine-tuning to GGUF pipeline")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--quant-format", default="Q4_0", help="Quantization format")
    parser.add_argument("--model-name", default="finetuned-model", help="Name for GGUF package")
    args = parser.parse_args()

    run_finetuning(num_train_epochs=args.epochs)

    exported = export_merged_model()
    config_path = EXPORTED_DIR / "config.json"
    tokenizer_path = EXPORTED_DIR / "tokenizer.model"

    quant_out = QUANTIZED_DIR / f"model.{args.quant_format}.safetensors"
    quantize_model(exported, config_path, quant_out, fmt=args.quant_format)

    gguf_out = GGUF_DIR / f"model.{args.quant_format}.gguf"
    write_gguf(quant_out, config_path, tokenizer_path, gguf_out, name=args.model_name)


if __name__ == "__main__":
    main()
