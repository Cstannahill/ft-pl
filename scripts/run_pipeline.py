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
import logging
from pathlib import Path
from typing import Optional
import itertools
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from colorama import Fore, Style, init as colorama_init
from tqdm import tqdm

# Optional imports (not strictly required if stages are skipped)
try:
    from datasets import load_dataset

except Exception:
    # Training dependencies may not be installed in all environments.
    pass

WORKSPACE = Path(__file__).resolve().parents[1]
BASE_MODELS_DIR = WORKSPACE / "base_models"
DATA_DIR = WORKSPACE / "data"
FINETUNE_OUTPUT_DIR = WORKSPACE / "finetune_output"
EXPORTED_DIR = WORKSPACE / "exported_model"
QUANTIZED_DIR = WORKSPACE / "quantized"
GGUF_DIR = WORKSPACE / "gguf"

logging.basicConfig(level=logging.INFO, format="%(message)s")
colorama_init(autoreset=True)
logger = logging.getLogger(__name__)


def unique_path(path: Path) -> Path:
    """Return a non-conflicting path by appending an index if needed."""
    if not path.exists():
        return path
    for i in itertools.count(1):
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate


# --------------------------- Stage 1 ---------------------------------------

def run_finetuning(base_model: Path,
                   data_dir: Path = DATA_DIR,
                   output_dir: Path = FINETUNE_OUTPUT_DIR,
                   num_train_epochs: int = 1) -> None:
    """Fine-tune `base_model` using LoRA/QLoRA."""

    logger.info(f"{Fore.GREEN}Stage 1: Fine-tuning model{Style.RESET_ALL}")
    logger.debug(f"Base model: {base_model}")
    logger.debug(f"Dataset path: {data_dir}")

    # Use BitsAndBytesConfig to avoid deprecated warning
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

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

    raw_ds = load_dataset("json", data_files=str(data_dir / "dataset.jsonl"))["train"]

    max_seq_len = 768

    def format_example(example: dict) -> dict:
        prompt = example["instruction"]
        if example["input"]:
            prompt += f"\n{example['input']}"
        prompt += "\n\n### Response:\n" + example["output"]

        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            return_tensors="pt",
        )

        # Important: Ensure `labels` matches `input_ids` exactly
        encoded["labels"] = encoded["input_ids"].clone()

        # Debug tensor shapes
        print("Input shape:", encoded["input_ids"].shape)
        print("Label shape:", encoded["labels"].shape)

        return {k: v.squeeze(0) for k, v in encoded.items()}

    train_dataset = raw_ds.map(format_example, remove_columns=raw_ds.column_names)

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
    logger.info("Fine-tuning complete")


# --------------------------- Stage 2 ---------------------------------------
def export_merged_model(finetune_dir: Path = FINETUNE_OUTPUT_DIR,
                        output_dir: Path = EXPORTED_DIR,
                        base_model_dir: Path = None) -> Path:
    """Merge LoRA adapter into base model and export full model.safetensors."""
    logger.info(f"{Fore.GREEN}Stage 2: Exporting merged model{Style.RESET_ALL}")
    from peft import AutoPeftModelForCausalLM

    # Load adapter
    model = AutoPeftModelForCausalLM.from_pretrained(
        finetune_dir,
        device_map="cpu",
        torch_dtype=torch.float16,
    )

    # Merge LoRA into base weights
    model = model.merge_and_unload()

    # Save merged full model to model.safetensors
    model.save_pretrained(output_dir, safe_serialization=True)

    # Export tokenizer from base or finetune dir
    tokenizer_source = base_model_dir or finetune_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    tokenizer.save_pretrained(output_dir)

    logger.info("Export complete")
    return output_dir / "model.safetensors"


# --------------------------- Stage 3 ---------------------------------------

def quantize_model(model_path: Path,
                   outfile: Path,
                   fmt: str = "Q4_0") -> None:
    """Invoke the Rust quantizer CLI."""
    logger.info(f"{Fore.GREEN}Stage 3: Quantizing model{Style.RESET_ALL}")
    quantize_bin = WORKSPACE / "rust-gguf-tools" / "target" / "release" / "quantize-rs"

    cmd = [
        str(quantize_bin),
        "--input", str(model_path),
        "--output", str(outfile),
        "--format", fmt,
    ]
    subprocess.run(cmd, check=True)
    logger.info("Quantization complete")


# --------------------------- Stage 4 ---------------------------------------

def write_gguf(quantized_path: Path,
               config_path: Path,
               tokenizer_path: Path,
               outfile: Path,
               name: Optional[str] = None) -> None:
    """Invoke the Rust GGUF writer CLI."""
    logger.info(f"{Fore.GREEN}Stage 4: Writing GGUF package{Style.RESET_ALL}")
    logger.debug(f"Quantized weights: {quantized_path}")
    logger.debug(f"Output file: {outfile}")
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
    logger.info("GGUF package created")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run fine-tuning to GGUF pipeline")
    parser.add_argument("-e","--epochs", type=int, help="Number of training epochs")
    parser.add_argument("-q","--quant-format", help="Quantization format")
    parser.add_argument("-m","--model-name", help="Name for GGUF package")
    parser.add_argument("-b","--base-model",
                        help="Directory name of the base model inside base_models/")
    args = parser.parse_args()

    # Interactive prompts for missing values
    if args.base_model is None:
        available = [p.name for p in BASE_MODELS_DIR.glob('*') if p.is_dir()]
        prompt = "Base model directory"
        if available:
            prompt += f" ({', '.join(available)})"
        prompt += ": "
        args.base_model = input(prompt).strip()

    if args.epochs is None:
        inp = input("Number of training epochs [1]: ").strip()
        args.epochs = int(inp) if inp else 1

    if args.quant_format is None:
        args.quant_format = input("Quantization format [Q4_0]: ").strip() or "Q4_0"

    if args.model_name is None:
        args.model_name = input("Name for GGUF package [finetuned-model]: ").strip() or "finetuned-model"

    logger.info(f"{Fore.CYAN}Starting fine-tuning pipeline{Style.RESET_ALL}")

    base_model_dir = BASE_MODELS_DIR / args.base_model
    if not base_model_dir.exists():
        raise FileNotFoundError(f"Base model directory '{base_model_dir}' does not exist")

    with tqdm(total=4, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        pbar.set_description("Finetuning")
        run_finetuning(base_model=base_model_dir, num_train_epochs=args.epochs)
        pbar.update(1)

        pbar.set_description("Exporting")
        exported = export_merged_model(base_model_dir=base_model_dir)

        pbar.update(1)

        config_path = EXPORTED_DIR / "config.json"
        tokenizer_path = EXPORTED_DIR / "tokenizer.model"
        quant_out = unique_path(QUANTIZED_DIR / f"model.{args.quant_format}.safetensors")
        pbar.set_description("Quantizing")
        quantize_model(exported, quant_out, fmt=args.quant_format)
        pbar.update(1)

        gguf_out = unique_path(GGUF_DIR / f"model.{args.quant_format}.gguf")
        pbar.set_description("Packaging")
        write_gguf(quant_out, gguf_out, name=args.model_name)
        pbar.update(1)

    logger.info(f"{Fore.CYAN}Pipeline completed successfully{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
