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

from colorama import Fore, Style, init as colorama_init
from tqdm import tqdm

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
BASE_MODELS_DIR = WORKSPACE / "base_models"
DATA_DIR = WORKSPACE / "data"
FINETUNE_OUTPUT_DIR = WORKSPACE / "finetune_output"
EXPORTED_DIR = WORKSPACE / "exported_model"
QUANTIZED_DIR = WORKSPACE / "quantized"
GGUF_DIR = WORKSPACE / "gguf"

logging.basicConfig(level=logging.INFO, format="%(message)s")
colorama_init(autoreset=True)
logger = logging.getLogger(__name__)


# --------------------------- Stage 1 ---------------------------------------

def run_finetuning(base_model: Path,
                   data_dir: Path = DATA_DIR,
                   output_dir: Path = FINETUNE_OUTPUT_DIR,
                   num_train_epochs: int = 1) -> None:
    """Fine-tune `base_model` using LoRA/QLoRA."""

    logger.info(f"{Fore.GREEN}Stage 1: Fine-tuning model{Style.RESET_ALL}")
    logger.debug(f"Base model: {base_model}")
    logger.debug(f"Dataset path: {data_dir}")

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
    logger.info("Fine-tuning complete")


# --------------------------- Stage 2 ---------------------------------------

def export_merged_model(finetune_dir: Path = FINETUNE_OUTPUT_DIR,
                        output_dir: Path = EXPORTED_DIR) -> Path:
    """Merge LoRA adapters into the base model and export safetensors."""
    logger.info(f"{Fore.GREEN}Stage 2: Exporting merged model{Style.RESET_ALL}")
    logger.debug(f"Finetuned directory: {finetune_dir}")
    logger.debug(f"Export directory: {output_dir}")
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
    logger.info("Export complete")
    return output_dir / "model.safetensors"


# --------------------------- Stage 3 ---------------------------------------

def quantize_model(model_path: Path,
                   config_path: Path,
                   outfile: Path,
                   fmt: str = "Q4_0") -> None:
    """Invoke the Rust quantizer CLI."""
    logger.info(f"{Fore.GREEN}Stage 3: Quantizing model{Style.RESET_ALL}")
    logger.debug(f"Model path: {model_path}")
    logger.debug(f"Output file: {outfile}")
    cmd = [
        "quantize-rs",
        "--model", str(model_path),
        "--config", str(config_path),
        "--outfile", str(outfile),
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
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--quant-format", default="Q4_0", help="Quantization format")
    parser.add_argument("--model-name", default="finetuned-model", help="Name for GGUF package")
    parser.add_argument("--base-model", required=True,
                        help="Directory name of the base model inside base_models/")
    args = parser.parse_args()

    logger.info(f"{Fore.CYAN}Starting fine-tuning pipeline{Style.RESET_ALL}")

    base_model_dir = BASE_MODELS_DIR / args.base_model

    with tqdm(total=4, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        pbar.set_description("Finetuning")
        run_finetuning(base_model=base_model_dir, num_train_epochs=args.epochs)
        pbar.update(1)

        pbar.set_description("Exporting")
        exported = export_merged_model()
        pbar.update(1)

        config_path = EXPORTED_DIR / "config.json"
        tokenizer_path = EXPORTED_DIR / "tokenizer.model"
        quant_out = QUANTIZED_DIR / f"model.{args.quant_format}.safetensors"
        pbar.set_description("Quantizing")
        quantize_model(exported, config_path, quant_out, fmt=args.quant_format)
        pbar.update(1)

        gguf_out = GGUF_DIR / f"model.{args.quant_format}.gguf"
        pbar.set_description("Packaging")
        write_gguf(quant_out, config_path, tokenizer_path, gguf_out, name=args.model_name)
        pbar.update(1)

    logger.info(f"{Fore.CYAN}Pipeline completed successfully{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
