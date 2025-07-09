import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM


def merge_and_export(base_model_dir: Path, adapter_model_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Loading adapter model from: {adapter_model_dir}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_model_dir,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print("🔧 Merging adapter weights into base model...")
    model = model.merge_and_unload()

    print(f"💾 Saving merged model to: {output_dir}")
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="30GB",  # ⬅️ Prevent automatic sharding
    )

    print("📦 Copying tokenizer and config files...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, use_fast=True)
    tokenizer.save_pretrained(output_dir)

    print("✅ Export complete!")



def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model and export full model.")
    parser.add_argument("-b", "--base_model", required=True, help="Path to base model directory")
    parser.add_argument("-a", "--adapter_model", required=True, help="Path to fine-tuned adapter model directory")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to output merged model")

    args = parser.parse_args()

    merge_and_export(
        base_model_dir=Path(args.base_model),
        adapter_model_dir=Path(args.adapter_model),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
