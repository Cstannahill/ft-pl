#!/usr/bin/env python3
"""Interactive script to download base models from Hugging Face."""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub is required. Install via `pip install huggingface_hub`.", file=sys.stderr)
    sys.exit(1)

WORKSPACE = Path(__file__).resolve().parents[1]
BASE_DIR = WORKSPACE / "base_model"

MODELS = {
    "Meta Llama 3 8B": "meta-llama/Meta-Llama-3.1-8B",
    "Gemma 7B": "google/gemma-7b",
    "Phi-2": "microsoft/phi-2",
    "Llama 2 7B": "meta-llama/Llama-2-7b-hf",
    "Mistral 7B": "mistralai/Mistral-7B-v0.1",
    "Mistral 7B Instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen1.5 7B": "Qwen/Qwen1.5-7B",
    "Zephyr 7B Beta": "huggingfaceh4/zephyr-7b-beta",
    "Flan-T5 Base": "google/flan-t5-base",
    "Gemma 2B": "google/gemma-2b",
    "StableLM Zephyr 3B": "stabilityai/stablelm-zephyr-3b",
    "Falcon 7B": "tiiuae/falcon-7b",
    "OPT 1.3B": "facebook/opt-1.3b",
}


def choose_model() -> str:
    print("Available models:\n")
    items = list(MODELS.items())
    for idx, (name, repo) in enumerate(items, start=1):
        print(f"{idx}. {name} ({repo})")
    print()
    while True:
        choice = input(f"Select a model [1-{len(items)}]: ")
        if not choice.isdigit():
            print("Please enter a number.")
            continue
        idx = int(choice)
        if 1 <= idx <= len(items):
            return items[idx - 1][1]
        print("Invalid selection. Try again.")


def download_model(repo_id: str) -> None:
    dest = BASE_DIR / repo_id.replace("/", "_")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} to {dest} ...")
    snapshot_download(repo_id, local_dir=dest, local_dir_use_symlinks=False)
    print("Download complete.")


def main() -> None:
    BASE_DIR.mkdir(exist_ok=True)
    repo_id = choose_model()
    download_model(repo_id)


if __name__ == "__main__":
    main()
