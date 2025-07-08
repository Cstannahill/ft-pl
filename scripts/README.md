This directory contains automation scripts for the fine-tuning pipeline.

* `run_pipeline.py` - Python implementation of the pipeline described in `guide.md`.
  It performs fine-tuning with LoRA/QLoRA, exports merged weights, quantizes the
  model using the `quantize-rs` CLI, and packages the result into GGUF format via
  `gguf-writer`.
* `run_pipeline.sh` - Convenience wrapper that invokes `run_pipeline.py`.

Run `./run_pipeline.sh` from this directory (or via `scripts/run_pipeline.sh`)
with optional arguments such as `--epochs` or `--quant-format` to execute the
pipeline end-to-end.
