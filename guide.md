# Fine-Tuning to GGUF Pipeline (Rust-Based Quantization & Packaging)

## Overview of the Pipeline

This guide describes a complete end-to-end pipeline for fine-tuning large language models (e.g. Meta’s **LLaMA 3**, DeepMind’s **Gemma**, **Phi**, etc.) and converting them into the **GGUF** format using **Rust-based tools**. The pipeline is designed to avoid any reliance on C++/`llama.cpp` binaries for model preparation, instead leveraging Rust CLI utilities (e.g. `rust-gguf-tools`) for quantization and file packaging. The process is composed of distinct stages, each with clear responsibilities and interfaces:

- **Stage 1 – Training/Fine-tuning:** Use Hugging Face Transformers (with techniques like **LoRA/QLoRA**) to fine-tune the model on your dataset, while minimizing GPU memory usage. This produces fine-tuned weight updates or adapter files.
- **Stage 2 – Artifact Export:** Merge any LoRA adapters into the base model to obtain final fine-tuned weights. Save the model to disk in a framework-agnostic format (preferably **Safetensors** for safety and speed) along with its config and tokenizer files.
- **Stage 3 – Quantization (Rust CLI):** Use a Rust-based quantizer tool to convert the high-precision weights (float32/16 or bfloat16) into lower-bit **quantized** weights (e.g. 4-bit or 5-bit) compatible with GGUF. The output is a quantized model checkpoint ready for packaging.
- **Stage 4 – GGUF Packaging (Rust CLI):** Use a Rust-based GGUF writer tool to combine the quantized weights, model architecture metadata, and tokenizer data into a single **`.gguf`** file. This step populates all required GGUF metadata fields (model parameters, quantization info, etc.) and produces the final model file for inference.
- **Stage 5 – Integration & Automation:** (Optional) Integrate these steps with automation tools or **agent** frameworks (e.g. a `Codex` AI coding agent) to orchestrate the pipeline end-to-end. The Rust CLI tools are designed to be scriptable and cross-platform, facilitating easy automation.
- **Stage 6 – Inference & Deployment:** The resulting GGUF model can be loaded into compatible inference engines such as **Ollama**, **llama.cpp**, or other GGUF-enabled runtimes. The model is now optimized for efficient local inference (with significantly reduced memory footprint due to quantization) without further conversion.

Each stage is detailed below, with expected directory structures, configuration files, and input/output formats. Command examples and best practices are provided to ensure a smooth fine-tuning-to-GGUF workflow.

## Workspace Structure and File Organization

Organizing your workspace by stages will make the process more manageable. A recommended directory structure is as follows:

```text
fine-tune-workspace/
├── base_model/              # Original base model files (from HF or elsewhere)
│   ├── config.json          # Model architecture configuration
│   ├── tokenizer.json       # Tokenizer (or vocab files, merges, etc.)
│   └── model.safetensors    # (Optional) base model weights in safetensors (FP16/FP32)
├── data/                    # Training dataset & related scripts
├── finetune_output/         # Outputs from Stage 1 (fine-tuning)
│   ├── adapter_model.bin    # LoRA adapter weights (if using PEFT/QLoRA)
│   ├── pytorch_model.bin    # Full model weights (if fine-tuned fully, or after merging)
│   └── ...                  # Training logs, etc.
├── exported_model/          # Stage 2: exported full model after merging, in safetensors
│   ├── model.safetensors    # Merged fine-tuned weights (FP16/BF16, safe format)
│   ├── config.json          # Copy of model config (architecture hyperparameters)
│   └── tokenizer.model      # Tokenizer file (e.g. SentencePiece model or tokenizer.json)
├── quantized/               # Stage 3: quantizer output (quantized weights)
│   └── model.Q4_0.safetensors   # Example quantized weights (4-bit, in safetensors or raw)
├── gguf/                    # Stage 4: final packaged model
│   └── model.Q4_0.gguf          # Final GGUF file ready for inference
└── scripts/                 # (Optional) Automation scripts or agent recipes
    └── run_pipeline.sh          # Example script to run all stages
```

**Configuration files:** Ensure the **`config.json`** (with model architecture details like hidden size, layer count, etc.) and the **tokenizer files** (e.g. `tokenizer.json` or `tokenizer.model` plus any vocab/merges if applicable) are accessible. These will be needed in Stages 3 and 4 to correctly quantize and package the model. The pipeline will attempt to infer these if not explicitly provided (for example, by looking in the same directory as the model weights).

**Input/Output formats:** We use **Safetensors** for intermediate weight files (both float and quantized) because it’s a safe, efficient binary format that Rust can read easily (via the `safetensors` crate). The final output is a single **GGUF** file, which includes both the weights and all metadata needed for inference. At each stage, verify that the expected files are produced before moving on.

## Stage 1: Fine-Tuning with HuggingFace (QLoRA)

**Goal:** Fine-tune the base model on your dataset using Hugging Face Transformers, while employing memory-efficient methods (like **LoRA** or **QLoRA**) to reduce VRAM requirements. In this stage, you **do not** quantize the model; training typically uses 16-bit or 32-bit weights for stability.

**Procedure:** Use the Hugging Face Trainer or PEFT (Parameter-Efficient Fine-Tuning) with LoRA. For example, you might load the model with `load_in_4bit=True` and attach LoRA adapters via PEFT for QLoRA fine-tuning. This approach allows fine-tuning LLaMA-3 or similar 70B+ models on a single GPU by keeping weights in 4-bit during training, while accumulating updates in higher precision (NF4 + bfloat16).

**LoRA/QLoRA Setup:** Configure a LoRA adapter with the desired rank and alpha, and prepare the model for k-bit training (using utilities like `prepare_model_for_kbit_training` if using bitsandbytes 4-bit quantization). During training, gradients update the low-rank adapter matrices. The base model’s original weights remain unchanged on disk.

**Training Output:** After fine-tuning, you will have one of two outcomes:

- If you performed a standard fine-tune (full model update, no LoRA), the output directory (e.g. `finetune_output/`) will contain a new model checkpoint (commonly `pytorch_model.bin` or `.safetensors`) with the fully fine-tuned weights.
- If you used LoRA/QLoRA, the output directory will contain the small LoRA adapter weights (e.g. `adapter_model.bin/safetensors`) and possibly a final merged checkpoint if you explicitly merged during training. The base model weights are typically not saved again (since they were loaded from `base_model/`).

**Merging LoRA Weights:** In a LoRA scenario, the next step is to merge the learned LoRA deltas into the base model to produce a full set of fine-tuned weights. This can be done using the PEFT library’s utilities. For example, after training you can load the base model and apply `PeftModel.from_pretrained(...)` to load the LoRA, then call `model.merge_and_unload()` to merge the adapter into the model’s weights. This gives you a **merged model** in memory which represents the fully fine-tuned model in full precision. Save this merged model for the export stage.

**Tip:** It’s recommended to save the merged model in **Safetensors** format. You can use `model.save_pretrained('exported_model/', safe_serialization=True)` in Transformers, which will save `model.safetensors` along with the config and tokenizer. This ensures a safe, direct hand-off to the Rust quantization tool.

## Stage 2: Exporting the Fine-Tuned Model

**Goal:** Prepare the fine-tuned model artifacts for quantization by exporting them to a standardized format and gathering necessary files. At the end of this stage, you should have:

- A safetensors file of the **full fine-tuned weights** (merged if LoRA was used).
- The model’s `config.json` (architecture hyperparameters).
- The tokenizer files needed for inference.

**Export process:**

1. **Merge (if needed):** If using LoRA/QLoRA, perform the merge as described above and obtain the full weights in memory. If you fine-tuned the entire model directly, you already have full weights.
2. **Save weights to safetensors:** Use Hugging Face’s save functionality with `safe_serialization=True` to write out `model.safetensors`. This avoids potential pickle security issues and is faster for our Rust tooling to load.
3. **Copy config:** Ensure the `config.json` for the model (usually included with the base model or output by Trainer) is copied to the same `exported_model/` directory. This file contains important metadata like number of layers, hidden size, vocabulary size, etc., which will be needed for packaging.
4. **Copy tokenizer:** Include the tokenizer files. Many models (like LLaMA) have a `tokenizer.model` (SentencePiece model) or `tokenizer.json`, plus associated files (e.g. `vocab.txt`, `merges.txt` for BPE-based tokenizers). These need to be available so the GGUF writer can embed the vocabulary. Place them in `exported_model/` or note their path for the packaging step.

Your `exported_model/` directory should now look like:

```text
exported_model/
├── model.safetensors    # Fine-tuned weights (float32/16 or bf16)
├── config.json          # Model architecture config
└── tokenizer.model      # Tokenizer (or tokenizer.json / vocab files)
```

_(The tokenizer filename will vary by model; ensure you have whatever the model uses.)_

**Input/output formats:** At this point, everything is still in standard Hugging Face format (except using safetensors instead of PyTorch bin). The model weights are high precision (e.g. FP16 or BF16) and unquantized. This is the last chance to verify the model’s performance in full precision if desired (for example, you could do a quick inference or evaluate perplexity before quantization).

## Stage 3: Quantization with Rust CLI Tool

**Goal:** Convert the full-precision model weights into a quantized form to drastically reduce model size and inference memory, using a **Rust-based quantizer CLI** (let’s call it `quantize-rs`). This tool replaces the functionality of `llama.cpp`’s `quantize` C++ utility with a more portable Rust implementation.

**How it works:** The quantizer will **load the safetensors model** (using a Rust safetensors library) and systematically convert each weight matrix to a lower-bit representation. We support multiple quantization schemes:

- **4-bit integer (Q4)** and **5-bit integer (Q5)** in various flavors (e.g. `Q4_0`, `Q4_K`, `Q5_0`, `Q5_1`, `Q5_K`).
- **8-bit (Q8_0)** for a higher-precision quantization option.
- (Future/advanced) Newer quantization methods like K-quants or grouped quantization can be added, but the pipeline defaults to the well-established methods.

By default, a practical choice is **Q4_0 or Q5_1** quantization, which are legacy methods known to offer a good balance between size and accuracy. `Q4_0` will yield the smallest model (approx 4 bits per weight) at the cost of some quality loss, whereas `Q5_1` uses 5 bits and typically preserves more accuracy with a slight size increase. In practice, many find Q4_0 and Q5_1 quantizations of LLMs still perform well on language tasks, while being _2-4x smaller_ than the original model.

**Running the quantizer:** From the command line, you might invoke the tool like so (example):

```bash
# Example usage of the Rust quantize tool
quantize-rs \
  --model exported_model/model.safetensors \
  --config exported_model/config.json \
  --outfile quantized/model.Q4_0.safetensors \
  --format Q4_0
```

- `--model`: Path to the float32/16 safetensors file from Stage 2.
- `--config`: Path to the config.json (if not provided, the tool will attempt to locate it or infer architecture from the model file name or shapes).
- `--outfile`: Path for the output quantized weight file. Here we choose a name indicating the format (Q4_0).
- `--format`: The quantization format to apply (e.g. `Q4_0`, `Q5_1`, etc.). If not specified, the tool may default to a safe choice like Q4_0.

Under the hood, the quantizer will read each tensor (matrix) from the safetensors, perform the quantization (e.g. computing scale/zero-point or k-means cluster centroids depending on the scheme), and then write out the quantized tensors. The output could be another safetensors file containing quantized data (e.g. stored as int4/int5 values packed into bytes) or a temporary raw binary. In our design, we can output a safetensors for convenience, which the next stage will consume.

**Quantization specifics:** Not all parts of the model may be quantized. Typically, large weight matrices (attention weights, feed-forward weights) are quantized, while some small values (like layer norms or biases) might remain in higher precision if required. The Rust tool will handle these details similar to llama.cpp’s approach. It ensures the resulting weights are **“GGUF-compatible”** in terms of data types and tensor ordering (meaning any GGUF loader will understand them).

**Output:** After running `quantize-rs`, you should have a file like `model.Q4_0.safetensors` in the `quantized/` directory. This contains the quantized weights. At this point, the model is much smaller (for example, a 70B parameter model at Q4_0 might be \~40% of the original 16-bit size). However, it’s not yet in GGUF format, so we cannot run it in llama.cpp/Ollama until we package it.

**Recommended default:** If you are unsure which quantization to use, start with **Q4_0** for maximum RAM savings, or **Q5_1** if you want slightly better fidelity. Both are widely used and supported. You can quantize to multiple formats and later compare their performance (the pipeline can be rerun with a different `--format` and the GGUF packaging repeated).

**Optional – Evaluate quantization loss:** It’s a good practice (optional) to inspect the impact of quantization. For example, you could measure the model’s perplexity on a validation set before and after quantization. A small drop in accuracy (\~a few percent) is expected with 4-bit quantization, but if the drop is too large, you might choose a higher-precision format. This step is not required, but advanced users can use libraries like 🤗Transformers to load the quantized model (Transformers can load GGUF as unquantized for evaluation) or use an inference engine in CPU mode to compare outputs.

## Stage 4: Packaging into GGUF

**Goal:** Take the quantized weights and convert them into a self-contained **GGUF** file. This step is handled by a Rust CLI tool (let’s call it `gguf-writer`) that writes the GGUF file according to the specification, including all necessary metadata blocks.

**Inputs:** You will need:

- The quantized model checkpoint (from Stage 3, e.g. `model.Q5_1.safetensors`).
- The `config.json` (for architecture metadata, if not already included or accessible).
- The tokenizer data (tokenizer JSON or model, plus vocab/merges if applicable).
- (Optionally) Any additional metadata you want to include (e.g. model description, author name, license, etc., which can be added to the GGUF general metadata).

**Running the GGUF writer:** Example invocation:

```bash
gguf-writer \
  --model quantized/model.Q5_1.safetensors \
  --config exported_model/config.json \
  --tokenizer exported_model/tokenizer.model \
  --outfile gguf/model.Q5_1.gguf \
  --name "MyModel-FT-Q5_1"
```

This will:

- Read the quantized weights.
- Read the config to know the model architecture (e.g. number of layers, context length, hidden dim, etc.).
- Read the tokenizer (if it's a SentencePiece model like LLaMA uses, the tool will embed it; if it’s a Hugging Face tokenizer.json, it will parse the vocabulary and merges).
- Compose the GGUF file.

**Packaging details:** The writer will create the GGUF header and then insert **metadata key-value pairs** that describe the model. This includes:

- **General metadata:** e.g. model name, description, version, author, license, etc., if provided. You can pass these via CLI args or a template file, or accept defaults.
- **Architecture metadata:** keys specifying the model architecture. For example, for a LLaMA-based model it will set `llama.context_length`, `llama.embedding_length`, `llama.block_count` (layers), `llama.attention.head_count`, etc. based on the config. These values come directly from the config.json of the model.
- **Tokenizer metadata:** tokenizer type and vocabulary. It will include the tokenizer’s vocabulary size and possibly the actual tokens and scores if required by the spec (for SentencePiece, the file can be stored directly in a metadata field or as a binary blob). The tokenizer config (e.g. BOS/EOS tokens, special tokens) is also recorded.
- **Quantization metadata:** fields indicating that the weights are quantized and in which format. For example, the writer will set `general.file_type` to an enumeration value corresponding to the chosen quantization (e.g. `MOSTLY_Q5_1` for Q5_1 quantization). This tells loaders what precision to expect. It may also set a field like `general.quantized_by` to note that the model was quantized by a certain tool or person (this is optional metadata). Essentially, the GGUF will contain a flag that it’s a quantized model and which scheme was used, so inference software can handle it appropriately.
- **Tensor data:** finally, the actual quantized tensors are written into the GGUF file after the metadata. Each tensor is aligned to the required byte boundaries (typically 32-byte alignment by default). The writer ensures all alignment and format details follow the official GGUF spec, so that the file will load correctly.

**Output:** The output is a single file (e.g. `model.Q5_1.gguf` in the `gguf/` directory). This file is the portable, deployable model. You can rename it following GGUF naming conventions if desired (which encode base model, size, version, quantization in the filename), but this is optional.

After packaging, it’s wise to **validate the GGUF**. Our pipeline’s writer includes a verification step to ensure the file can be opened by the reference implementation (llama.cpp’s GGUF loader) without errors. If you have llama.cpp installed, you can also test loading the model (see next stage).

**Fallback behaviors:** The GGUF writer will try to fill in as much metadata as possible even if not all information is explicitly provided. For example, if you don’t pass a `--name`, it might use the base model name from the config or a generic name. If `--config` isn’t given, the tool might rely on the safetensors file metadata or structure (though providing the config is highly recommended for accuracy). If the tokenizer file isn’t provided, and the base model directory is known, it may try to locate a tokenizer there. The goal is to make it easy to use: minimal arguments for common cases (e.g., pointing to a Hugging Face model directory and output path could be enough for the tool to gather all needed files). However, explicit is better than implicit; for reproducibility, it’s best to specify the paths.

At this point, you have a complete GGUF model ready to use!

## Stage 5: Automation with Codex Agents (Optional)

**Goal:** Integrate the above stages into an automated workflow using agent-based tools or scripts. Because our tools are CLI-driven and require no human intervention once set up, they can be chained in build scripts or controlled by AI agents for continuous training pipelines.

For example, you could use an AI coding assistant (like OpenAI’s Codex or similar autonomous agents) to monitor a dataset repository and trigger this pipeline when new data is available. The agent can:

1. Launch the fine-tuning job (Stage 1) via a Python script or `accelerate launch` command.
2. Wait for completion, then call the `quantize-rs` CLI with appropriate arguments (Stage 3).
3. Then call `gguf-writer` to package the model (Stage 4).
4. Finally, run some tests on the new GGUF model (Stage 6) and deploy or notify you.

The motivation for building this in Rust was specifically to **enable easier automation and integration**. Unlike the original C++ tools, the Rust binaries are cross-platform and easier to manage in custom workflows (no complex compiler setups or platform-specific issues). This means a CI/CD pipeline or an agent running on Windows, Linux, or Mac can all execute the same quantizer and packager seamlessly.

**Codex integration example:** A Codex-powered agent script could be written to use the Python `subprocess` module to invoke these CLI commands as needed. Since the tools provide clear logging and exit codes, the agent can catch any errors (e.g., out-of-memory during quantization) and handle them – perhaps by automatically switching to a smaller quantization (if Q5_1 fails due to memory, try Q4_0), or splitting the model into shards for processing. The Rust tools are built to be modular and script-friendly, enabling such higher-level orchestration.

In summary, by designing the pipeline with distinct CLI tools, we’ve made it easy for automation agents to treat each stage as a “black box” tool – which can be composed and repeated as needed. This could be extended further with a proper API (the project could expose a Rust library or Python bindings via `pyo3` in the future), but even with the CLI, integration into agent workflows is straightforward.

## Stage 6: Inference and Compatibility with GGUF Runtimes

Now for the payoff – using the quantized GGUF model in real-world inference. The GGUF format is widely supported by lightweight inference engines:

- **Ollama:** A local LLM runtime that uses GGUF models. You can import your new `.gguf` model into Ollama and run it. For example, create an Ollama **Modelfile** that points to your GGUF and use `ollama create` to register it. Once added, you can prompt the model with `ollama run your-model-name -p "Your prompt here"`. The quantization will allow even large models to run on modest hardware.
- **llama.cpp:** The C/C++ LLM engine can directly load GGUF models. Simply pass the `.gguf` file to its CLI (e.g., `./main -m ./gguf/model.Q5_1.gguf -p "Hello AI"`). Llama.cpp will recognize the quantization type and use the appropriate dequantization kernels for inference. Because we validated the file, it should load without issues.
- **Other frameworks:** The GGUF spec is designed to be implementation-agnostic. There are Rust inference libraries (like `rustformers/llm` or the `llama-rs` project) that can load GGUF, as well as Python wrappers. Tools like GPT4All, text-generation-webui, and others are quickly adopting GGUF. You should be able to use the model anywhere that advertises GGUF support.

**Compatibility considerations:** We emphasized setting all the correct metadata in Stage 4 so that any standard GGUF loader will know how to handle the model. For example, the **tokenizer is embedded** – this means you don’t need to supply a separate tokenizer file at inference time; the `.gguf` includes it. The **quantization information is present** – loaders will know it’s a 4-bit or 5-bit model and allocate data structures accordingly. Model architecture keys (like context length, layer count) are also in the file, so the inference engine does not need a separate config.json; it’s all self-contained.

**Testing the model:** It’s a good idea to do a quick sanity check generation with the new model. For instance, you might prompt the model with a known input in Ollama or llama.cpp and see that you get reasonable output (and that it’s not outputting gibberish – a sign something might be off in the quantization if it were). Usually, if the metadata is wrong (say, you mismatched the vocab size), the model might produce incorrect tokens or error out. If you encounter issues, use a GGUF inspector tool (like `gguf-utils show` from the Rust tools or llama.cpp’s `convert-llama-gguf` with a dry-run mode) to dump the metadata and verify it matches your expectations (you can compare it with the original Hugging Face config). This is effectively **metadata diffing** – ensuring the GGUF’s metadata (e.g. hidden size, layer count, special tokens) matches the original model. Discrepancies here could explain any anomalies.

If everything looks good, congratulations – you now have a fine-tuned, quantized model running efficiently!

## Tips, Defaults, and Fallback Behaviors

To wrap up, here are some additional tips and default behaviors in the pipeline:

- **Memory considerations:** Quantization (Stage 3) can be memory-intensive for large models since it needs to load the full model and allocate buffers for quantization. Ensure you have enough RAM. If not, consider quantizing one layer at a time or using a smaller dtype (some advanced quantizers stream from disk). Our Rust tool could be extended with a streaming option if needed (future work).
- **Default quantization format:** If you don’t specify `--format` for quantization, the tool defaults to **Q4_0** (as it’s widely compatible and smallest). You can override to Q5 or others for better accuracy. Similarly, the default output file can be named automatically based on format (e.g. adding `-Q4_0` suffix) if you don’t provide `--outfile`.
- **Config inference:** The quantizer and writer will try to infer model architecture if `config.json` isn’t given. This might involve reading the safetensors metadata or using known defaults for certain model families. For example, if the model file is named `Llama-3b.safetensors`, the tool might assume a LLaMA architecture and attempt to deduce layer count from tensor shapes. **However, it’s safest to provide the config.json.** If not, the pipeline will log a warning and do its best. In our design, if no config is found, the tool may proceed by using the tensor names/shapes to guess (e.g., counting layers by unique weight group patterns), but a mismatch could lead to incorrect metadata.
- **Tokenizer handling:** By default, the GGUF writer looks for a `tokenizer.model` or `tokenizer.json` in the same directory as the model. If your model uses SentencePiece (common for LLaMA/Gemma), ensure the `.model` file is there. If BPE (GPT2 style), ensure both `vocab.json` and `merges.txt` or the consolidated `tokenizer.json` is provided. The writer can often detect the tokenizer type from the config (e.g., LLaMA config might indicate SentencePiece). If not found, it will error out prompting you to specify it.
- **Metadata customization:** You can supply additional metadata like `--author "Your Name"` or `--license "MIT"` if you want to embed those. Otherwise, the fields may remain empty or carry over from the base model if known. It’s good to set `general.finetune` to something descriptive (like “Alpaca-style instruction tuning”) to document how the model was specialized.
- **Validation and debug:** Use the companion tool (if available) to print out the GGUF contents. For example, a command like `gguf-utils show gguf/model.Q5_1.gguf` could list all metadata keys and tensor shapes. This is very useful for debugging. You can diff this output between two GGUF files to see what changed (for instance, diff between a Q4_0 and Q5_1 version of the model to confirm only the `file_type` and tensor data differ, not the model architecture metadata).
- **Inference fallback:** If you find that a quantized model is not giving good results, you can always fall back to a higher precision (e.g., quantize to 8-bit or even use an F16 GGUF which is essentially unquantized in GGUF container). The pipeline supports creating an F16 GGUF as well, which would be almost the same size as original but packaged for llama.cpp (just use `--format F16` if implemented, or skip Stage 3 quantization and feed an F16 safetensors to the writer). This is a good baseline to test the pipeline end-to-end before trying aggressive quantization.

**Sources:** The design and recommendations above are informed by the llama.cpp GGUF specification, industry best practices for LoRA fine-tuning, and the growing ecosystem of GGUF quantization methods.
