from ctransformers import AutoModelForCausalLM
from difflib import SequenceMatcher
from time import perf_counter
import os

# === Configuration ===
BASE_MODEL_PATH = "exported_model/model.gguf"
QUANT_MODEL_PATH = "quantized/model.Q4_0.gguf"
PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "What are the benefits of using Rust for system programming?",
    "Summarize the causes of World War I.",
    "Write a short poem about the stars.",
    "Translate 'I love programming' to French.",
]
MAX_TOKENS = 128


# === Utility Functions ===
def similarity_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()


def timed_response(model, prompt: str):
    start = perf_counter()
    output = model(prompt, max_new_tokens=MAX_TOKENS)
    end = perf_counter()
    return output.strip(), end - start


# === Load Models ===
print("🔄 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, model_type="llama", gpu_layers=0)

print("🔄 Loading quantized model...")
quant_model = AutoModelForCausalLM.from_pretrained(QUANT_MODEL_PATH, model_type="llama")

print("\n🧪 Evaluating...")

# === Evaluation Loop ===
results = []
for i, prompt in enumerate(PROMPTS, 1):
    print(f"\n🔹 Prompt {i}: {prompt}")

    base_output, base_time = timed_response(base_model, prompt)
    quant_output, quant_time = timed_response(quant_model, prompt)

    sim = similarity_score(base_output, quant_output)

    print(f"  🧠 Base:   {base_output[:100]}... ({base_time:.2f}s)")
    print(f"  🧠 Quant:  {quant_output[:100]}... ({quant_time:.2f}s)")
    print(f"  🔍 Similarity: {sim:.3f} | Speedup: {base_time/quant_time:.2f}x")

    results.append({
        "prompt": prompt,
        "base_time": base_time,
        "quant_time": quant_time,
        "similarity": sim,
    })

# === Summary ===
avg_sim = sum(r["similarity"] for r in results) / len(results)
avg_speedup = sum(r["base_time"] / r["quant_time"] for r in results) / len(results)

print("\n✅ Summary:")
print(f"  🔁 Avg Similarity: {avg_sim:.3f}")
print(f"  ⚡ Avg Speedup:    {avg_speedup:.2f}x")
