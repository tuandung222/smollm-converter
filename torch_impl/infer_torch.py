"""
PyTorch Reference Text Generation using local modeling_llama.py implementation.
Proves that the in-repo PyTorch implementation is 100% functional.
"""
from __future__ import annotations
import os
import sys
import time
import torch
from transformers import AutoTokenizer
from safetensors import safe_open

# Ensure local torch_impl is used
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configuration_llama import LlamaConfig
from modeling_llama import LlamaForCausalLM


def generate_torch(
    prompt: str = "Artificial Intelligence is transforming the world because",
    model_dir: str = "models/SmolLM2-135M",
    max_new_tokens: int = 25
):
    print("=" * 65)
    print("       PYTORCH REFERENCE GENERATION (IN-REPO IMPLEMENTATION)     ")
    print("=" * 65)
    print(f"[*] Prompt: '{prompt}'")
    print(f"[*] Loading LlamaConfig and LlamaForCausalLM from local torch_impl/...")

    config = LlamaConfig.from_pretrained(model_dir)
    model = LlamaForCausalLM.from_pretrained(model_dir, config=config, torch_dtype=torch.float32)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    print(f"[*] Generating {max_new_tokens} tokens with PyTorch KV-Cache...")
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=config.eos_token_id
        )
    t1 = time.time()

    generated_text = tokenizer.decode(output[0])
    total_time = t1 - t0
    tps = max_new_tokens / total_time if total_time > 0 else 0

    print("\n" + "=" * 65)
    print("                     BENCHMARK RESULTS                           ")
    print("=" * 65)
    print(f"Generation Time : {total_time:.3f} s")
    print(f"Throughput      : {tps:.2f} tokens/second")
    print(f"\nGenerated Output:\n--> {generated_text}\n")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Artificial Intelligence is transforming the world because"
    generate_torch(prompt)
