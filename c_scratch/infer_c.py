"""
Python wrapper for Pure C SmolLM2 Inference Engine.
Tokenizes arbitrary text prompts and delegates execution to our compiled smollm C binary.
"""
from __future__ import annotations
import os
import sys
import subprocess
from transformers import AutoTokenizer

C_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_PATH = os.path.join(C_DIR, "smollm")
MODEL_BIN = os.path.join(C_DIR, "smollm2_135m_raw.bin")
TOKENIZER_BIN = os.path.join(C_DIR, "tokenizer.bin")


def generate_c(
    prompt: str = "The theory of relativity explains that",
    n_predict: int = 25,
    model_dir: str = "models/SmolLM2-135M"
):
    print("=" * 65)
    print("       PURE C ENGINE AUTOREGRESSIVE GENERATION RUNNER            ")
    print("=" * 65)

    if not os.path.exists(BINARY_PATH):
        print(f"[*] Compiling {BINARY_PATH} with clang...")
        subprocess.run(["clang", "-O3", "-o", BINARY_PATH, os.path.join(C_DIR, "smollm.c"), "-lm"], check=True)

    if not os.path.exists(MODEL_BIN):
        print(f"[*] Exporting raw binary weights...")
        from export_to_c import export_model_to_raw_bin
        export_model_to_raw_bin(model_dir, MODEL_BIN, TOKENIZER_BIN)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokens = tokenizer.encode(prompt)
    print(f"[*] Prompt: '{prompt}' ({len(tokens)} tokens)")

    # Execute C binary with token IDs passed via CLI
    cmd = [BINARY_PATH, MODEL_BIN, TOKENIZER_BIN, str(n_predict)] + [str(t) for t in tokens]
    subprocess.run(cmd)


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The theory of relativity explains that"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    generate_c(prompt, n)
