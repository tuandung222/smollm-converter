"""
Text generation runner for GGUF models using llama-completion.
"""
import os
import shutil
import sys
import subprocess


def generate_gguf(
    prompt: str = "Artificial Intelligence is transforming",
    gguf_path: str = "smollm2_135m_q8_0.gguf",
    n_predict: int = 25
):
    print("=" * 65)
    print("            Llama.cpp (GGUF) TEXT GENERATION TEST                ")
    print("=" * 65)
    print(f"[*] Prompt: '{prompt}'")
    print(f"[*] Model : {gguf_path}")

    bin_path = shutil.which("llama-completion") or "/opt/homebrew/bin/llama-completion"
    if not os.path.exists(bin_path):
        print(f"[!] Warning: llama-completion binary not found at {bin_path}")
        return

    cmd = [
        bin_path,
        "-m", gguf_path,
        "-p", prompt,
        "-n", str(n_predict),
        "--temp", "0.0"
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The gravity of the earth is"
    model = sys.argv[2] if len(sys.argv) > 2 else "smollm2_135m_q8_0.gguf"
    generate_gguf(prompt, model)
