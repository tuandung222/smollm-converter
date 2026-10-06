"""
Master CLI Coordinator for SmolLM2 Cross-Framework Conversion Project.
Builds and converts SmolLM2 to:
  1. GGUF (llama.cpp) from scratch (F16 and Q8_0)
  2. JAX from scratch (Pure JAX PyTree + parity verification)
  3. LiteRT (.tflite) from scratch (Edge architecture + LiteRT runtime verification)
"""
from __future__ import annotations
import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def banner():
    print(r"""
========================================================================
   SmolLM2-135M Cross-Framework Converter (Built Completely From Scratch)
   [x] Target 1: GGUF (Llama.cpp) - Custom Binary Serializer & Q8_0 Quantizer
   [x] Target 2: JAX - Pure Functional Transformer Architecture (Zero HF/Flax wrappers)
   [x] Target 3: LiteRT (TFLite) - Clean Edge Architecture + On-device FlatBuffer
========================================================================
""")


def run_gguf(model_dir: str, out_type: str = "f16"):
    banner()
    print(f"\n>>> [1/3] RUNNING GGUF CONVERSION ({out_type.upper()})...")
    from gguf_scratch.convert_smollm_gguf import convert_smollm_to_gguf
    from gguf_scratch.verify_gguf import verify_gguf
    from gguf_scratch.infer_gguf import generate_gguf

    out_file = os.path.join(PROJECT_ROOT, f"smollm2_135m_{out_type}.gguf")
    t0 = time.time()
    convert_smollm_to_gguf(model_dir, out_file, out_type)
    t1 = time.time()
    print(f"[*] Conversion took: {t1 - t0:.2f} seconds.")
    print("\n>>> Verifying exported GGUF file...")
    verify_gguf(out_file)
    print("\n>>> Testing real text generation with llama.cpp binary...")
    generate_gguf("The gravity of the earth is", out_file, n_predict=25)


def run_jax(model_dir: str):
    banner()
    print("\n>>> [2/3] RUNNING JAX EXPORT & NUMERICAL PARITY VERIFICATION...")
    from jax_scratch.convert_smollm_jax import export_jax_npz
    from jax_scratch.verify_jax import verify_numerical_parity
    from jax_scratch.infer_jax import generate_jax

    out_npz = os.path.join(PROJECT_ROOT, "smollm2_135m_jax.npz")
    t0 = time.time()
    export_jax_npz(model_dir, out_npz)
    t1 = time.time()
    print(f"[*] JAX NPZ export took: {t1 - t0:.2f} seconds.")
    print("\n>>> Verifying JAX numerical parity vs PyTorch...")
    verify_numerical_parity(model_dir)
    print("\n>>> Testing real text generation with pure JAX engine...")
    generate_jax("Machine learning allows computers to", out_npz, model_dir, max_new_tokens=15)


def run_litert(model_dir: str, seq_len: int = 16):
    banner()
    print("\n>>> [3/3] RUNNING LiteRT (.tflite) EXPORT & RUNTIME VERIFICATION...")
    from litert_scratch.convert_smollm_litert import convert_smollm_to_litert
    from litert_scratch.verify_litert import verify_litert
    from litert_scratch.infer_litert import generate_litert

    out_tflite = os.path.join(PROJECT_ROOT, "smollm2_135m.tflite")
    t0 = time.time()
    convert_smollm_to_litert(model_dir, out_tflite, seq_len)
    t1 = time.time()
    print(f"[*] LiteRT conversion took: {t1 - t0:.2f} seconds.")
    print("\n>>> Verifying LiteRT runtime vs PyTorch...")
    verify_litert(out_tflite, model_dir, seq_len)
    print("\n>>> Testing real on-device text generation with LiteRT FlatBuffer...")
    generate_litert("The future of science is", out_tflite, model_dir, max_new_tokens=10, seq_len=seq_len)


def main():
    parser = argparse.ArgumentParser(description="SmolLM2 Cross-Framework Converter")
    parser.add_argument("--model-dir", type=str, default="models/SmolLM2-135M", help="Path to local SmolLM2 directory")
    parser.add_argument("--target", type=str, choices=["gguf", "jax", "litert", "all"], default="all", help="Target framework")
    parser.add_argument("--gguf-type", type=str, choices=["f16", "q8_0"], default="f16", help="GGUF quantization format")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length for LiteRT static shape")

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.model_dir, "model.safetensors")):
        print(f"[-] Model weights not found in {args.model_dir}. Downloading first...")
        from download_model import download_smollm2_135m
        download_smollm2_135m(args.model_dir)

    if args.target in ["gguf", "all"]:
        run_gguf(args.model_dir, args.gguf_type)

    if args.target in ["jax", "all"]:
        run_jax(args.model_dir)

    if args.target in ["litert", "all"]:
        run_litert(args.model_dir, args.seq_len)

    print("\n[SUCCESS] All requested target conversions completed successfully!")


if __name__ == "__main__":
    main()
