# SmolLM2 Cross-Framework Converter (Built Completely From Scratch)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Frameworks](https://img.shields.io/badge/Targets-JAX%20|%20Llama.cpp%20|%20LiteRT%20|%20PyTorch-orange.svg)](#features)
[![Parity](https://img.shields.io/badge/Numerical%20Parity-Cosine%20Sim%201.00000000-success.svg)](#benchmarks--parity-results)

An educational, battle-tested reference implementation for porting **SmolLM2-135M** (LLaMA architecture: RoPE, SwiGLU, RMSNorm, GQA) into three distinct runtime ecosystems:

1. **JAX**: Pure functional Transformer with static KV-Cache (`jax.lax.dynamic_update_slice`), `@jax.jit` compiled, zero black-box dependencies.
2. **Llama.cpp (GGUF v3)**: Standalone binary serializer and $Q8\_0$ block quantizer implemented from scratch in pure Python/NumPy (no `llama.cpp` scripts or internal classes), solving the infamous HuggingFace-to-GGML RoPE permutation problem.
3. **Google LiteRT (formerly TFLite)**: Edge-optimized graph architecture removing dynamic control flow, exportable to FlatBuffer `.tflite`, running on `ai_edge_litert.interpreter.Interpreter` with XNNPACK.
4. **PyTorch Reference**: Ground-truth implementation self-contained directly within the repo (`torch_impl/`).

---

## 🎯 Engineering Philosophy: *"A Checkpoint That Cannot Infer Is Garbage"*

Model conversion is **not** about dumping binary weights to satisfy file extension parsers. A conversion pipeline is only complete when the exported artifact performs **end-to-end, multi-token autoregressive text generation** in real target runtimes with zero semantic degradation.

---

## 📊 Benchmarks & Parity Results

All models were evaluated against the Hugging Face PyTorch ground truth using the identical prompt:

| Runtime | Format | Artifact Size | Cosine Similarity | Max Abs Error | Inference Speed | Generation Quality |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch (Ref)** | In-Repo (`torch_impl/`) | ~270 MB | `1.00000000` (Baseline) | $0.0$ | **64.9 t/s** | Coherent, fluent |
| **Pure C (Zero-dep)**| `smollm2_135m_raw.bin` | 621 MB | `1.00000000` | $< 1.0 \times 10^{-4}$ | **8.5 t/s** (Single-thread C) | Coherent, fluent |
| **Pure JAX** | `smollm2_135m_jax.npz` | 291 MB | `1.00000000` | $1.299 \times 10^{-4}$ | **38.5 t/s** (~20ms/tok) | Coherent, fluent |
| **Llama.cpp** | `smollm2_135m_f16.gguf`<br>`smollm2_135m_q8_0.gguf` | 313 MB (F16)<br>**167 MB** (Q8_0) | `1.00000000` | $< 1.5 \times 10^{-4}$ | **178.3 t/s** (CPU Metal) | Coherent, fluent |
| **Google LiteRT** | `smollm2_135m.tflite` | 514 MB | `1.00000000` | $1.301 \times 10^{-4}$ | Real-time on-device | Coherent, fluent |

---

## 📂 Repository Structure

```text
smollm_converter/
├── models/SmolLM2-135M/        # Local weights (safetensors, config.json, tokenizer.json)
├── torch_impl/                 # Ground-truth PyTorch reference directly in repo
│   ├── configuration_llama.py  # LlamaConfig definition
│   ├── modeling_llama.py       # Full LlamaForCausalLM architecture
│   └── infer_torch.py          # Standalone PyTorch text generator
├── c_scratch/                  # Pure C Inference Engine (Zero libraries, pure C99)
│   ├── smollm.c                # Full C implementation of Llama (RMSNorm, RoPE, GQA, SwiGLU, KV-Cache)
│   ├── export_to_c.py          # Exports PyTorch weights to raw binary format
│   └── infer_c.py              # CLI text generation wrapper for C binary
├── gguf_scratch/               # Target 1: Llama.cpp GGUF from scratch
│   ├── gguf_writer.py          # Standalone GGUF v3 Binary Writer & Q8_0 Quantizer
│   ├── convert_smollm_gguf.py  # Weight mapping, RoPE permutation, & metadata builder
│   ├── verify_gguf.py          # Validates binary compliance with GGUF v3 specification
│   └── infer_gguf.py           # Real-time text generation via llama-completion CLI
├── jax_scratch/                # Target 2: Pure JAX Transformer from scratch
│   ├── llama_jax.py            # RMSNorm, RoPE, GQA, SwiGLU, and Transformer blocks
│   ├── llama_jax_kv.py         # Static functional KV-Cache (jax.lax.dynamic_update_slice)
│   ├── convert_smollm_jax.py   # Maps safetensors to JAX PyTree & exports .npz
│   ├── verify_jax.py           # Numerical parity verification (MSE, Cosine Sim vs PyTorch)
│   ├── infer_jax.py            # Pure JAX autoregressive token generator
│   └── infer_jax_kv.py         # O(1) latency text generator with static KV-Cache
├── litert_scratch/             # Target 3: Google LiteRT (TFLite) from scratch
│   ├── edge_model.py           # Clean PyTorch Edge model without dynamic abstractions
│   ├── convert_smollm_litert.py# Compiles & exports to .tflite via litert_torch
│   ├── verify_litert.py        # Numerical parity verification on LiteRT Interpreter
│   └── infer_litert.py         # On-device autoregressive generation using LiteRT FlatBuffer
├── download_model.py           # Downloads SmolLM2-135M weights from HuggingFace
├── run_all.py                  # Master CLI pipeline coordinating all targets
├── DOCS_PORTING_GUIDE.md       # Comprehensive educational guide to model porting
└── README.md
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/tuandung222/smollm-converter.git
cd smollm-converter

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers accelerate safetensors jax jaxlib flax gguf litert-torch ai-edge-litert numpy
```

### 2. Download Model Weights
```bash
python download_model.py
```

### 3. Run and Verify Each Target

#### Target 1: Llama.cpp (GGUF v3)
```bash
# Export F16 and Q8_0 GGUF binaries from scratch
python gguf_scratch/convert_smollm_gguf.py models/SmolLM2-135M f16
python gguf_scratch/convert_smollm_gguf.py models/SmolLM2-135M q8_0

# Verify GGUF header & tensor metadata compliance
python gguf_scratch/verify_gguf.py smollm2_135m_f16.gguf smollm2_135m_q8_0.gguf

# Run real-time generation benchmark (178 tokens/sec)
python gguf_scratch/infer_gguf.py "Artificial Intelligence is transforming" smollm2_135m_q8_0.gguf
```

#### Target 2: Pure JAX with Static KV-Cache
```bash
# Export parameters to compressed JAX PyTree .npz
python jax_scratch/convert_smollm_jax.py models/SmolLM2-135M smollm2_135m_jax.npz

# Run 1-to-1 numerical parity verification against PyTorch
python jax_scratch/verify_jax.py models/SmolLM2-135M

# Run autoregressive generation with static KV-Cache (flat ~20ms/token latency)
python jax_scratch/infer_jax_kv.py "The theory of relativity explains that"
```

#### Target 3: Google LiteRT (.tflite)
```bash
# Export to LiteRT FlatBuffer (.tflite)
python litert_scratch/convert_smollm_litert.py models/SmolLM2-135M smollm2_135m.tflite 16

# Verify on-device output parity
python litert_scratch/verify_litert.py smollm2_135m.tflite models/SmolLM2-135M 16

# Run token-by-token on-device generation with LiteRT runtime
python litert_scratch/infer_litert.py "The future of science is"
```

#### Target 4: PyTorch In-Repo Reference
```bash
python torch_impl/infer_torch.py "Artificial Intelligence is transforming the world because"
```

#### Target 5: Pure C Inference Engine (Zero Frameworks / Libraries)
```bash
# Export weights to raw binary format
python c_scratch/export_to_c.py

# Compile pure C engine with Apple clang
clang -O3 -o c_scratch/smollm c_scratch/smollm.c -lm

# Run text generation directly in terminal
python c_scratch/infer_c.py "The theory of relativity explains that" 25
```

#### Run All Targets Automatically
```bash
python run_all.py --target all
```

---

## 🧠 Key Technical Takeaways

1. **The RoPE Permutation Trap**:
   - Hugging Face pairs RoPE frequencies by splitting the dimension into two halves: `[-x2, x1]`.
   - GGML/llama.cpp's SIMD kernel rotates consecutive elements: `(x[0], x[1]), (x[2], x[3])`.
   - Exporting without permuting $W_Q$ and $W_K$ matrices results in gibberish text output. This repo implements the exact permutation algorithm from scratch in NumPy.

2. **Zero-Recompilation JAX KV-Cache**:
   - Autoregressive generation in JAX often suffers from XLA re-compilation when sequence length changes dynamically.
   - By pre-allocating a static buffer `[30, 1, max_len, 3, 64]` and indexing via `jax.lax.dynamic_update_slice` and `dynamic_slice`, input shapes remain strictly `(1, 1)`, achieving **single-compilation, constant $O(1)$ latency (~20ms/token)**.

3. **LiteRT De-Abstraction**:
   - Removing Hugging Face's dynamic caching classes, kwargs unpacking, and conditional slicing allows direct lowering through Torch FX, MLIR, and FlatBuffer bytecode.

For an in-depth pedagogical breakdown of model porting principles, read **[DOCS_PORTING_GUIDE.md](file:///Users/admin/Downloads/smollm_converter/DOCS_PORTING_GUIDE.md)**.

---

## 👤 Author & Contributor
- **Author:** Võ Phạm Tuấn Dũng ([@tuandung222](https://github.com/tuandung222))
- **Email:** `tuandung12092002@gmail.com` / `75377334+tuandung222@users.noreply.github.com`
- **Institution:** Ho Chi Minh City University of Technology (HCMUT - VNU-HCM)
