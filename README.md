# SmolLM2 Cross-Framework Converter (From Scratch)

Dự án chuyển đổi mô hình **SmolLM2-135M** (kiến trúc LLaMA thu nhỏ: RoPE, SwiGLU, RMSNorm, GQA) sang 3 hệ sinh thái khác nhau hoàn toàn **được xây dựng từ con số 0 (from scratch)**:

1. **JAX Engine**: Tự hiện thực toàn bộ kiến trúc Transformer (RMSNorm, RoPE, GQA, SwiGLU) bằng pure JAX/NumPy, không phụ thuộc vào `transformers` modeling classes hay `FlaxAutoModel`. Đạt **100% numerical parity** (Cosine Similarity = `1.00000000`, sai số tuyệt đối tối đa `< 1.3e-4`).
2. **Llama.cpp (GGUF v3)**: Tự viết **Binary Serializer** và **Q8_0 Block Quantizer** từ scratch (không dùng script `convert_hf_to_gguf.py` hay class nội bộ của `llama.cpp`). Xuất ra định dạng `.gguf` chuẩn (F16 và Q8_0) tương thích tuyệt đối với mọi runtime `llama.cpp` / `llama-cli`.
3. **Google LiteRT (TFLite)**: Tự thiết kế kiến trúc Edge-friendly loại bỏ các dynamic control flow của Hugging Face, export sang FlatBuffer `.tflite` và đối chiếu suy luận trên runtime `ai_edge_litert.interpreter.Interpreter`.

---

## Cấu trúc thư mục

```text
smollm_converter/
├── models/                     # Trọng số SmolLM2-135M (safetensors, config, tokenizer)
│   └── SmolLM2-135M/
├── torch_impl/                 # Ground Truth PyTorch Implementation trực tiếp trong repo
│   ├── configuration_llama.py  # LlamaConfig reference
│   ├── modeling_llama.py       # LlamaForCausalLM reference
│   └── infer_torch.py          # Script chạy PyTorch inference trực tiếp từ repo
├── gguf_scratch/               # Nhánh 1: GGUF từ con số 0
│   ├── gguf_writer.py          # Bộ ghi Binary GGUF v3 & thuật toán Q8_0 Quantizer
│   ├── convert_smollm_gguf.py  # Map weights, hoán vị RoPE và xuất ra file .gguf
│   ├── verify_gguf.py          # Đọc lại và kiểm tra tính hợp lệ của binary format
│   └── infer_gguf.py           # Chạy inference trực tiếp trên llama.cpp runtime
├── jax_scratch/                # Nhánh 2: LLaMA JAX Engine từ con số 0
│   ├── llama_jax.py            # RMSNorm, RoPE, GQA, SwiGLU, Decoder stack pure JAX
│   ├── llama_jax_kv.py         # Cài đặt Static Functional KV-Cache bằng dynamic_slice
│   ├── convert_smollm_jax.py   # Map safetensors sang JAX PyTree & xuất file .npz
│   ├── verify_jax.py           # So sánh logits 1-1 với PyTorch và demo greedy decoding
│   ├── infer_jax.py            # Chạy inference autoregressive JAX
│   └── infer_jax_kv.py         # Chạy inference JAX có KV-Cache O(1) latency
├── litert_scratch/             # Nhánh 3: Google LiteRT từ con số 0
│   ├── edge_model.py           # Model PyTorch tinh giản tối ưu cho Edge NPU/CPU
│   ├── convert_smollm_litert.py# Export mô hình sang .tflite bằng LiteRT
│   ├── verify_litert.py        # Đối chiếu logits giữa LiteRT Runtime vs PyTorch
│   └── infer_litert.py         # Chạy inference autoregressive với LiteRT FlatBuffer
├── download_model.py           # Tự động tải weights từ HuggingFace Hub
├── run_all.py                  # CLI điều phối tổng thể
└── README.md
```

---

## Hướng dẫn cài đặt & Chạy

### 1. Kích hoạt môi trường ảo
```bash
source .venv/bin/activate
```

### 2. Tải trọng số SmolLM2-135M
```bash
python download_model.py
```

### 3. Nhánh 1: Chuyển đổi sang GGUF (Llama.cpp)
```bash
# Xuất bản F16 (~312 MB)
python gguf_scratch/convert_smollm_gguf.py models/SmolLM2-135M f16

# Xuất bản Q8_0 Quantized (~166 MB) bằng thuật toán Quantizer tự viết
python gguf_scratch/convert_smollm_gguf.py models/SmolLM2-135M q8_0

# Kiểm tra tính chuẩn chỉ của file GGUF
python gguf_scratch/verify_gguf.py smollm2_135m_f16.gguf smollm2_135m_q8_0.gguf

# Chạy inference sinh văn bản thực tế với GGUF:
python gguf_scratch/infer_gguf.py "The gravity of the earth is" smollm2_135m_q8_0.gguf
```

### 4. Nhánh 2: Chuyển đổi và Chạy JAX Engine
```bash
# Xuất weights sang JAX PyTree / NPZ
python jax_scratch/convert_smollm_jax.py models/SmolLM2-135M smollm2_135m_jax.npz

# Kiểm tra đối chứng logits (numerical parity) vs PyTorch:
python jax_scratch/verify_jax.py models/SmolLM2-135M

# Chạy inference sinh văn bản thực tế bằng pure JAX:
python jax_scratch/infer_jax.py "Machine learning allows computers to"
```

### 5. Nhánh 3: Chuyển đổi sang LiteRT (.tflite)
```bash
# Export sang file FlatBuffer .tflite
python litert_scratch/convert_smollm_litert.py models/SmolLM2-135M smollm2_135m.tflite 16

# Kiểm tra đối chứng logits giữa LiteRT Runtime vs PyTorch:
python litert_scratch/verify_litert.py smollm2_135m.tflite models/SmolLM2-135M 16

# Chạy inference sinh văn bản thực tế bằng LiteRT FlatBuffer:
python litert_scratch/infer_litert.py "The future of science is"
```

### 6. Chạy toàn bộ pipeline tự động
```bash
python run_all.py --target all
```
