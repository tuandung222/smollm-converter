# Cẩm Nang Kỹ Thuật: Nguyên Tắc & Phương Pháp Luận Port Mô Hình (Model Porting A ➔ B)

> **Tác giả:** Võ Phạm Tuấn Dũng (`tuandung222`)  
> **Dự án mẫu:** [SmolLM2-135M Cross-Framework Converter](https://github.com/tuandung222/smollm-converter) (PyTorch ➔ JAX, Llama.cpp, LiteRT)  
> **Mục tiêu:** Tài liệu giáo khoa nhập môn dành cho ML Engineers, AI Researchers muốn nắm vững bản chất cốt lõi của việc chuyển đổi và triển khai mô hình học sâu xuyên nền tảng (Cross-Runtime Deployment).

---

## 1. Bản Chất Của "Model Porting" Là Gì?

Nhiều người mới bắt đầu thường nghĩ port mô hình là "dùng một tool tự động convert file đuôi `.pt` sang `.tflite` hay `.onnx`". Thực tế, nếu chỉ bấm lệnh convert bằng tool đóng hộp mà không hiểu cấu trúc bên trong, khi mô hình sinh ra kết quả sai (hoặc rác), bạn sẽ hoàn toàn bất lực trong việc debug.

Một mô hình AI khi đưa vào thực tế gồm **3 thành phần độc lập**:

```text
┌─────────────────────────────────────────────────────────────┐
│                       AI MODEL ARTIFACT                     │
├──────────────────────────────┬──────────────────────────────┤
│ 1. Kiến Trúc (Architecture) │ Code đồ thị tính toán (Ops)  │
│ 2. Trọng Số (Weights)        │ Ma trận tham số (Parameters) │
│ 3. Tiền/Hậu Xử Lý (Tokenizer)│ Text <-> Token IDs mapping   │
└──────────────────────────────┴──────────────────────────────┘
```

👉 **Bản chất của việc Port $A \rightarrow B$:**  
Tái hiện lại **chính xác 100% đồ thị toán học** trên runtime đích $B$, sau đó ánh xạ toàn bộ ma trận trọng số từ format $A$ sang định dạng bộ nhớ tương thích của $B$, sao cho với cùng một tensor đầu vào $X$, sai số ở đầu ra $Y$ tiệm cận 0 ($\approx 0.0001$ do sai số dấu phẩy động).

---

## 2. Năm Nguyên Tắc Cốt Tử Khi Port Mô Hình (The 5 Golden Rules)

### Nguyên Tắc 1: Bất Biến Toán Học (Mathematical Parity Invariant)
Toán học không thay đổi giữa các framework:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
$$\text{SwiGLU}(x) = \left(\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\right) W_{\text{down}}$$

Dù bạn viết bằng PyTorch, JAX, C++ (GGML) hay TFLite:
- Nếu $X_{A} = X_{B}$, thì $Y_{A}$ bắt buộc phải xấp xỉ $Y_{B}$.
- **Độ đo bắt buộc:**
  - **Cosine Similarity:** $\ge 0.99999$
  - **Max Absolute Difference:** $< 10^{-4}$ (đối với float32)
  - **Top-1 Token Prediction:** Trùng khớp 100%.

---

### Nguyên Tắc 2: Khác Biệt Bố Cục Bộ Nhớ & Thứ Tự Chiều (Memory Layout & Stride)

Đây là cạm bẫy số 1 khiến mô hình chạy được nhưng sinh ra kết quả sai lệch hoàn toàn.

#### A. Phép Nhân Ma Trận (Linear Projection)
- **PyTorch (`nn.Linear`)**: Định nghĩa $y = x W^T$. Trọng số lưu trữ dưới dạng:
  $$\text{shape} = [\text{out\_features}, \text{in\_features}]$$
- **JAX (`x @ W`)**: Phép nhân ma trận tiêu chuẩn đòi hỏi:
  $$\text{shape} = [\text{in\_features}, \text{out\_features}]$$
  $\rightarrow$ **Quy tắc:** Khi nạp từ PyTorch sang JAX, **bắt buộc phải Transpose ma trận 2D** ($W_{\text{JAX}} = W_{\text{PT}}^T$).

#### B. Quy Ước Thứ Tự Chiều Trong GGUF / GGML
- PyTorch/C lưu shape theo chuẩn C-contiguous (Row-major): `[batch, seq_len, hidden_dim]`.
- GGML lưu metadata shape theo quy ước **Fortran / Innermost-dimension first** (ngược lại với NumPy):
  - Ma trận embedding PyTorch `[49152, 576]` $\rightarrow$ GGUF Tensor Dimension lưu là `[576, 49152]`.

---

### Nguyên Tắc 3: Bẫy Biến Đổi Toán Tử (Operator Nuances — Case Study: RoPE)

> [!CAUTION]
> **Bài học xương máu:** Cùng một công thức Rotary Positional Embedding (RoPE), nhưng cách nhóm các cặp tọa độ trong không gian 2D giữa Hugging Face và GGML/llama.cpp là khác nhau!

- **Hugging Face LLaMA**: Nhóm nửa đầu và nửa sau vector:
  $$\text{rotate\_half}(x) = [-x_{d/2 \dots d}, x_{0 \dots d/2}]$$
  Nghĩa là phần tử $x_0$ ghép đôi với $x_{d/2}$, $x_1$ ghép đôi với $x_{d/2+1}$.
- **GGML (llama.cpp)**: Kernel C++ tối ưu SIMD xoay **các phần tử liền kề**:
  $$(x_0, x_1), (x_2, x_3), \dots$$

**Hậu quả nếu không biết điều này:**  
Nếu nạp thẳng trọng số $W_Q, W_K$ vào GGML mà không hoán vị, GGML sẽ xoay nhầm tần số của các chiều vector $\rightarrow$ Attention bị phá hủy hoàn toàn $\rightarrow$ **Sinh ra văn bản rác vô nghĩa** (*"Thetrendictheaviolate..."*).

**Cách xử lý (RoPE Permutation từ Scratch):**
```python
def permute_for_ggml(weights: np.ndarray, n_heads: int) -> np.ndarray:
    """Hoán vị ma trận W_Q hoặc W_K từ layout HF sang layout GGML."""
    out_dim, in_dim = weights.shape
    head_dim = out_dim // n_heads
    # 1. Tách head_dim thành 2 nửa (Hf convention)
    w = weights.reshape(n_heads, 2, head_dim // 2, in_dim)
    # 2. Hoán đổi trục để các cặp trở thành liền kề (GGML convention)
    w = np.swapaxes(w, 1, 2)
    return w.reshape(out_dim, in_dim)
```

---

### Nguyên Tắc 4: Triệt Tiêu Dynamic Abstractions (De-abstracting for Static Execution)

PyTorch rất linh hoạt ("Pythonic"): nó cho phép loop động, câu lệnh điều kiện `if/else` theo giá trị runtime, và các class phức tạp như `DynamicCache`.

Tuy nhiên, các runtime tối ưu cao cho Edge/Hardware (như **LiteRT/TFLite, TensorRT, XLA JIT**) đòi hỏi **đồ thị tĩnh (Static Computational Graph)**:
1. **Loại bỏ dynamic loop:** Không để Python control flow quyết định graph.
2. **Kích thước cố định (Fixed Dimensions):** Tránh dynamic slicing `x[:len]`. Thay bằng mask hoặc static sequence length.
3. **Thay thế in-place mutation:** Trong JAX, không thể gán `cache[i] = v`. Phải dùng functional updates (`jax.lax.dynamic_update_slice`).

---

### Nguyên Tắc 5: Cơ Chế Quản Lý Bộ Nhớ Đệm KV (KV-Cache Invariant)

Trong suy luận mô hình ngôn ngữ (LLM):
- **Naive Generation (Không có KV-cache):** Mỗi bước sinh token mới phải tính lại toàn bộ $1 \dots t-1$ token cũ $\rightarrow$ Độ phức tạp thời gian $\mathcal{O}(N^2)$.
- **Autoregressive with KV-cache:** Lưu trữ Key và Value của các bước trước $\rightarrow$ Mỗi bước chỉ tính $Q, K, V$ cho **duy nhất 1 token mới** $\rightarrow$ Độ phức tạp giảm còn $\mathcal{O}(1)$ FLOPs/step.

| Nền tảng | Cách tiếp cận KV-Cache |
| :--- | :--- |
| **PyTorch (Reference)** | `DynamicCache`: Tự append tensor động vào danh sách theo từng layer. |
| **Llama.cpp (GGUF)** | Tầng C++ tự quản lý Ring-Buffer KV-cache, hỗ trợ cả lượng tử hóa KV (Q8_0, Q4_0). |
| **JAX (From scratch)** | Pre-allocate mảng tĩnh `[Layers, Batch, MaxSeqLen, Heads, HeadDim]` và dùng `jax.lax.dynamic_update_slice`. Vì kích thước input vào JIT luôn là `[1, 1]`, **XLA chỉ compile đúng 1 lần duy nhất**! |
| **LiteRT (On-Device)** | Sử dụng Stateful Graph hoặc Signature 2 giai đoạn: `Prefill` (xử lý prompt) và `DecodeStep` (1 token + cache). |

---

## 3. Quy Trình Chuẩn 6 Bước Khi Port Bất Kỳ Mô Hình Nào

```text
   Bước 1: Khảo sát Ground Truth (PyTorch Reference)
                    │
                    ▼
   Bước 2: Viết Architecture thuần trên Target B
                    │
                    ▼
   Bước 3: Xây dựng Weight Transposition & Loader
                    │
                    ▼
   Bước 4: Kiểm chứng Kim Tự Tháp (Verification Pyramid)
                    │
                    ▼
   Bước 5: Hiện thực & Tối ưu hóa KV-Cache
                    │
                    ▼
   Bước 6: Kiểm thử suy luận thực tế (Real Text Inference)
```

### Bước 1: Khảo sát Ground Truth (Nền Tảng A)
- Đọc kỹ file `config.json` và code forward pass gốc.
- Ghi lại toàn bộ siêu tham số: `hidden_size`, `num_heads`, `num_kv_heads` (GQA hay MHA?), `intermediate_size`, `eps`, `rope_theta`.
- Lưu lại một input mẫu và output logits gốc làm mốc chuẩn (Ground Truth Tensor).

### Bước 2: Viết Architecture Thuần Trên Nền Tảng B
- Không dùng wrapper đóng gói sẵn. Tự viết từ atomic ops:
  - `RMSNorm`
  - `RoPE precompute & rotate`
  - `Multi-Head / Grouped-Query Attention`
  - `SwiGLU / MLP`
  - `TransformerBlock`

### Bước 3: Ánh Xạ Trọng Số (Weight Mapping)
- Lập bảng dictionary map tên:
  - `model.layers.0.self_attn.q_proj.weight` $\rightarrow$ `blk.0.attn_q.weight` (GGUF) hoặc `layer_0.q_proj` (JAX).
- Chú ý chuyển đổi định dạng (Transpose nếu cần, cast kiểu dữ liệu FP32/FP16/BF16).
- Kiểm tra `tie_word_embeddings`: Nếu `lm_head` không có trong file safetensors, phải tái sử dụng `embed_tokens`.

### Bước 4: Kiểm Chứng Kim Tự Tháp (The Verification Pyramid)
Đừng bao giờ đợi ghép xong cả 30 layer mới chạy thử. Hãy kiểm tra theo từng bậc:
1. **Bậc 1:** Test riêng hàm RMSNorm (cho cùng input vector, so sánh output).
2. **Bậc 2:** Test riêng hàm RoPE.
3. **Bậc 3:** Test riêng 1 khối DecoderLayer.
4. **Bậc 4:** Test toàn bộ mạng trên 1 sequence cố định $\rightarrow$ Tính Max Diff và Cosine Similarity của logits cuối cùng.

### Bước 5: Hiện Thực KV-Cache
- Đảm bảo logic decode nhận `(token_t, cache_{t-1}) \rightarrow (\text{logits}_t, cache_t)`.
- Xác nhận rằng logits sinh ra từ cơ chế có KV-cache phải **trùng khớp từng bit** với khi chạy forward full sequence không cache.

### Bước 6: Kiểm Thử Suy Luận Thực Tế (End-to-End Inference)
> **Nguyên tắc vàng:** *"Checkpoint không inference được văn bản có nghĩa là checkpoint rác."*
- Nạp checkpoint vào runtime thực tế (`llama-completion`, `jax.jit`, `LiteRT Interpreter`).
- Prompt thử các câu thông dụng (ví dụ: *"The capital of France is"*).
- Mô hình phải sinh ra câu văn mạch lạc, đúng ngữ nghĩa và đo lường throughput (tokens/s).

---

## 4. Bảng Tra Cứu Sự Khác Biệt Giữa 4 Hệ Sinh Thái

| Đặc điểm | PyTorch (Hugging Face) | JAX (Flax / Pure) | Llama.cpp (GGUF) | Google LiteRT (TFLite) |
| :--- | :--- | :--- | :--- | :--- |
| **Mục tiêu chính** | Research & Prototyping | Research quy mô lớn, TPU/GPU XLA | Suy luận siêu nhẹ trên CPU/Metal/CUDA | Suy luận On-Device (Mobile, Edge NPU) |
| **Mô hình thực thi** | Dynamic / Eager | Pure Functional JIT (XLA) | C/C++ compiled native engine | Static FlatBuffer Interpreter |
| **Chiều Ma trận Linear** | $[D_{out}, D_{in}]$ ($y = x W^T$) | $[D_{in}, D_{out}]$ ($y = x W$) | Fortran $[D_{in}, D_{out}]$ trong header | Tùy biến theo kernel hạ tầng |
| **RoPE Convention** | Split nửa: `[-x2, x1]` | Tùy biến (thường theo HF) | Nhóm cặp liền kề `(x0, x1)` | Theo graph export |
| **Lưu Trữ KV-Cache** | Danh sách Dynamic Tensor | Mảng tĩnh + `dynamic_update_slice` | Paged Ring-Buffer C++ | Static Tensor / Stateful Node |
| **Định dạng file** | `.safetensors`, `.bin` | `.npz`, Orbax, safetensors | `.gguf` v3 | `.tflite` (FlatBuffer) |

---

## 5. Tổng Kết

Porting mô hình không phải là công việc sao chép code máy móc. Đó là quá trình **kỹ thuật số học chuẩn xác (Numerical Engineering)** đòi hỏi người kỹ sư phải thấu suốt từ phương trình toán học trên giấy, cấu trúc bố cục bộ nhớ trên RAM/VRAM, cho tới đặc thù tập lệnh SIMD/XLA của phần cứng thực thi.

Khi bạn nắm vững các nguyên tắc trên, bạn có thể tự tin chuyển đổi bất kỳ mô hình Transformer nào (LLaMA, Mistral, Gemma, Qwen, DeepSeek) sang bất kỳ ngôn ngữ hay phần cứng nào mà không bao giờ gặp tình trạng "mô hình xuất ra file nhưng chạy ra rác".
