"""
Export weights and vocabulary from PyTorch model to raw binary format for pure C inference.
Zero dependencies on external runtimes.
"""
import os
import sys
import struct
import json
import numpy as np
from safetensors import safe_open


def export_model_to_raw_bin(
    model_dir: str = "models/SmolLM2-135M",
    output_bin: str = "c_scratch/smollm2_135m_raw.bin",
    output_vocab: str = "c_scratch/tokenizer.bin"
):
    print("=" * 65)
    print("       EXPORTING TO RAW BINARY FOR PURE C ENGINE                 ")
    print("=" * 65)

    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    dim = config["hidden_size"]  # 576
    hidden_dim = config["intermediate_size"]  # 1536
    n_layers = config["num_hidden_layers"]  # 30
    n_heads = config["num_attention_heads"]  # 9
    n_kv_heads = config.get("num_key_value_heads", n_heads)  # 3
    vocab_size = config["vocab_size"]  # 49152
    seq_len = 2048

    print(f"[*] Architecture params: dim={dim}, hidden={hidden_dim}, layers={n_layers}, heads={n_heads}/{n_kv_heads}, vocab={vocab_size}")

    # Read safetensors
    tensors = {}
    print(f"[*] Reading {weights_path}...")
    with safe_open(weights_path, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            tensors[k] = sf.get_tensor(k).float().numpy()

    # 1. Write Model Binary
    print(f"[*] Writing weights to {output_bin}...")
    with open(output_bin, "wb") as f:
        # Header (7 ints): dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        header = struct.pack(
            "iiiiiii",
            dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        )
        f.write(header)

        # 1. token_embedding_table: [vocab_size, dim]
        f.write(tensors["model.embed_tokens.weight"].astype(np.float32).tobytes())

        # 2. rms_att_weight: [n_layers, dim]
        for i in range(n_layers):
            f.write(tensors[f"model.layers.{i}.input_layernorm.weight"].astype(np.float32).tobytes())

        # 3. wq: [n_layers, dim, dim]
        for i in range(n_layers):
            f.write(tensors[f"model.layers.{i}.self_attn.q_proj.weight"].astype(np.float32).tobytes())

        # 4. wk: [n_layers, kv_dim, dim]
        for i in range(n_layers):
            f.write(tensors[f"model.layers.{i}.self_attn.k_proj.weight"].astype(np.float32).tobytes())

        # 5. wv: [n_layers, kv_dim, dim]
        for i in range(n_layers):
            f.write(tensors[f"model.layers.{i}.self_attn.v_proj.weight"].astype(np.float32).tobytes())

        # 6. wo: [n_layers, dim, dim]
        for i in range(n_layers):
            f.write(tensors[f"model.layers.{i}.self_attn.o_proj.weight"].astype(np.float32).tobytes())

        # 7. rms_ffn_weight: [n_layers, dim]
        for i in range(n_layers):
            f.write(tensors[f"model.layers.{i}.post_attention_layernorm.weight"].astype(np.float32).tobytes())

        # 8. w1 (gate_proj): [n_layers, hidden_dim, dim]
        for i in range(n_layers):
            f.write(tensors[f"model.layers.{i}.mlp.gate_proj.weight"].astype(np.float32).tobytes())

        # 9. w2 (down_proj): [n_layers, dim, hidden_dim]
        for i in range(n_layers):
            f.write(tensors[f"model.layers.{i}.mlp.down_proj.weight"].astype(np.float32).tobytes())

        # 10. w3 (up_proj): [n_layers, hidden_dim, dim]
        for i in range(n_layers):
            f.write(tensors[f"model.layers.{i}.mlp.up_proj.weight"].astype(np.float32).tobytes())

        # 11. rms_final_weight: [dim]
        f.write(tensors["model.norm.weight"].astype(np.float32).tobytes())

        # 12. wcls (lm_head): [vocab_size, dim]
        if "lm_head.weight" in tensors:
            f.write(tensors["lm_head.weight"].astype(np.float32).tobytes())
        else:
            f.write(tensors["model.embed_tokens.weight"].astype(np.float32).tobytes())

    size_mb = os.path.getsize(output_bin) / (1024 * 1024)
    print(f"[+] Successfully exported model weights: {output_bin} ({size_mb:.2f} MB)")

    # 2. Write Tokenizer Binary
    print(f"[*] Parsing tokenizer from {tokenizer_path}...")
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    vocab_dict = tok_data["model"]["vocab"]  # token_str -> id
    sorted_tokens = [""] * vocab_size
    for token_str, idx in vocab_dict.items():
        if idx < vocab_size:
            sorted_tokens[idx] = token_str

    for item in tok_data.get("added_tokens", []):
        t_id = item["id"]
        if t_id < vocab_size:
            sorted_tokens[t_id] = item["content"]

    print(f"[*] Writing tokenizer vocab to {output_vocab}...")
    with open(output_vocab, "wb") as f:
        # Number of tokens
        f.write(struct.pack("i", vocab_size))
        for token_str in sorted_tokens:
            token_bytes = token_str.encode("utf-8")
            # score (float), length (int), bytes
            f.write(struct.pack("f", 0.0))
            f.write(struct.pack("i", len(token_bytes)))
            f.write(token_bytes)

    vocab_mb = os.path.getsize(output_vocab) / (1024 * 1024)
    print(f"[+] Successfully exported tokenizer: {output_vocab} ({vocab_mb:.2f} MB)")


if __name__ == "__main__":
    export_model_to_raw_bin()
