"""
SmolLM2 to GGUF Converter implemented completely from scratch.
Reads Hugging Face config.json, tokenizer.json, and model.safetensors.
Writes standard GGUF v3 file compatible with any llama.cpp runtime.
"""
from __future__ import annotations
import json
import os
import sys

# Ensure local directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors import safe_open
import numpy as np

from gguf_writer import GGUFWriter, GGMLType


def parse_tokenizer_json(tokenizer_path: str):
    """
    Extract vocabulary, merges, and scores from tokenizer.json from scratch.
    """
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    vocab_dict = tok_data["model"]["vocab"]  # token -> id
    # Sort tokens by ID
    tokens = [""] * len(vocab_dict)
    for token, idx in vocab_dict.items():
        tokens[idx] = token

    merges = tok_data["model"].get("merges", [])
    scores = [0.0] * len(tokens)
    # Default token types: 1 = normal, 2 = unknown, 3 = control, 4 = user defined
    token_types = [1] * len(tokens)

    # Check added_tokens
    for item in tok_data.get("added_tokens", []):
        t_id = item["id"]
        t_content = item["content"]
        if t_id < len(tokens):
            tokens[t_id] = t_content
            if item.get("special", False):
                token_types[t_id] = 3

    return tokens, scores, token_types, merges


def permute_for_ggml(weights: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Permutes weight matrix from HuggingFace RoPE layout to GGML layout.
    HF groups (d/2, 2) while GGML groups (2, d/2) for consecutive rotation.
    Without this permutation, attention RoPE calculates completely wrong frequencies.
    """
    out_dim, in_dim = weights.shape
    head_dim = out_dim // n_heads
    w = weights.reshape(n_heads, 2, head_dim // 2, in_dim)
    w = np.swapaxes(w, 1, 2)
    return w.reshape(out_dim, in_dim)


def convert_smollm_to_gguf(
    model_dir: str,
    output_path: str,
    out_type: str = "f16"  # "f16" or "q8_0"
):
    print(f"[*] Starting SmolLM2 -> GGUF ({out_type}) conversion from scratch...")
    config_path = os.path.join(model_dir, "config.json")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    weights_path = os.path.join(model_dir, "model.safetensors")

    assert os.path.exists(config_path), f"Missing {config_path}"
    assert os.path.exists(tokenizer_path), f"Missing {tokenizer_path}"
    assert os.path.exists(weights_path), f"Missing {weights_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    writer = GGUFWriter(output_path)

    # 1. Architecture metadata
    arch = "llama"
    writer.add_string("general.architecture", arch)
    writer.add_string("general.name", "SmolLM2-135M")
    writer.add_uint32(f"{arch}.block_count", config["num_hidden_layers"])
    writer.add_uint32(f"{arch}.context_length", config.get("max_position_embeddings", 8192))
    writer.add_uint32(f"{arch}.embedding_length", config["hidden_size"])
    writer.add_uint32(f"{arch}.feed_forward_length", config["intermediate_size"])
    writer.add_uint32(f"{arch}.attention.head_count", config["num_attention_heads"])
    writer.add_uint32(f"{arch}.attention.head_count_kv", config.get("num_key_value_heads", config["num_attention_heads"]))
    writer.add_float32(f"{arch}.rope.freq_base", config.get("rope_theta", 100000.0))
    writer.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", config.get("rms_norm_eps", 1e-5))

    # 2. Tokenizer metadata
    print("[*] Parsing tokenizer vocabulary & merges...")
    tokens, scores, token_types, merges = parse_tokenizer_json(tokenizer_path)
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "smollm")
    writer.add_string_array("tokenizer.ggml.tokens", tokens)
    writer.add_float32_array("tokenizer.ggml.scores", scores)
    writer.add_int32_array("tokenizer.ggml.token_type", token_types)
    if merges:
        writer.add_string_array("tokenizer.ggml.merges", merges)

    bos_id = config.get("bos_token_id", 0)
    eos_id = config.get("eos_token_id", 0)
    writer.add_uint32("tokenizer.ggml.bos_token_id", bos_id)
    writer.add_uint32("tokenizer.ggml.eos_token_id", eos_id)

    # 3. Read & Map safetensors weights
    print("[*] Mapping PyTorch weights to GGML tensor naming convention...")
    tensors = {}
    with safe_open(weights_path, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            tensors[k] = sf.get_tensor(k).float().numpy()

    # GGML tensor name mapping
    name_map = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    for i in range(config["num_hidden_layers"]):
        name_map.update({
            f"model.layers.{i}.input_layernorm.weight": f"blk.{i}.attn_norm.weight",
            f"model.layers.{i}.self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
            f"model.layers.{i}.self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
            f"model.layers.{i}.self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
            f"model.layers.{i}.self_attn.o_proj.weight": f"blk.{i}.attn_output.weight",
            f"model.layers.{i}.post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
            f"model.layers.{i}.mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
            f"model.layers.{i}.mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
            f"model.layers.{i}.mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
        })

    # If tie_word_embeddings is true and lm_head is missing, reuse embed_tokens
    if config.get("tie_word_embeddings", False) and "lm_head.weight" not in tensors:
        print("[*] Tied word embeddings: Duplicating token_embd.weight as output.weight")
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].copy()

    # Add tensors to writer
    selected_type = GGMLType.Q8_0 if out_type == "q8_0" else GGMLType.F16

    for pt_name, ggml_name in name_map.items():
        if pt_name not in tensors:
            print(f"[!] Warning: {pt_name} not found in safetensors!")
            continue

        arr = tensors[pt_name]

        # Apply RoPE permutation for Q and K weights for GGML
        if ggml_name.endswith(".attn_q.weight"):
            arr = permute_for_ggml(arr, config["num_attention_heads"])
        elif ggml_name.endswith(".attn_k.weight"):
            arr = permute_for_ggml(arr, config.get("num_key_value_heads", config["num_attention_heads"]))

        # 1D tensors (layernorms) always remain F32 or F16, never quantized
        if arr.ndim == 1:
            writer.add_tensor(ggml_name, arr, GGMLType.F32)
        else:
            # 2D matrix weights: use selected_type
            writer.add_tensor(ggml_name, arr, selected_type)

    # 4. Write binary file
    writer.write()
    print(f"[+] Done converting to {output_path}!")


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/SmolLM2-135M"
    out_format = sys.argv[2] if len(sys.argv) > 2 else "f16"
    out_file = f"smollm2_135m_{out_format}.gguf"
    convert_smollm_to_gguf(model_dir, out_file, out_format)
