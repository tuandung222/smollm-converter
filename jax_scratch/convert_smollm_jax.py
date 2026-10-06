"""
Weight Converter for SmolLM2 to JAX PyTree format.
Loads safetensors directly and formats parameter dictionary for our pure JAX LLaMA model.
Can save to .npz or return in-memory JAX PyTree.
"""
from __future__ import annotations
import json
import os
import sys
from typing import Dict, Any, Tuple
from safetensors import safe_open
import jax.numpy as jnp
import numpy as np


def load_smollm_to_jax_params(model_dir: str, dtype=jnp.float32) -> Tuple[Dict[str, Any], dict]:
    """
    Load PyTorch safetensors and convert into JAX PyTree parameter dict.
    Transposes linear weights to match JAX (x @ W) convention.
    """
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"[*] Reading PyTorch safetensors from {weights_path}...")
    pt_tensors = {}
    with safe_open(weights_path, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            pt_tensors[k] = sf.get_tensor(k).float().numpy()

    params = {}

    # Token embeddings: [vocab_size, hidden_size]
    embed_w = pt_tensors["model.embed_tokens.weight"]
    params["embed_tokens"] = jnp.array(embed_w, dtype=dtype)

    # Output normalization: [hidden_size]
    params["output_norm"] = jnp.array(pt_tensors["model.norm.weight"], dtype=dtype)

    # LM Head: [hidden_size, vocab_size]
    if "lm_head.weight" in pt_tensors:
        params["lm_head"] = jnp.array(pt_tensors["lm_head.weight"].T, dtype=dtype)
    else:
        # Tied embeddings
        params["lm_head"] = jnp.array(embed_w.T, dtype=dtype)

    # Layers
    num_layers = config["num_hidden_layers"]
    print(f"[*] Converting {num_layers} Transformer layers to JAX convention...")

    for i in range(num_layers):
        layer_params = {
            # Attention norms
            "attn_norm": jnp.array(pt_tensors[f"model.layers.{i}.input_layernorm.weight"], dtype=dtype),
            "ffn_norm": jnp.array(pt_tensors[f"model.layers.{i}.post_attention_layernorm.weight"], dtype=dtype),
            # Attention linear projections (transposed from [out, in] to [in, out])
            "q_proj": jnp.array(pt_tensors[f"model.layers.{i}.self_attn.q_proj.weight"].T, dtype=dtype),
            "k_proj": jnp.array(pt_tensors[f"model.layers.{i}.self_attn.k_proj.weight"].T, dtype=dtype),
            "v_proj": jnp.array(pt_tensors[f"model.layers.{i}.self_attn.v_proj.weight"].T, dtype=dtype),
            "o_proj": jnp.array(pt_tensors[f"model.layers.{i}.self_attn.o_proj.weight"].T, dtype=dtype),
            # SwiGLU MLP linear projections
            "gate_proj": jnp.array(pt_tensors[f"model.layers.{i}.mlp.gate_proj.weight"].T, dtype=dtype),
            "up_proj": jnp.array(pt_tensors[f"model.layers.{i}.mlp.up_proj.weight"].T, dtype=dtype),
            "down_proj": jnp.array(pt_tensors[f"model.layers.{i}.mlp.down_proj.weight"].T, dtype=dtype),
        }
        params[f"layer_{i}"] = layer_params

    print("[+] Successfully mapped all parameters into JAX PyTree!")
    return params, config


def export_jax_npz(model_dir: str, output_npz_path: str):
    """
    Flatten PyTree and save as compressed .npz file.
    """
    params, config = load_smollm_to_jax_params(model_dir)
    flat_dict = {}

    def _flatten(d, prefix=""):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, full_key)
            else:
                flat_dict[full_key] = np.array(v)

    _flatten(params)
    print(f"[*] Saving {len(flat_dict)} arrays to {output_npz_path}...")
    np.savez_compressed(output_npz_path, **flat_dict)
    size_mb = os.path.getsize(output_npz_path) / (1024 * 1024)
    print(f"[+] Saved JAX weights to {output_npz_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    src_dir = sys.argv[1] if len(sys.argv) > 1 else "models/SmolLM2-135M"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "smollm2_135m_jax.npz"
    export_jax_npz(src_dir, out_file)
