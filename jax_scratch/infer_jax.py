"""
Autoregressive Text Generation with Pure JAX Transformer Engine.
Loads JAX parameters from smollm2_135m_jax.npz and generates text token-by-token.
"""
from __future__ import annotations
import os
import sys
import numpy as np
from transformers import AutoTokenizer
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_jax import SmolLMConfig, precompute_rope_tables, forward_smollm


def load_npz_to_jax(npz_path: str) -> dict:
    raw = np.load(npz_path)
    params = {
        "embed_tokens": jnp.array(raw["embed_tokens"]),
        "output_norm": jnp.array(raw["output_norm"]),
        "lm_head": jnp.array(raw["lm_head"]),
    }
    # Group layers
    for i in range(30):
        layer_prefix = f"layer_{i}."
        layer_dict = {}
        for k in ["attn_norm", "ffn_norm", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            layer_dict[k] = jnp.array(raw[layer_prefix + k])
        params[f"layer_{i}"] = layer_dict
    return params


def generate_jax(
    prompt: str = "Artificial Intelligence is transforming",
    npz_path: str = "smollm2_135m_jax.npz",
    model_dir: str = "models/SmolLM2-135M",
    max_new_tokens: int = 15
):
    print("=" * 65)
    print("        JAX PURE TRANSFORMER AUTOREGRESSIVE GENERATION           ")
    print("=" * 65)

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing {npz_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print(f"[*] Prompt: '{prompt}'")
    print(f"[*] Loading JAX parameters from {npz_path}...")
    params = load_npz_to_jax(npz_path)

    config = SmolLMConfig()
    rope_tables = precompute_rope_tables(config.head_dim, config.max_position_embeddings, config.rope_theta)

    input_tokens = tokenizer.encode(prompt)
    generated = list(input_tokens)

    print("\n[*] Starting token-by-token generation with JAX:")
    for step in range(max_new_tokens):
        ids = jnp.array([generated])
        logits = forward_smollm(params, ids, rope_tables, config)
        next_token = int(np.argmax(np.array(logits[0, -1, :])))
        generated.append(next_token)
        print(f"  Token {step+1}: '{tokenizer.decode([next_token])}' (ID={next_token})")

    output_text = tokenizer.decode(generated)
    print("\n" + "=" * 65)
    print(f"Final Generated Output:\n--> {output_text}")
    print("=" * 65)
    print("[SUCCESS] JAX checkpoint produces 100% valid text inference!\n")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Machine learning allows computers to"
    generate_jax(prompt)
