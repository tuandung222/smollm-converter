"""
Verify JAX from-scratch LLaMA implementation against official PyTorch Hugging Face model.
Tests numerical parity (MSE, Max Diff, Cosine Similarity) and autoregressive generation.
"""
from __future__ import annotations
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import jax
import jax.numpy as jnp
import numpy as np

# Ensure local dir is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_jax import SmolLMConfig, precompute_rope_tables, forward_smollm
from convert_smollm_jax import load_smollm_to_jax_params


def verify_numerical_parity(model_dir: str = "models/SmolLM2-135M"):
    print("=" * 65)
    print("      VERIFYING JAX LLaMA FROM SCRATCH vs PYTORCH HUGGINGFACE    ")
    print("=" * 65)

    # 1. Load Tokenizer & Prompt
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompt = "Artificial Intelligence is transforming the world because"
    input_ids_pt = tokenizer.encode(prompt, return_tensors="pt")
    input_ids_np = input_ids_pt.numpy()
    seq_len = input_ids_np.shape[1]

    print(f"[*] Prompt: '{prompt}'")
    print(f"[*] Token count: {seq_len} tokens")

    # 2. PyTorch Forward Pass
    print("\n[*] [1/2] Running PyTorch Hugging Face Forward Pass...")
    pt_model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model(input_ids_pt)
        logits_pt = pt_out.logits.numpy()

    # 3. JAX Forward Pass
    print("[*] [2/2] Running Pure JAX Forward Pass (with jax.jit)...")
    jax_params, config_dict = load_smollm_to_jax_params(model_dir, dtype=jnp.float32)

    config = SmolLMConfig(
        vocab_size=config_dict["vocab_size"],
        hidden_size=config_dict["hidden_size"],
        intermediate_size=config_dict["intermediate_size"],
        num_hidden_layers=config_dict["num_hidden_layers"],
        num_attention_heads=config_dict["num_attention_heads"],
        num_key_value_heads=config_dict.get("num_key_value_heads", config_dict["num_attention_heads"]),
        rms_norm_eps=config_dict.get("rms_norm_eps", 1e-5),
        rope_theta=config_dict.get("rope_theta", 100000.0),
        max_position_embeddings=config_dict.get("max_position_embeddings", 8192),
    )

    rope_tables = precompute_rope_tables(config.head_dim, config.max_position_embeddings, config.rope_theta)

    # JIT compile the forward pass
    @jax.jit
    def jitted_forward(ids):
        return forward_smollm(jax_params, ids, rope_tables, config)

    input_ids_jax = jnp.array(input_ids_np)
    # Warmup / compile
    logits_jax = jitted_forward(input_ids_jax)
    logits_jax = np.array(logits_jax)

    # 4. Compare Logits
    print("\n" + "=" * 65)
    print("                   NUMERICAL PARITY RESULTS                      ")
    print("=" * 65)

    diff = np.abs(logits_pt - logits_jax)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Cosine similarity on the last token's logits
    pt_last = logits_pt[0, -1, :].astype(np.float64)
    jax_last = logits_jax[0, -1, :].astype(np.float64)
    cos_sim = np.dot(pt_last, jax_last) / (np.linalg.norm(pt_last) * np.linalg.norm(jax_last))

    # Top-5 predicted next tokens
    top5_pt = np.argsort(pt_last)[::-1][:5]
    top5_jax = np.argsort(jax_last)[::-1][:5]

    print(f"Max Absolute Error  : {max_diff:.6e}")
    print(f"Mean Absolute Error : {mean_diff:.6e}")
    print(f"Cosine Similarity   : {cos_sim:.8f}")

    print("\nNext Token Predictions:")
    print("  PyTorch Top-5 Tokens:")
    for rank, tid in enumerate(top5_pt):
        token_str = repr(tokenizer.decode([tid]))
        print(f"    {rank+1}. ID={tid:<6} Logit={pt_last[tid]:.3f} Token={token_str}")

    print("  JAX Top-5 Tokens:")
    for rank, tid in enumerate(top5_jax):
        token_str = repr(tokenizer.decode([tid]))
        print(f"    {rank+1}. ID={tid:<6} Logit={jax_last[tid]:.3f} Token={token_str}")

    assert top5_pt[0] == top5_jax[0], "Top token mismatch between PyTorch and JAX!"
    assert cos_sim > 0.9999, f"Cosine similarity {cos_sim} too low!"
    print("\n[SUCCESS] JAX from scratch matches PyTorch with near-zero error!")

    # 5. Greedy Autoregressive Generation Test in JAX
    print("\n" + "=" * 65)
    print("            JAX GREEDY GENERATION DEMO (5 TOKENS)                ")
    print("=" * 65)
    generated = list(input_ids_np[0])
    for step in range(5):
        curr_ids = jnp.array([generated])
        # Direct execution without re-jitting per token length
        curr_logits = forward_smollm(jax_params, curr_ids, rope_tables, config)
        next_token = int(np.argmax(np.array(curr_logits[0, -1, :])))
        generated.append(next_token)
        print(f"  Step {step+1}: + '{tokenizer.decode([next_token])}' (ID={next_token})")

    output_text = tokenizer.decode(generated)
    print(f"\nGenerated full prompt:\n--> {output_text}\n")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/SmolLM2-135M"
    verify_numerical_parity(model_path)
