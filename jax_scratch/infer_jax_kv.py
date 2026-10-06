"""
Benchmark & Verify JAX Inference with Static KV-Cache.
Demonstrates O(1) decode latency with zero recompilation.
"""
from __future__ import annotations
import os
import sys
import time
from functools import partial
import numpy as np
from transformers import AutoTokenizer
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_jax import SmolLMConfig, precompute_rope_tables
from llama_jax_kv import KVCache, decode_step_smollm
from infer_jax import load_npz_to_jax


def run_jax_kv_cache_demo(
    prompt: str = "The theory of relativity explains that",
    npz_path: str = "smollm2_135m_jax.npz",
    model_dir: str = "models/SmolLM2-135M",
    max_new_tokens: int = 25,
    max_seq_len: int = 128
):
    print("=" * 65)
    print("      JAX AUTOREGRESSIVE GENERATION WITH STATIC KV-CACHE         ")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print(f"[*] Prompt: '{prompt}'")

    print(f"[*] Loading parameters from {npz_path}...")
    params = load_npz_to_jax(npz_path)
    config = SmolLMConfig()

    rope_tables = precompute_rope_tables(config.head_dim, max_seq_len, config.rope_theta)

    # 1. Allocate static KV-cache
    print(f"[*] Allocating static KV-Cache [30 layers, 1 batch, {max_seq_len} max_len, 3 kv_heads, 64 head_dim]...")
    cache = KVCache.create(
        num_layers=config.num_hidden_layers,
        batch=1,
        max_seq_len=max_seq_len,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )

    # 2. JIT compile the decode step
    # cur_pos is static or dynamic? Let's make it a dynamic argument or pass as scalar
    @jax.jit
    def jitted_decode_step(token_id, cur_pos, c):
        return decode_step_smollm(params, token_id, cur_pos, c, rope_tables, config)

    prompt_tokens = tokenizer.encode(prompt)
    print(f"[*] Prefilling {len(prompt_tokens)} prompt tokens into KV-Cache...")

    cur_pos = 0
    t0_prefill = time.time()
    # Feed prompt tokens to build initial cache
    next_logits = None
    for tid in prompt_tokens:
        tok_tensor = jnp.array([[tid]])
        next_logits, cache = jitted_decode_step(tok_tensor, cur_pos, cache)
        cur_pos += 1
    t1_prefill = time.time()
    print(f"[+] Prefill completed in {t1_prefill - t0_prefill:.3f}s (includes first-time JIT compile)")

    # 3. Autoregressive Generation with KV-Cache
    print(f"\n[*] Generating {max_new_tokens} new tokens with KV-Cache...")
    generated = list(prompt_tokens)

    step_times = []
    for step in range(max_new_tokens):
        t0 = time.time()
        next_token = int(np.argmax(np.array(next_logits[0])))
        generated.append(next_token)

        # Feed single new token with KV-cache
        tok_tensor = jnp.array([[next_token]])
        next_logits, cache = jitted_decode_step(tok_tensor, cur_pos, cache)
        # Block until ready for accurate timing
        next_logits.block_until_ready()
        t1 = time.time()
        step_times.append(t1 - t0)

        token_str = tokenizer.decode([next_token])
        print(f"  Token {step+1:02d} (+{(t1 - t0)*1000:5.1f}ms): '{token_str}' (ID={next_token})")
        cur_pos += 1

    avg_latency = np.mean(step_times) * 1000
    tps = 1000.0 / avg_latency
    output_text = tokenizer.decode(generated)

    print("\n" + "=" * 65)
    print("                     BENCHMARK RESULTS                           ")
    print("=" * 65)
    print(f"Average Decode Latency : {avg_latency:.2f} ms/token")
    print(f"Throughput             : {tps:.2f} tokens/second (Pure JAX CPU)")
    print(f"\nFull Output Text:\n--> {output_text}\n")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The theory of relativity explains that"
    run_jax_kv_cache_demo(prompt)
