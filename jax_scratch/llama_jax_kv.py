"""
SmolLM2 / LLaMA Architecture with Static Functional KV-Cache in pure JAX.
No Hugging Face model classes, zero dynamic recompilations.
Operates in 2 phases:
  1. Prefill: processes initial prompt tokens, initializes KV-Cache.
  2. Decode Step: processes exactly 1 token at cur_pos, updates cache via dynamic_update_slice.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import jax
import jax.numpy as jnp
import numpy as np

# Ensure local imports
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_jax import SmolLMConfig, rms_norm, precompute_rope_tables, apply_rotary_emb, repeat_kv, swiglu_mlp


@dataclass
class KVCache:
    """
    Fixed-size preallocated KV-Cache for all 30 layers.
    Shape: [num_layers, batch, max_seq_len, num_kv_heads, head_dim]
    """
    k: jnp.ndarray
    v: jnp.ndarray

    @classmethod
    def create(cls, num_layers: int, batch: int, max_seq_len: int, num_kv_heads: int, head_dim: int, dtype=jnp.float32):
        k = jnp.zeros((num_layers, batch, max_seq_len, num_kv_heads, head_dim), dtype=dtype)
        v = jnp.zeros((num_layers, batch, max_seq_len, num_kv_heads, head_dim), dtype=dtype)
        return cls(k=k, v=v)


# Register KVCache as a native JAX PyTree node
jax.tree_util.register_pytree_node(
    KVCache,
    lambda c: ((c.k, c.v), None),
    lambda aux, children: KVCache(k=children[0], v=children[1])
)


def attention_decode_step(
    x: jnp.ndarray,
    layer_idx: int,
    cur_pos: int,
    cache: KVCache,
    params: Dict[str, jnp.ndarray],
    cos_table: jnp.ndarray,
    sin_table: jnp.ndarray,
    config: SmolLMConfig,
) -> Tuple[jnp.ndarray, KVCache]:
    """
    Attention forward for a SINGLE token using KV-Cache.
    x: [batch, 1, hidden_size]
    """
    batch = x.shape[0]

    # Linear projections for single token
    q = x @ params["q_proj"]  # [batch, 1, num_heads * head_dim]
    k = x @ params["k_proj"]  # [batch, 1, num_kv_heads * head_dim]
    v = x @ params["v_proj"]  # [batch, 1, num_kv_heads * head_dim]

    q = q.reshape(batch, 1, config.num_attention_heads, config.head_dim)
    k = k.reshape(batch, 1, config.num_key_value_heads, config.head_dim)
    v = v.reshape(batch, 1, config.num_key_value_heads, config.head_dim)

    # RoPE for single token at cur_pos using dynamic_slice
    cos = jax.lax.dynamic_slice(cos_table, (cur_pos, 0), (1, config.head_dim))
    sin = jax.lax.dynamic_slice(sin_table, (cur_pos, 0), (1, config.head_dim))
    q, k = apply_rotary_emb(q, k, cos, sin)

    # Update KV-Cache at cur_pos using dynamic_update_slice
    # cache.k[layer_idx] shape: [batch, max_seq_len, num_kv_heads, head_dim]
    layer_k = cache.k[layer_idx]
    layer_v = cache.v[layer_idx]

    updated_layer_k = jax.lax.dynamic_update_slice(
        layer_k, k, (0, cur_pos, 0, 0)
    )
    updated_layer_v = jax.lax.dynamic_update_slice(
        layer_v, v, (0, cur_pos, 0, 0)
    )

    new_k = cache.k.at[layer_idx].set(updated_layer_k)
    new_v = cache.v.at[layer_idx].set(updated_layer_v)
    new_cache = KVCache(k=new_k, v=new_v)

    # Read all keys and values up to max_seq_len
    # For dot product, we attend to tokens 0..cur_pos
    k_all = updated_layer_k  # [batch, max_seq_len, num_kv_heads, head_dim]
    v_all = updated_layer_v

    # GQA repeat
    n_rep = config.num_attention_heads // config.num_key_value_heads
    k_all = repeat_kv(k_all, n_rep)
    v_all = repeat_kv(v_all, n_rep)

    # [batch, num_heads, 1, head_dim]
    q = jnp.transpose(q, (0, 2, 1, 3))
    # [batch, num_heads, max_seq_len, head_dim]
    k_all = jnp.transpose(k_all, (0, 2, 1, 3))
    v_all = jnp.transpose(v_all, (0, 2, 1, 3))

    scale = 1.0 / np.sqrt(config.head_dim)
    # [batch, num_heads, 1, max_seq_len]
    scores = jnp.matmul(q, jnp.swapaxes(k_all, -1, -2)) * scale

    # Mask out positions > cur_pos
    seq_idx = jnp.arange(cache.k.shape[2])
    mask = jnp.where(seq_idx <= cur_pos, 0.0, -1e9)
    scores = scores + mask[None, None, None, :]

    attn_weights = jax.nn.softmax(scores, axis=-1)
    context = jnp.matmul(attn_weights, v_all)  # [batch, num_heads, 1, head_dim]

    context = jnp.transpose(context, (0, 2, 1, 3)).reshape(batch, 1, config.hidden_size)
    out = context @ params["o_proj"]
    return out, new_cache


def decode_step_smollm(
    params: Dict[str, Any],
    token_id: jnp.ndarray,  # [batch, 1]
    cur_pos: int,
    cache: KVCache,
    rope_tables: Tuple[jnp.ndarray, jnp.ndarray],
    config: SmolLMConfig,
) -> Tuple[jnp.ndarray, KVCache]:
    """
    Single-step decode with KV-Cache.
    Always receives input shape [1, 1], guaranteeing zero XLA recompilations!
    """
    cos_table, sin_table = rope_tables
    x = params["embed_tokens"][token_id]  # [batch, 1, hidden_size]

    current_cache = cache
    for i in range(config.num_hidden_layers):
        layer_params = params[f"layer_{i}"]
        # 1. Attn Pre-norm
        norm_x = rms_norm(x, layer_params["attn_norm"], config.rms_norm_eps)
        attn_out, current_cache = attention_decode_step(
            norm_x, i, cur_pos, current_cache, layer_params, cos_table, sin_table, config
        )
        x = x + attn_out

        # 2. MLP Pre-norm
        norm_x2 = rms_norm(x, layer_params["ffn_norm"], config.rms_norm_eps)
        mlp_out = swiglu_mlp(norm_x2, layer_params)
        x = x + mlp_out

    # Final norm & LM head
    x = rms_norm(x, params["output_norm"], config.rms_norm_eps)
    logits = x @ params["lm_head"]  # [batch, 1, vocab_size]
    return logits[:, 0, :], current_cache
