"""
SmolLM2 / LLaMA Architecture implemented from scratch in pure JAX.
No Hugging Face model classes, no Flax Linen modules.
Functional, JIT-compilable, transparent.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class SmolLMConfig:
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    max_position_embeddings: int = 8192

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """
    Root Mean Square Layer Normalization.
    x: [..., hidden_size]
    weight: [hidden_size]
    """
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + eps)
    return (normed * weight).astype(x.dtype)


def precompute_rope_tables(
    head_dim: int, max_seq_len: int = 8192, theta: float = 100000.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precompute cosine and sine tables according to Hugging Face LLaMA RoPE convention.
    """
    dim_indices = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (theta ** (dim_indices / head_dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)  # [max_seq_len, head_dim // 2]
    # HF concatenates (freqs, freqs) along last dimension
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # [max_seq_len, head_dim]
    cos = jnp.cos(emb)
    sin = jnp.sin(emb)
    return cos, sin


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """
    Rotates half the hidden dimensions of the input vector (Hugging Face convention).
    """
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_emb(
    q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply Rotary Position Embeddings to query and key tensors.
    q: [batch, seq_len, num_heads, head_dim]
    k: [batch, seq_len, num_kv_heads, head_dim]
    cos, sin: [seq_len, head_dim]
    """
    # Expand cos/sin to [1, seq_len, 1, head_dim]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(q.dtype), k_embed.astype(k.dtype)


def repeat_kv(x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """
    Grouped Query Attention helper: expand KV heads to match Q heads.
    x: [batch, seq_len, n_kv_heads, head_dim]
    """
    if n_rep == 1:
        return x
    batch, seq_len, n_kv_heads, head_dim = x.shape
    x = jnp.expand_dims(x, axis=3)  # [batch, seq_len, n_kv_heads, 1, head_dim]
    x = jnp.broadcast_to(x, (batch, seq_len, n_kv_heads, n_rep, head_dim))
    return x.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)


def grouped_query_attention(
    x: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    mask: Optional[jnp.ndarray],
    config: SmolLMConfig,
) -> jnp.ndarray:
    """
    Grouped-Query Attention (GQA) with RoPE.
    """
    batch, seq_len, _ = x.shape
    q = x @ params["q_proj"]  # [batch, seq_len, num_heads * head_dim]
    k = x @ params["k_proj"]  # [batch, seq_len, num_kv_heads * head_dim]
    v = x @ params["v_proj"]  # [batch, seq_len, num_kv_heads * head_dim]

    q = q.reshape(batch, seq_len, config.num_attention_heads, config.head_dim)
    k = k.reshape(batch, seq_len, config.num_key_value_heads, config.head_dim)
    v = v.reshape(batch, seq_len, config.num_key_value_heads, config.head_dim)

    # RoPE
    q, k = apply_rotary_emb(q, k, cos, sin)

    # GQA repeat
    n_rep = config.num_attention_heads // config.num_key_value_heads
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    # Transpose for dot product: [batch, num_heads, seq_len, head_dim]
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    scale = 1.0 / np.sqrt(config.head_dim)
    # [batch, num_heads, seq_len_q, seq_len_k]
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale

    if mask is not None:
        scores = scores + mask

    attn_weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    context = jnp.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]

    # Recombine heads
    context = jnp.transpose(context, (0, 2, 1, 3)).reshape(batch, seq_len, config.hidden_size)
    return context @ params["o_proj"]


def swiglu_mlp(x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    SwiGLU Feed-Forward Network: down_proj(silu(gate_proj(x)) * up_proj(x))
    """
    gate = jax.nn.silu(x @ params["gate_proj"])
    up = x @ params["up_proj"]
    return (gate * up) @ params["down_proj"]


def transformer_layer(
    x: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    mask: Optional[jnp.ndarray],
    config: SmolLMConfig,
) -> jnp.ndarray:
    """
    Single Decoder Layer with pre-norm and residual connections.
    """
    # 1. Pre-norm attention
    norm_x = rms_norm(x, params["attn_norm"], config.rms_norm_eps)
    attn_out = grouped_query_attention(norm_x, params, cos, sin, mask, config)
    x = x + attn_out

    # 2. Pre-norm MLP
    norm_x2 = rms_norm(x, params["ffn_norm"], config.rms_norm_eps)
    mlp_out = swiglu_mlp(norm_x2, params)
    x = x + mlp_out
    return x


def forward_smollm(
    params: Dict[str, Any],
    input_ids: jnp.ndarray,
    rope_tables: Tuple[jnp.ndarray, jnp.ndarray],
    config: SmolLMConfig,
) -> jnp.ndarray:
    """
    Complete SmolLM2 forward pass in JAX from scratch.
    input_ids: [batch, seq_len]
    returns: logits [batch, seq_len, vocab_size]
    """
    batch, seq_len = input_ids.shape
    cos_table, sin_table = rope_tables
    cos = cos_table[:seq_len]
    sin = sin_table[:seq_len]

    # Causal triangular attention mask: 0 for attend, -1e9 for masked
    mask = jnp.triu(jnp.full((seq_len, seq_len), -1e9), k=1)
    mask = mask[None, None, :, :]  # [1, 1, seq_len, seq_len]

    # Token embeddings
    x = params["embed_tokens"][input_ids]

    # Layers
    for i in range(config.num_hidden_layers):
        x = transformer_layer(x, params[f"layer_{i}"], cos, sin, mask, config)

    # Final norm
    x = rms_norm(x, params["output_norm"], config.rms_norm_eps)

    # Head projection
    logits = x @ params["lm_head"]
    return logits
