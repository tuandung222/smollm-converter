"""
Clean, self-contained SmolLM2 PyTorch Model definition optimized for Edge / LiteRT export.
Removes Hugging Face dynamic caches and control flows to ensure 100% graph trace compatibility.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EdgeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class EdgeRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 100000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed


class EdgeAttention(nn.Module):
    def __init__(self, hidden_size: int = 576, num_heads: int = 9, num_kv_heads: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads  # 64
        self.n_rep = num_heads // num_kv_heads  # 3

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, rotary: EdgeRotaryEmbedding) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        q, k = rotary(q, k, seq_len)

        # Repeat KV for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # Transpose: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Causal mask
        mask = torch.triu(torch.full((seq_len, seq_len), -1e9, device=x.device), diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # [batch, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(out)


class EdgeMLP(nn.Module):
    def __init__(self, hidden_size: int = 576, intermediate_size: int = 1536):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class EdgeDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, num_kv_heads: int, eps: float):
        super().__init__()
        self.input_layernorm = EdgeRMSNorm(hidden_size, eps)
        self.self_attn = EdgeAttention(hidden_size, num_heads, num_kv_heads)
        self.post_attention_layernorm = EdgeRMSNorm(hidden_size, eps)
        self.mlp = EdgeMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, rotary: EdgeRotaryEmbedding) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), rotary)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class SmolLMEdgeModel(nn.Module):
    """
    Exportable SmolLM2-135M Model for LiteRT / TFLite.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.num_layers = config["num_hidden_layers"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config.get("num_key_value_heads", self.num_heads)
        self.eps = config.get("rms_norm_eps", 1e-5)
        self.theta = config.get("rope_theta", 100000.0)

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.rotary = EdgeRotaryEmbedding(self.hidden_size // self.num_heads, max_seq_len=2048, theta=self.theta)
        self.layers = nn.ModuleList([
            EdgeDecoderLayer(self.hidden_size, self.intermediate_size, self.num_heads, self.num_kv_heads, self.eps)
            for _ in range(self.num_layers)
        ])
        self.norm = EdgeRMSNorm(self.hidden_size, self.eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [1, seq_len]
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, self.rotary)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
