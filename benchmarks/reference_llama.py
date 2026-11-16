"""A plain, monolithic Llama-style model written directly in PyTorch.

This is the "standard implementation" baseline for overhead benchmarks: no specs, no
abstractions, separate q/k/v projections, ``torch.nn.RMSNorm``, SDPA attention and a
growing-tensor KV cache — the way such a model is typically hand-written.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def rope_tables(head_dim: int, max_len: int, theta: float = 10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    f = torch.outer(torch.arange(max_len, dtype=torch.float64), inv)
    emb = torch.cat([f, f], -1)
    return emb.cos().float(), emb.sin().float()


def rotate(x, cos, sin):
    x1, x2 = x.chunk(2, -1)
    return (x.float() * cos + torch.cat([-x2, x1], -1).float() * sin).to(x.dtype)


class Attn(nn.Module):
    def __init__(self, dim, heads, kv_heads):
        super().__init__()
        self.h, self.kvh, self.hd = heads, kv_heads, dim // heads
        self.q = nn.Linear(dim, heads * self.hd, bias=False)
        self.k = nn.Linear(dim, kv_heads * self.hd, bias=False)
        self.v = nn.Linear(dim, kv_heads * self.hd, bias=False)
        self.o = nn.Linear(heads * self.hd, dim, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.kvh, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.kvh, self.hd).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], 2)
                v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        causal = k.shape[2] == T and T > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=self.h != self.kvh)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Layer(nn.Module):
    def __init__(self, dim, heads, kv_heads, hidden, eps):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(dim, eps=eps), nn.RMSNorm(dim, eps=eps)
        self.attn, self.mlp = Attn(dim, heads, kv_heads), MLP(dim, hidden)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.n1(x), cos, sin, cache)
        return x + self.mlp(self.n2(x))


class RefLlama(nn.Module):
    def __init__(self, vocab, dim, depth, heads, kv_heads, hidden, max_len=4096, eps=1e-6):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([Layer(dim, heads, kv_heads, hidden, eps) for _ in range(depth)])
        self.norm = nn.RMSNorm(dim, eps=eps)
        self.head = nn.Linear(dim, vocab, bias=False)
        cos, sin = rope_tables(dim // heads, max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, labels=None, caches=None, pos=0):
        T = idx.shape[1]
        cos, sin = self.cos[pos:pos + T], self.sin[pos:pos + T]
        x = self.emb(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, caches[i] if caches is not None else None)
        logits = self.head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        return logits, loss


def load_from_llm_blocks(ref: RefLlama, model) -> None:
    """Copy weights from an equivalent llm_blocks model (GQA + RoPE + SwiGLU + RMSNorm)."""
    sd = {"emb.weight": model.embed.weight, "head.weight": model.lm_head.weight, "norm.weight": model.final_norm.weight}
    for i, L in enumerate(model.layers):
        a = L.mixer
        q, k, v = a.qkv_proj.weight.split([a.heads * a.head_dim, a.kv_heads * a.head_dim, a.kv_heads * a.head_dim])
        g, u = L.ffn.gate_up.weight.chunk(2)
        sd.update({f"layers.{i}.attn.q.weight": q, f"layers.{i}.attn.k.weight": k, f"layers.{i}.attn.v.weight": v,
                   f"layers.{i}.attn.o.weight": a.o_proj.weight, f"layers.{i}.mlp.gate.weight": g,
                   f"layers.{i}.mlp.up.weight": u, f"layers.{i}.mlp.down.weight": L.ffn.down.weight,
                   f"layers.{i}.n1.weight": L.norm1.weight, f"layers.{i}.n2.weight": L.norm2.weight})
    ref.load_state_dict({k: v.detach().clone() for k, v in sd.items()})
