"""Softmax attention with a configurable number of key/value heads (MHA / GQA / MQA)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...configs.spec import BuildContext, SpecError
from ...kernels._dtype import acc_dtype
from ..base import MixerSpec, MixerState, SequenceMixer, mark_residual_output
from ..norm import QKNorm, resolve_qk_norm
from ..position import PositionSpec, RoPE, check_position

__all__ = ["KVCache", "Attention", "MHA", "GQA", "MQA", "attention_core"]


class KVCache(MixerState):
    """Pre-allocated key/value cache for one attention layer.

    Shapes: ``k: (batch, kv_heads, max_len, head_dim)``, ``v: (batch, kv_heads, max_len, v_dim)``.
    ``key_valid`` records padding so left-padded batched prompts are handled exactly.
    """

    def __init__(self, batch_size: int, kv_heads: int, max_len: int, head_dim: int, v_dim: int, device, dtype):
        self.k = torch.zeros(batch_size, kv_heads, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(batch_size, kv_heads, max_len, v_dim, device=device, dtype=dtype)
        self.key_valid = torch.ones(batch_size, max_len, device=device, dtype=torch.bool)
        self.has_padding = False
        self.max_len = max_len
        self.offset = 0

    def update(self, k: Tensor, v: Tensor, padding_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        t = k.shape[2]
        start, end = self.offset, self.offset + t
        if end > self.max_len:
            raise RuntimeError(f"KV cache overflow: need {end} positions, cache was allocated for {self.max_len}")
        self.k[:, :, start:end] = k.to(self.k.dtype)
        self.v[:, :, start:end] = v.to(self.v.dtype)
        if padding_mask is not None:
            self.key_valid[:, start:end] = padding_mask
            if not self.has_padding:
                self.has_padding = bool((~padding_mask).any())
        self.offset = end
        kv = self.key_valid[:, :end] if self.has_padding else None
        return self.k[:, :, :end], self.v[:, :, :end], kv


def attention_core(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    offset: int,
    key_valid: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    scale: Optional[float] = None,
    backend: str = "sdpa",
) -> Tensor:
    """Causal attention between ``T`` new queries (positions ``offset..offset+T-1``) and
    ``S`` keys (positions ``0..S-1``).

    Args:
        q: ``(B, H, T, d)``; k: ``(B, Hkv, S, d)``; v: ``(B, Hkv, S, dv)`` with ``H % Hkv == 0``.
        key_valid: optional ``(B, S)`` bool padding mask.
        bias: optional additive ``(H, T, S)`` float bias (ALiBi).
        backend: ``"sdpa"`` (PyTorch fused SDPA: FlashAttention / memory-efficient kernels
            on CUDA) or ``"eager"`` (explicit float32 math, a reference implementation).
    """
    B, H, T, d = q.shape
    S, Hkv = k.shape[2], k.shape[1]
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    gqa = H != Hkv

    simple = key_valid is None and bias is None
    if backend == "sdpa" and simple and (S == T or T == 1):
        # Fast paths without materialized masks: full causal prefill, or single-token decode.
        return F.scaled_dot_product_attention(q, k, v, is_causal=(S == T and T > 1), scale=scale, enable_gqa=gqa)

    q_pos = torch.arange(offset, offset + T, device=q.device)
    k_pos = torch.arange(S, device=q.device)
    allowed = k_pos[None, :] <= q_pos[:, None]  # (T, S)
    if key_valid is not None:
        # Always let a query see itself so fully-padded rows don't produce NaNs.
        allowed = (allowed[None] & key_valid[:, None, :]) | (k_pos[None, None, :] == q_pos[None, :, None])
        allowed = allowed[:, None]  # (B, 1, T, S)
    else:
        allowed = allowed[None, None]  # (1, 1, T, S)

    if backend == "eager":
        if gqa:
            k = k.repeat_interleave(H // Hkv, dim=1)
            v = v.repeat_interleave(H // Hkv, dim=1)
        acc = acc_dtype(q)
        scores = torch.matmul(q.to(acc), k.to(acc).transpose(-1, -2)) * scale
        if bias is not None:
            scores = scores + bias[None].to(acc)
        scores = scores.masked_fill(~allowed, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v.to(acc)).to(q.dtype)

    if backend != "sdpa":
        raise ValueError(f"Unknown attention backend {backend!r}")
    if bias is not None:
        mask = torch.zeros(allowed.shape, device=q.device, dtype=q.dtype).masked_fill(~allowed, float("-inf"))
        mask = mask + bias[None].to(q.dtype)
    else:
        mask = allowed
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa)


class Attention(SequenceMixer):
    """Multi-head softmax attention with ``kv_heads`` key/value heads.

    ``kv_heads == heads`` is MHA, ``kv_heads == 1`` is MQA, anything in between is GQA.
    Q, K and V come from a single fused projection (one GEMM).
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        *,
        position: Optional[PositionSpec] = None,
        qk_norm: Optional[QKNorm] = None,
        bias: bool = False,
        max_seq_len: int = 2048,
        backend: str = "sdpa",
        name: str = "Attention",
    ):
        super().__init__()
        if heads % kv_heads:
            raise SpecError(f"{name}: heads ({heads}) must be divisible by kv_heads ({kv_heads})")
        self.dim, self.heads, self.kv_heads, self.head_dim = dim, heads, kv_heads, head_dim
        self.backend, self.name = backend, name
        self.qkv_proj = nn.Linear(dim, (heads + 2 * kv_heads) * head_dim, bias=bias)
        self.o_proj = mark_residual_output(nn.Linear(heads * head_dim, dim, bias=bias))
        kind = check_position(position, ("rotary", "bias", "none"), name)
        self.rotary = position.build_rotary(head_dim, max_seq_len) if kind == "rotary" else None  # type: ignore[union-attr]
        self.alibi = position.build_bias(heads) if kind == "bias" else None  # type: ignore[union-attr]
        self.q_norm = qk_norm.build(BuildContext(dim=head_dim)) if qk_norm else None
        self.k_norm = qk_norm.build(BuildContext(dim=head_dim)) if qk_norm else None

    def forward(self, x: Tensor, *, state: Optional[KVCache] = None, padding_mask: Optional[Tensor] = None) -> Tensor:
        B, T, _ = x.shape
        hd = self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.heads * hd, self.kv_heads * hd, self.kv_heads * hd], dim=-1)
        q = q.view(B, T, self.heads, hd).transpose(1, 2)
        k = k.view(B, T, self.kv_heads, hd).transpose(1, 2)
        v = v.view(B, T, self.kv_heads, hd).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)  # type: ignore[misc]
        offset = state.offset if state is not None else 0
        if self.rotary is not None:
            q, k = self.rotary(q, k, offset)
        if state is not None:
            k, v, key_valid = state.update(k, v, padding_mask)
        else:
            key_valid = padding_mask
        bias = self.alibi.bias(T, k.shape[2], offset, x.device) if self.alibi is not None else None
        out = attention_core(q, k, v, offset=offset, key_valid=key_valid, bias=bias, backend=self.backend)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, self.heads * hd))

    def init_state(self, batch_size, max_seq_len, *, device=None, dtype=None) -> KVCache:
        p = self.qkv_proj.weight
        return KVCache(batch_size, self.kv_heads, max_seq_len, self.head_dim, self.head_dim,
                       device or p.device, dtype or p.dtype)

    def flops_per_token(self, seq_len: int) -> float:
        proj = 2.0 * (self.qkv_proj.weight.numel() + self.o_proj.weight.numel())
        avg_ctx = (seq_len + 1) / 2.0  # causal: token i attends to i+1 keys
        return proj + 2.0 * 2.0 * self.heads * self.head_dim * avg_ctx

    def extra_repr(self) -> str:
        pos = "rope" if self.rotary is not None else ("alibi" if self.alibi is not None else "none")
        return (f"heads={self.heads}, kv_heads={self.kv_heads}, head_dim={self.head_dim}, "
                f"position={pos}, qk_norm={self.q_norm is not None}, backend={self.backend}")


# ----------------------------------------------------------------------------- specs


def _head_dim(dim: int, heads: int, head_dim: Optional[int], name: str) -> int:
    if head_dim is not None:
        return head_dim
    if dim % heads:
        raise SpecError(f"{name}: model dim {dim} is not divisible by heads={heads}; pass head_dim explicitly")
    return dim // heads


@dataclass(frozen=True, kw_only=True)
class _AttentionSpec(MixerSpec):
    heads: int
    head_dim: Optional[int] = None
    position: Optional[PositionSpec] = field(default_factory=RoPE)
    qk_norm: object = False
    bias: bool = False
    backend: str = "sdpa"

    def _kv_heads(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def build(self, ctx: BuildContext) -> Attention:
        name = type(self).__name__
        if self.backend not in ("sdpa", "eager"):
            raise SpecError(f"{name}: backend must be 'sdpa' or 'eager', got {self.backend!r}")
        return Attention(
            ctx.dim, self.heads, self._kv_heads(), _head_dim(ctx.dim, self.heads, self.head_dim, name),
            position=self.position, qk_norm=resolve_qk_norm(self.qk_norm), bias=self.bias,
            max_seq_len=ctx.max_seq_len, backend=self.backend, name=name,
        )


@dataclass(frozen=True, kw_only=True)
class MHA(_AttentionSpec):
    """Multi-head attention (Vaswani et al., 2017): one K/V head per query head."""

    def _kv_heads(self) -> int:
        return self.heads


@dataclass(frozen=True, kw_only=True)
class GQA(_AttentionSpec):
    """Grouped-query attention (Ainslie et al., 2023): ``kv_heads`` shared K/V heads."""

    kv_heads: int

    def _kv_heads(self) -> int:
        return self.kv_heads


@dataclass(frozen=True, kw_only=True)
class MQA(_AttentionSpec):
    """Multi-query attention (Shazeer, 2019): a single K/V head shared by all query heads."""

    def _kv_heads(self) -> int:
        return 1
