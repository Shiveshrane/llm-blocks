"""Multi-head Latent Attention (DeepSeek-V2, arXiv:2405.04434).

Keys and values are reconstructed from a shared low-rank latent ``c_kv`` (``kv_lora_rank``
channels per token). Position information uses *decoupled* RoPE: a small ``qk_rope_head_dim``
slice of each query head and a single rotary key shared by all heads.

During incremental decoding only ``c_kv`` and the rotary key are cached, i.e.
``kv_lora_rank + qk_rope_head_dim`` numbers per token per layer (independent of the head
count). Decoding uses the *absorbed* formulation: the key up-projection is folded into
the query and the value up-projection is applied after attention, so attention runs
directly over the cached latents as a single-KV-head (MQA-style) attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn

from ...configs.spec import BuildContext, SpecError
from ..base import MixerSpec, MixerState, SequenceMixer, mark_residual_output
from ..norm import RMSNormModule
from ..position import PositionSpec, RoPE, check_position
from .attention import attention_core

__all__ = ["MLACache", "MultiHeadLatentAttention", "MLA"]


class MLACache(MixerState):
    """Compressed MLA cache: normalized latent ``c_kv`` and the post-RoPE shared key."""

    def __init__(self, batch_size, max_len, kv_lora_rank, rope_dim, device, dtype):
        self.c_kv = torch.zeros(batch_size, 1, max_len, kv_lora_rank, device=device, dtype=dtype)
        self.k_pe = torch.zeros(batch_size, 1, max_len, rope_dim, device=device, dtype=dtype)
        self.key_valid = torch.ones(batch_size, max_len, device=device, dtype=torch.bool)
        self.has_padding = False
        self.max_len = max_len
        self.offset = 0

    def update(self, c_kv: Tensor, k_pe: Tensor, padding_mask: Optional[Tensor]):
        t = c_kv.shape[2]
        s, e = self.offset, self.offset + t
        if e > self.max_len:
            raise RuntimeError(f"MLA cache overflow: need {e} positions, allocated {self.max_len}")
        self.c_kv[:, :, s:e] = c_kv.to(self.c_kv.dtype)
        self.k_pe[:, :, s:e] = k_pe.to(self.k_pe.dtype)
        if padding_mask is not None:
            self.key_valid[:, s:e] = padding_mask
            if not self.has_padding:
                self.has_padding = bool((~padding_mask).any())
        self.offset = e
        return self.c_kv[:, :, :e], self.k_pe[:, :, :e], (self.key_valid[:, :e] if self.has_padding else None)


class MultiHeadLatentAttention(SequenceMixer):
    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        kv_lora_rank: int,
        q_lora_rank: Optional[int],
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        position: Optional[PositionSpec],
        max_seq_len: int = 2048,
        absorb_for_decode: bool = True,
        backend: str = "sdpa",
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        kind = check_position(position, ("rotary", "none"), "MLA")
        if kind == "none":
            qk_rope_head_dim = 0
        elif qk_rope_head_dim <= 0 or qk_rope_head_dim % 2:
            raise SpecError(f"MLA: qk_rope_head_dim must be a positive even number with RoPE, got {qk_rope_head_dim}")
        if kind == "rotary" and getattr(position, "rotary_fraction", 1.0) != 1.0:
            raise SpecError("MLA: decoupled RoPE size is set by qk_rope_head_dim; rotary_fraction must be 1.0")
        self.dim, self.heads = dim, heads
        self.kv_lora_rank, self.q_lora_rank = kv_lora_rank, q_lora_rank
        self.nope, self.rope, self.v_dim = qk_nope_head_dim, qk_rope_head_dim, v_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)
        self.absorb_for_decode, self.backend = absorb_for_decode, backend

        if q_lora_rank:
            self.q_a_proj = nn.Linear(dim, q_lora_rank, bias=False)
            self.q_a_norm = RMSNormModule(q_lora_rank, norm_eps)
            self.q_b_proj = nn.Linear(q_lora_rank, heads * self.qk_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(dim, heads * self.qk_head_dim, bias=False)
        self.kv_a_proj = nn.Linear(dim, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_a_norm = RMSNormModule(kv_lora_rank, norm_eps)
        self.kv_b_proj = nn.Linear(kv_lora_rank, heads * (qk_nope_head_dim + v_head_dim), bias=False)
        self.o_proj = mark_residual_output(nn.Linear(heads * v_head_dim, dim, bias=False))
        self.rotary = position.build_rotary(qk_rope_head_dim, max_seq_len) if kind == "rotary" else None  # type: ignore[union-attr]

    def _queries(self, x: Tensor) -> Tensor:
        if self.q_lora_rank:
            return self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        return self.q_proj(x)

    def forward(self, x: Tensor, *, state: Optional[MLACache] = None, padding_mask: Optional[Tensor] = None) -> Tensor:
        B, T, _ = x.shape
        H = self.heads
        q = self._queries(x).view(B, T, H, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = q.split([self.nope, self.rope], dim=-1)
        c_kv, k_pe = self.kv_a_proj(x).split([self.kv_lora_rank, self.rope], dim=-1)
        c_kv = self.kv_a_norm(c_kv).unsqueeze(1)  # (B, 1, T, r)
        k_pe = k_pe.unsqueeze(1)  # (B, 1, T, rope)
        offset = state.offset if state is not None else 0
        if self.rotary is not None:
            q_pe, k_pe = self.rotary(q_pe, k_pe, offset)

        if state is not None:
            c_all, k_pe_all, key_valid = state.update(c_kv, k_pe, padding_mask)
        else:
            c_all, k_pe_all, key_valid = c_kv, k_pe, padding_mask

        if state is not None and self.absorb_for_decode:
            out = self._absorbed(q_nope, q_pe, c_all, k_pe_all, offset, key_valid)
        else:
            out = self._expanded(q_nope, q_pe, c_all, k_pe_all, offset, key_valid)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * self.v_dim))

    def _expanded(self, q_nope, q_pe, c_all, k_pe_all, offset, key_valid) -> Tensor:
        B, H, T, _ = q_nope.shape
        S = c_all.shape[2]
        kv = self.kv_b_proj(c_all.squeeze(1)).view(B, S, H, self.nope + self.v_dim).transpose(1, 2)
        k_nope, v = kv.split([self.nope, self.v_dim], dim=-1)
        k = torch.cat([k_nope, k_pe_all.expand(B, H, S, self.rope)], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        return attention_core(q, k, v, offset=offset, key_valid=key_valid, scale=self.scale, backend=self.backend)

    def _absorbed(self, q_nope, q_pe, c_all, k_pe_all, offset, key_valid) -> Tensor:
        H = self.heads
        w = self.kv_b_proj.weight.view(H, self.nope + self.v_dim, self.kv_lora_rank)
        w_uk, w_uv = w[:, : self.nope], w[:, self.nope :]
        q_lat = torch.einsum("bhtd,hdr->bhtr", q_nope, w_uk)
        q_cat = torch.cat([q_lat, q_pe], dim=-1)  # (B, H, T, r + rope)
        k_cat = torch.cat([c_all, k_pe_all], dim=-1)  # (B, 1, S, r + rope): one shared "KV head"
        o_lat = attention_core(q_cat, k_cat, c_all, offset=offset, key_valid=key_valid, scale=self.scale,
                               backend=self.backend)
        return torch.einsum("bhtr,hdr->bhtd", o_lat, w_uv)

    def init_state(self, batch_size, max_seq_len, *, device=None, dtype=None) -> MLACache:
        p = self.kv_a_proj.weight
        return MLACache(batch_size, max_seq_len, self.kv_lora_rank, self.rope, device or p.device, dtype or p.dtype)

    def flops_per_token(self, seq_len: int) -> float:
        proj = 2.0 * sum(p.numel() for p in self.parameters() if p.dim() >= 2)
        avg_ctx = (seq_len + 1) / 2.0
        return proj + 2.0 * self.heads * (self.qk_head_dim + self.v_dim) * avg_ctx

    def extra_repr(self) -> str:
        return (f"heads={self.heads}, kv_lora_rank={self.kv_lora_rank}, q_lora_rank={self.q_lora_rank}, "
                f"qk_nope={self.nope}, qk_rope={self.rope}, v_head_dim={self.v_dim}")


@dataclass(frozen=True, kw_only=True)
class MLA(MixerSpec):
    """Multi-head Latent Attention spec.

    Unset dimensions are derived from ``head_dim = dim // heads`` following the
    proportions of DeepSeek-V2 (``kv_lora_rank = 4 * head_dim``, ``qk_rope = head_dim / 2``).

    Args:
        heads: number of attention heads.
        kv_lora_rank: latent (cache) width per token.
        q_lora_rank: optional low-rank query bottleneck (``None`` = full-rank query projection).
        qk_nope_head_dim / qk_rope_head_dim / v_head_dim: per-head dimensions.
        position: ``RoPE()`` (decoupled RoPE) or ``None``/``NoPE()``. ALiBi is not supported.
        absorb_for_decode: use the weight-absorbed formulation when decoding with a cache.
    """

    heads: int
    kv_lora_rank: Optional[int] = None
    q_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    position: Optional[PositionSpec] = field(default_factory=RoPE)
    absorb_for_decode: bool = True
    backend: str = "sdpa"

    def build(self, ctx: BuildContext) -> MultiHeadLatentAttention:
        hd = ctx.dim // self.heads
        if hd == 0:
            raise SpecError(f"MLA: heads={self.heads} exceeds model dim {ctx.dim}")
        return MultiHeadLatentAttention(
            ctx.dim, self.heads,
            kv_lora_rank=self.kv_lora_rank or 4 * hd,
            q_lora_rank=self.q_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim or hd,
            qk_rope_head_dim=self.qk_rope_head_dim if self.qk_rope_head_dim is not None else max(2, hd // 2 // 2 * 2),
            v_head_dim=self.v_head_dim or hd,
            position=self.position, max_seq_len=ctx.max_seq_len,
            absorb_for_decode=self.absorb_for_decode, backend=self.backend,
        )
