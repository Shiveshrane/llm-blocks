"""Gated DeltaNet sequence mixer (Yang, Kautz & Hatamizadeh, ICLR 2025, arXiv:2412.06464).

Layer structure follows the ``GatedDeltaNet`` layer of flash-linear-attention (MIT):
projections for q, k, v, short causal convolutions + SiLU on q/k/v, per-head L2-normalised
q and k, ``beta = sigmoid(W_b x)``, Mamba-2-style decay ``g = -exp(A_log) * softplus(W_a x + dt_bias)``,
gated delta rule, per-head RMSNorm gated by ``silu(W_g x)``, output projection.

Training / prefill use the chunked WY-representation algorithm in pure PyTorch
(:mod:`llm_blocks.kernels.delta_rule`); decoding uses the O(1) recurrent update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...configs.spec import BuildContext, SpecError
from ...kernels._dtype import acc_dtype
from ...kernels.delta_rule import gated_delta_rule_chunked, gated_delta_rule_step
from ..base import MixerSpec, MixerState, SequenceMixer, mark_residual_output
from ..norm import GatedRMSNorm
from ..position import PositionSpec, check_position
from .conv import CausalDepthwiseConv1d

__all__ = ["GatedDeltaNetState", "GatedDeltaNetMixer", "GatedDeltaNet"]


class GatedDeltaNetState(MixerState):
    """Conv window ``(B, 2*Dk + Dv, d_conv-1)`` and fast-weight state ``(B, H, dk, dv)`` (fp32)."""

    def __init__(self, conv: Tensor, S: Tensor):
        self.conv, self.S = conv, S
        self.offset = 0


def _l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


class GatedDeltaNetMixer(SequenceMixer):
    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        head_dim: int,
        expand_v: float = 2.0,
        d_conv: int = 4,
        chunk_size: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.dim, self.heads, self.head_k = dim, heads, head_dim
        self.head_v = int(head_dim * expand_v)
        if self.head_v <= 0:
            raise SpecError("GatedDeltaNet: expand_v * head_dim must be positive")
        self.key_dim, self.value_dim = heads * head_dim, heads * self.head_v
        self.chunk_size = chunk_size
        self.conv_dim = 2 * self.key_dim + self.value_dim
        # fused projection: [q | k | v | gate | a | b]
        self.in_proj = nn.Linear(dim, self.conv_dim + self.value_dim + 2 * heads, bias=False)
        self.conv1d = CausalDepthwiseConv1d(self.conv_dim, d_conv, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(heads).uniform_(0, 16)))
        dt = torch.exp(torch.rand(heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        for p in (self.A_log, self.dt_bias):
            p._no_weight_decay = True  # type: ignore[attr-defined]
        self.o_norm = GatedRMSNorm(self.head_v, eps=norm_eps, norm_before_gate=True)
        self.o_proj = mark_residual_output(nn.Linear(self.value_dim, dim, bias=False))
        self.d_conv = d_conv

    def forward(self, x: Tensor, *, state: Optional[GatedDeltaNetState] = None, padding_mask: Optional[Tensor] = None) -> Tensor:
        B, T, _ = x.shape
        H, dk, dv = self.heads, self.head_k, self.head_v
        acc = acc_dtype(x)
        qkv, gate, a, b = self.in_proj(x).split([self.conv_dim, self.value_dim, H, H], dim=-1)
        if padding_mask is not None:
            qkv = qkv * padding_mask[..., None].to(qkv.dtype)
        qkv, new_conv = self.conv1d(qkv, state.conv if state is not None else None)
        qkv = F.silu(qkv)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = _l2norm(q.to(acc).view(B, T, H, dk)) * (dk ** -0.5)
        k = _l2norm(k.to(acc).view(B, T, H, dk))
        v = v.to(acc).view(B, T, H, dv)
        beta = torch.sigmoid(b.to(acc))
        g = -torch.exp(self.A_log.to(acc)) * F.softplus(a.to(acc) + self.dt_bias.to(acc))
        if padding_mask is not None:
            m = padding_mask.to(acc)[..., None]
            beta, g = beta * m, g * m  # padded steps leave the state unchanged

        if state is not None and T == 1:
            o, S = gated_delta_rule_step(state.S, q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0])
            o = o.unsqueeze(1)
        else:
            o, S = gated_delta_rule_chunked(q, k, v, g, beta, self.chunk_size,
                                            state.S if state is not None else None)
        if state is not None:
            if new_conv is not None:
                state.conv = new_conv
            state.S = S
            state.offset += T
        o = self.o_norm(o.to(x.dtype), gate.view(B, T, H, dv))
        return self.o_proj(o.reshape(B, T, self.value_dim))

    def init_state(self, batch_size, max_seq_len, *, device=None, dtype=None) -> GatedDeltaNetState:
        p = self.in_proj.weight
        device, dtype = device or p.device, dtype or p.dtype
        conv = torch.zeros(batch_size, self.conv_dim, self.d_conv - 1, device=device, dtype=dtype)
        S = torch.zeros(batch_size, self.heads, self.head_k, self.head_v, device=device, dtype=torch.float64 if dtype == torch.float64 else torch.float32)
        return GatedDeltaNetState(conv, S)

    def flops_per_token(self, seq_len: int) -> float:
        proj = 2.0 * (self.in_proj.weight.numel() + self.o_proj.weight.numel())
        conv = 2.0 * self.conv_dim * self.d_conv
        # recurrent form: decay, k^T S, rank-1 update, q^T S  ≈ 4 MACs per state entry
        return proj + conv + 2.0 * 4 * self.heads * self.head_k * self.head_v

    def extra_repr(self) -> str:
        return f"heads={self.heads}, head_k={self.head_k}, head_v={self.head_v}, chunk_size={self.chunk_size}"


@dataclass(frozen=True, kw_only=True)
class GatedDeltaNet(MixerSpec):
    """Gated DeltaNet mixer spec.

    Args:
        heads: number of heads (default: ``round(0.75 * dim / head_dim)``, FLA's proportions).
        head_dim: key/query head dimension.
        expand_v: value head dim = ``expand_v * head_dim`` (FLA default 2).
    """

    heads: Optional[int] = None
    head_dim: int = 64
    expand_v: float = 2.0
    d_conv: int = 4
    chunk_size: int = 64
    position: Optional[PositionSpec] = None

    def build(self, ctx: BuildContext) -> GatedDeltaNetMixer:
        check_position(self.position, ("none",), "GatedDeltaNet")
        heads = self.heads or max(1, round(0.75 * ctx.dim / self.head_dim))
        return GatedDeltaNetMixer(ctx.dim, heads=heads, head_dim=self.head_dim, expand_v=self.expand_v,
                                  d_conv=self.d_conv, chunk_size=self.chunk_size)
