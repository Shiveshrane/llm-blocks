"""Mamba-2 sequence mixer (Dao & Gu, "Transformers are SSMs", ICML 2024).

Layer structure follows ``mamba_ssm.modules.mamba2.Mamba2`` (Apache-2.0):
``in_proj -> [z | xBC | dt]``, causal depthwise conv + SiLU on ``xBC``, selective SSM with
scalar-per-head decay ``A``, skip ``D``, gated RMSNorm with ``z``, ``out_proj``.

The scan uses the pure-PyTorch chunked SSD algorithm (:mod:`llm_blocks.kernels.ssd`) for
training / prefill and an O(1) recurrent update for single-token decoding. The fused
Triton kernels from ``mamba_ssm`` are not used (they require CUDA + Triton).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...configs.spec import BuildContext, SpecError
from ...kernels._dtype import acc_dtype
from ...kernels.ssd import ssd_chunked, ssd_step
from ..base import MixerSpec, MixerState, SequenceMixer, mark_residual_output
from ..norm import GatedRMSNorm
from ..position import PositionSpec, check_position
from .conv import CausalDepthwiseConv1d

__all__ = ["Mamba2State", "Mamba2Mixer", "Mamba2"]


class Mamba2State(MixerState):
    """Recurrent state: conv window ``(B, conv_dim, d_conv-1)`` and SSM state ``(B, H, P, N)`` (fp32)."""

    def __init__(self, conv: Tensor, ssm: Tensor):
        self.conv, self.ssm = conv, ssm
        self.offset = 0


class Mamba2Mixer(SequenceMixer):
    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_init_range: Tuple[float, float] = (1.0, 16.0),
        conv_bias: bool = True,
        bias: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        d_inner = expand * dim
        if d_inner % headdim:
            raise SpecError(f"Mamba2: expand*dim = {d_inner} must be divisible by headdim={headdim}")
        nheads = d_inner // headdim
        if nheads % ngroups:
            raise SpecError(f"Mamba2: nheads={nheads} must be divisible by ngroups={ngroups}")
        self.dim, self.d_inner, self.d_state, self.d_conv = dim, d_inner, d_state, d_conv
        self.headdim, self.nheads, self.ngroups, self.chunk_size = headdim, nheads, ngroups, chunk_size
        self.conv_dim = d_inner + 2 * ngroups * d_state

        self.in_proj = nn.Linear(dim, 2 * d_inner + 2 * ngroups * d_state + nheads, bias=bias)
        self.conv1d = CausalDepthwiseConv1d(self.conv_dim, d_conv, bias=conv_bias)

        dt = torch.exp(torch.rand(nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.empty(nheads).uniform_(*A_init_range)))
        self.D = nn.Parameter(torch.ones(nheads))
        for p in (self.dt_bias, self.A_log, self.D):
            p._no_weight_decay = True  # type: ignore[attr-defined]
        self.norm = GatedRMSNorm(d_inner, eps=norm_eps, norm_before_gate=False, group_size=d_inner // ngroups)
        self.out_proj = mark_residual_output(nn.Linear(d_inner, dim, bias=bias))

    def forward(self, x: Tensor, *, state: Optional[Mamba2State] = None, padding_mask: Optional[Tensor] = None) -> Tensor:
        B, T, _ = x.shape
        H, P, G, N = self.nheads, self.headdim, self.ngroups, self.d_state
        acc = acc_dtype(x)
        z, xBC, dt = self.in_proj(x).split([self.d_inner, self.conv_dim, H], dim=-1)
        if padding_mask is not None:
            xBC = xBC * padding_mask[..., None].to(xBC.dtype)
        xBC, new_conv = self.conv1d(xBC, state.conv if state is not None else None)
        xBC = F.silu(xBC)
        xs, Bm, Cm = xBC.split([self.d_inner, G * N, G * N], dim=-1)
        xs = xs.reshape(B, T, H, P)
        Bm = Bm.reshape(B, T, G, N).repeat_interleave(H // G, dim=2)
        Cm = Cm.reshape(B, T, G, N).repeat_interleave(H // G, dim=2)
        dt = F.softplus(dt.to(acc) + self.dt_bias.to(acc))  # (B, T, H)
        if padding_mask is not None:
            dt = dt * padding_mask[..., None].to(acc)  # padded steps: no decay, no input
        A = -torch.exp(self.A_log.to(acc))

        if state is not None and T == 1:
            y, new_ssm = ssd_step(state.ssm, xs[:, 0].to(acc) * dt[:, 0, :, None], A * dt[:, 0], Bm[:, 0], Cm[:, 0])
            y = y.unsqueeze(1)
        else:
            init = state.ssm if state is not None else None
            y, new_ssm = ssd_chunked(xs.to(acc) * dt[..., None], A * dt, Bm, Cm, self.chunk_size, init)
        y = y + xs.to(acc) * self.D.to(acc)[:, None]
        if state is not None:
            if new_conv is not None:
                state.conv = new_conv
            state.ssm = new_ssm
            state.offset += T
        y = self.norm(y.to(x.dtype).reshape(B, T, self.d_inner), z)
        return self.out_proj(y)

    def init_state(self, batch_size, max_seq_len, *, device=None, dtype=None) -> Mamba2State:
        p = self.in_proj.weight
        device, dtype = device or p.device, dtype or p.dtype
        conv = torch.zeros(batch_size, self.conv_dim, self.d_conv - 1, device=device, dtype=dtype)
        ssm = torch.zeros(batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=torch.float64 if dtype == torch.float64 else torch.float32)
        return Mamba2State(conv, ssm)

    def flops_per_token(self, seq_len: int) -> float:
        proj = 2.0 * (self.in_proj.weight.numel() + self.out_proj.weight.numel())
        conv = 2.0 * self.conv_dim * self.d_conv
        # recurrent form: state update (x⊗B, decay) and readout (h·C) ≈ 3 MACs per state entry
        ssm = 2.0 * 3 * self.nheads * self.headdim * self.d_state
        return proj + conv + ssm

    def extra_repr(self) -> str:
        return (f"d_inner={self.d_inner}, nheads={self.nheads}, headdim={self.headdim}, d_state={self.d_state}, "
                f"ngroups={self.ngroups}, d_conv={self.d_conv}, chunk_size={self.chunk_size}")


@dataclass(frozen=True, kw_only=True)
class Mamba2(MixerSpec):
    """Mamba-2 mixer spec (defaults follow ``mamba_ssm`` except ``chunk_size``).

    Mamba-2 carries position information through its recurrence; passing a position
    encoding raises an error.
    """

    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    chunk_size: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1
    A_init_range: Tuple[float, float] = (1.0, 16.0)
    conv_bias: bool = True
    bias: bool = False
    position: Optional[PositionSpec] = None

    def build(self, ctx: BuildContext) -> Mamba2Mixer:
        check_position(self.position, ("none",), "Mamba2")
        return Mamba2Mixer(
            ctx.dim, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand, headdim=self.headdim,
            ngroups=self.ngroups, chunk_size=self.chunk_size, dt_min=self.dt_min, dt_max=self.dt_max,
            A_init_range=tuple(self.A_init_range), conv_bias=self.conv_bias, bias=self.bias,
        )
