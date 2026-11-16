"""Normalization layers and specs: RMSNorm, LayerNorm, QK-Norm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...configs.spec import BuildContext, Spec
from ...kernels._dtype import acc_dtype

__all__ = [
    "NormSpec",
    "RMSNorm",
    "LayerNorm",
    "QKNorm",
    "RMSNormModule",
    "LayerNormModule",
    "GatedRMSNorm",
    "rms_norm",
]


def rms_norm(x: Tensor, weight: Optional[Tensor], eps: float) -> Tensor:
    """RMSNorm computed in float32 and cast back (uses the fused ``F.rms_norm`` op)."""
    shape = (x.shape[-1],)
    acc = acc_dtype(x)
    out = F.rms_norm(x.to(acc), shape, weight.to(acc) if weight is not None else None, eps)
    return out.to(x.dtype)


class RMSNormModule(nn.Module):
    """Root-mean-square normalization (Zhang & Sennrich, 2019).

    ``y = x / sqrt(mean(x^2) + eps) * weight``, computed in float32 for stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}"


class LayerNormModule(nn.Module):
    """Layer normalization (Ba et al., 2016) computed in float32."""

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True, elementwise_affine: bool = True):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None
        self.bias = nn.Parameter(torch.zeros(dim)) if (bias and elementwise_affine) else None

    def forward(self, x: Tensor) -> Tensor:
        acc = acc_dtype(x)
        w = self.weight.to(acc) if self.weight is not None else None
        b = self.bias.to(acc) if self.bias is not None else None
        return F.layer_norm(x.to(acc), (self.dim,), w, b, self.eps).to(x.dtype)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}"


class GatedRMSNorm(nn.Module):
    """``RMSNorm(x * silu(z))`` as used inside Mamba-2 and Gated DeltaNet outputs.

    ``norm_before_gate=True`` gives ``RMSNorm(x) * silu(z)`` instead (Gated DeltaNet / FLA
    ``FusedRMSNormGated``). Normalization can be grouped (``group_size``) as in Mamba-2
    with ``ngroups > 1``.
    """

    def __init__(self, dim: int, eps: float = 1e-6, norm_before_gate: bool = False, group_size: Optional[int] = None):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.norm_before_gate = norm_before_gate
        self.group_size = group_size or dim
        if dim % self.group_size:
            raise ValueError(f"dim {dim} not divisible by group_size {self.group_size}")
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        dtype = x.dtype
        acc = acc_dtype(x)
        x, z = x.to(acc), z.to(acc)
        if not self.norm_before_gate:
            x = x * F.silu(z)
        shape = x.shape
        xg = x.reshape(*shape[:-1], shape[-1] // self.group_size, self.group_size)
        xg = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xg.reshape(shape) * self.weight.to(acc)
        if self.norm_before_gate:
            x = x * F.silu(z)
        return x.to(dtype)


# ----------------------------------------------------------------------------- specs


@dataclass(frozen=True, kw_only=True)
class NormSpec(Spec):
    kind = "norm"

    def build(self, ctx: BuildContext) -> nn.Module:  # pragma: no cover
        raise NotImplementedError

    def build_dim(self, dim: int) -> nn.Module:
        return self.build(BuildContext(dim=dim))


@dataclass(frozen=True, kw_only=True)
class RMSNorm(NormSpec):
    """Spec for :class:`RMSNormModule`."""

    eps: float = 1e-6
    elementwise_affine: bool = True

    def build(self, ctx: BuildContext) -> nn.Module:
        return RMSNormModule(ctx.dim, self.eps, self.elementwise_affine)


@dataclass(frozen=True, kw_only=True)
class LayerNorm(NormSpec):
    """Spec for :class:`LayerNormModule`."""

    eps: float = 1e-5
    bias: bool = True
    elementwise_affine: bool = True

    def build(self, ctx: BuildContext) -> nn.Module:
        return LayerNormModule(ctx.dim, self.eps, self.bias, self.elementwise_affine)


@dataclass(frozen=True, kw_only=True)
class QKNorm(Spec):
    """Normalize queries and keys per head before the dot product (Dehghani et al. 2023;
    used by OLMo-2, Qwen-3, Gemma-3...). ``norm`` is applied over ``head_dim``.

    Pass ``qk_norm=True`` to an attention spec as a shortcut for ``QKNorm(norm=RMSNorm())``.
    """

    kind = "qk_norm"
    norm: NormSpec = field(default_factory=RMSNorm)

    def build(self, ctx: BuildContext) -> nn.Module:
        return self.norm.build(ctx)


def resolve_qk_norm(value) -> Optional[QKNorm]:
    if value is None or value is False:
        return None
    if value is True:
        return QKNorm()
    if isinstance(value, QKNorm):
        return value
    if isinstance(value, NormSpec):
        return QKNorm(norm=value)
    raise TypeError(f"qk_norm must be bool, QKNorm or a norm spec, got {type(value).__name__}")
