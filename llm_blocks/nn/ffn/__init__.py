"""Feed-forward networks: dense MLPs (GELU / ReLU / SiLU) and gated GLU variants
(SwiGLU, GeGLU, ReGLU), plus stacked-weight "expert" versions used by MoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...configs.spec import BuildContext, SpecError
from ..base import FeedForward, FFNSpec, ForwardContext, mark_residual_output

__all__ = [
    "ACTIVATIONS",
    "MLP",
    "GatedMLP",
    "StackedExperts",
    "FFN",
    "GeluFFN",
    "ReluFFN",
    "GLU",
    "SwiGLU",
    "GeGLU",
]

ACTIVATIONS: Dict[str, Callable[[Tensor], Tensor]] = {
    "gelu": lambda x: F.gelu(x, approximate="tanh"),
    "gelu_exact": F.gelu,
    "relu": F.relu,
    "relu2": lambda x: F.relu(x).square(),
    "silu": F.silu,
}


def _act(name: str) -> Callable[[Tensor], Tensor]:
    if name not in ACTIVATIONS:
        raise SpecError(f"Unknown activation {name!r}; choose from {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]


class MLP(FeedForward):
    """``down(act(up(x)))``."""

    def __init__(self, dim: int, hidden_dim: int, activation: str = "gelu", bias: bool = False):
        super().__init__()
        self.activation = activation
        self.act = _act(activation)
        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = mark_residual_output(nn.Linear(hidden_dim, dim, bias=bias))

    def forward(self, x: Tensor, ctx: Optional[ForwardContext] = None) -> Tensor:
        return self.down(self.act(self.up(x)))

    def extra_repr(self) -> str:
        return f"activation={self.activation}"


class GatedMLP(FeedForward):
    """``down(act(gate(x)) * up(x))`` (Shazeer, 2020). Gate and up share one fused GEMM."""

    def __init__(self, dim: int, hidden_dim: int, activation: str = "silu", bias: bool = False):
        super().__init__()
        self.activation, self.hidden_dim = activation, hidden_dim
        self.act = _act(activation)
        self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.down = mark_residual_output(nn.Linear(hidden_dim, dim, bias=bias))

    def forward(self, x: Tensor, ctx: Optional[ForwardContext] = None) -> Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(self.act(gate) * up)

    def extra_repr(self) -> str:
        return f"activation={self.activation}"


class StackedExperts(nn.Module):
    """``E`` independent (gated) MLPs stored as stacked weight tensors.

    * ``forward_batched(x: (E, C, D))`` runs all experts with one ``bmm`` per projection
      (fixed shapes, no host synchronisation — used by capacity-based dispatch).
    * ``forward_expert(e, x: (n, D))`` runs a single expert (used by dropless dispatch).
    """

    def __init__(self, num_experts: int, dim: int, hidden_dim: int, activation: str, gated: bool, init_std: float = 0.02):
        super().__init__()
        self.num_experts, self.dim, self.hidden_dim = num_experts, dim, hidden_dim
        self.gated, self.activation = gated, activation
        self.act = _act(activation)
        up = (2 if gated else 1) * hidden_dim
        self.w_up = nn.Parameter(torch.empty(num_experts, dim, up))
        self.w_down = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w_down._llm_blocks_residual_out = True  # type: ignore[attr-defined]
        self.reset_parameters(init_std, init_std)

    def reset_parameters(self, std: float, out_std: float) -> None:
        nn.init.normal_(self.w_up, std=std)
        nn.init.normal_(self.w_down, std=out_std)

    def _llm_blocks_init(self, std: float, out_std: float) -> None:
        self.reset_parameters(std, out_std)

    def _hidden(self, h: Tensor) -> Tensor:
        if self.gated:
            gate, up = h.chunk(2, dim=-1)
            return self.act(gate) * up
        return self.act(h)

    def forward_batched(self, x: Tensor) -> Tensor:
        w_up, w_down = self.w_up.to(x.dtype), self.w_down.to(x.dtype)
        return torch.bmm(self._hidden(torch.bmm(x, w_up)), w_down)

    def forward_expert(self, e: int, x: Tensor) -> Tensor:
        return self._hidden(x @ self.w_up[e].to(x.dtype)) @ self.w_down[e].to(x.dtype)

    def extra_repr(self) -> str:
        return f"experts={self.num_experts}, dim={self.dim}, hidden={self.hidden_dim}, gated={self.gated}, act={self.activation}"


# ----------------------------------------------------------------------------- specs


def _round_up(x: float, multiple: int) -> int:
    return int(multiple * ((int(x) + multiple - 1) // multiple))


@dataclass(frozen=True, kw_only=True)
class FFN(FFNSpec):
    """Dense two-layer MLP. ``hidden_dim`` defaults to ``4 * dim``."""

    hidden_dim: Optional[int] = None
    activation: str = "gelu"
    bias: bool = False

    def _hidden(self, dim: int) -> int:
        return self.hidden_dim or 4 * dim

    def build(self, ctx: BuildContext) -> MLP:
        return MLP(ctx.dim, self._hidden(ctx.dim), self.activation, self.bias)

    def build_experts(self, ctx: BuildContext, num_experts: int) -> StackedExperts:
        if self.bias:
            raise SpecError("MoE experts do not support bias=True")
        return StackedExperts(num_experts, ctx.dim, self._hidden(ctx.dim), self.activation, False, ctx.init_std)


@dataclass(frozen=True, kw_only=True)
class GeluFFN(FFN):
    """Dense MLP with GELU (tanh approximation, as in GPT-2)."""

    activation: str = "gelu"


@dataclass(frozen=True, kw_only=True)
class ReluFFN(FFN):
    """Dense MLP with ReLU (original Transformer)."""

    activation: str = "relu"


@dataclass(frozen=True, kw_only=True)
class GLU(FFNSpec):
    """Gated linear unit FFN with a configurable activation.

    ``hidden_dim`` defaults to ``8/3 * dim`` rounded up to a multiple of 64, which keeps the
    parameter count equal to a ``4 * dim`` dense MLP.
    """

    hidden_dim: Optional[int] = None
    activation: str = "silu"
    bias: bool = False

    def _hidden(self, dim: int) -> int:
        return self.hidden_dim or _round_up(8 * dim / 3, 64)

    def build(self, ctx: BuildContext) -> GatedMLP:
        return GatedMLP(ctx.dim, self._hidden(ctx.dim), self.activation, self.bias)

    def build_experts(self, ctx: BuildContext, num_experts: int) -> StackedExperts:
        if self.bias:
            raise SpecError("MoE experts do not support bias=True")
        return StackedExperts(num_experts, ctx.dim, self._hidden(ctx.dim), self.activation, True, ctx.init_std)


@dataclass(frozen=True, kw_only=True)
class SwiGLU(GLU):
    """SwiGLU FFN (Llama, Mistral, Qwen...)."""

    activation: str = "silu"


@dataclass(frozen=True, kw_only=True)
class GeGLU(GLU):
    """GeGLU FFN (Gemma, T5 v1.1)."""

    activation: str = "gelu"
