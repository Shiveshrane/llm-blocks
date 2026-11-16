"""Residual strategies: how a sublayer's output is merged into the residual stream.

A residual mechanism has two parts:

* a **stream type** (model level): how the embedding becomes the residual stream and how
  the final stream is read out (:class:`ResidualStream`). For ordinary residuals the stream
  is a ``(B, T, D)`` tensor; for hyper-connections it is ``n`` parallel copies
  ``(B, T, n, D)``; for attention residuals it is the list of all previous layer outputs.
* a **strategy** (per sublayer): ``strategy(stream, fn) -> new_stream`` where ``fn`` is the
  (pre-normalised) sublayer (:class:`ResidualStrategy`).

All blocks in a model must use strategies with the same stream type; ``nn.LLM`` checks this.

Implemented:
    ``Residual``          x + f(x)
    ``GatedResidual``     x + g ⊙ f(x), g a learned scalar/vector or an input-dependent sigmoid gate
    ``HyperConnections``  dynamic hyper-connections (Zhu et al., ICLR 2025, arXiv:2409.19606)
    ``MHC``               manifold-constrained hyper-connections (DeepSeek-AI, arXiv:2512.24880):
                          doubly-stochastic (Sinkhorn-Knopp) residual mixing, sigmoid pre/post maps
    ``AttnRes``           attention residuals (Kimi Team, arXiv:2603.15031, "Full AttnRes"):
                          softmax attention over all previous layer outputs with a learned
                          per-layer pseudo-query
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
from torch import Tensor, nn

from ...configs.spec import BuildContext, Spec, SpecError
from ...kernels._dtype import acc_dtype
from ..norm import rms_norm

__all__ = [
    "ResidualStream",
    "ResidualStrategy",
    "ResidualSpec",
    "Residual",
    "GatedResidual",
    "HyperConnections",
    "MHC",
    "AttnRes",
    "sinkhorn_knopp",
]

SublayerFn = Callable[[Tensor], Tensor]


class ResidualStream(nn.Module):
    """Model-level conversion between hidden states and the residual stream."""

    def init(self, x: Tensor):
        return x

    def finalize(self, stream) -> Tensor:
        return stream


class ResidualStrategy(nn.Module):
    """Per-sublayer residual merge: ``forward(stream, fn) -> stream``."""

    stream_key: Tuple = ("tensor",)

    def forward(self, stream, fn: SublayerFn):  # pragma: no cover - abstract
        raise NotImplementedError


# ----------------------------------------------------------------------------- standard


class StandardResidual(ResidualStrategy):
    def forward(self, stream: Tensor, fn: SublayerFn) -> Tensor:
        return stream + fn(stream)


class GatedResidualModule(ResidualStrategy):
    """``x + g ⊙ f(x)``.

    ``gate="scalar"``: one learned scalar; ``"channel"``: learned per-channel vector
    (LayerScale / ReZero when ``init=0``); ``"input"``: ``g = sigmoid(W x + b)`` per channel,
    ``W`` zero-initialised and ``b = logit(init)`` so training starts at ``g = init``.
    """

    def __init__(self, dim: int, gate: str = "channel", init: float = 1.0):
        super().__init__()
        self.gate_type = gate
        if gate == "scalar":
            self.gate = nn.Parameter(torch.full((1,), float(init)))
        elif gate == "channel":
            self.gate = nn.Parameter(torch.full((dim,), float(init)))
        elif gate == "input":
            if not 0.0 < init < 1.0:
                raise SpecError("GatedResidual(gate='input') needs 0 < init < 1 (sigmoid gate value at init)")
            self.proj = nn.Linear(dim, dim)
            self.proj._llm_blocks_skip_init = True  # type: ignore[attr-defined]
            nn.init.zeros_(self.proj.weight)
            nn.init.constant_(self.proj.bias, math.log(init / (1 - init)))
        else:
            raise SpecError(f"GatedResidual: gate must be 'scalar', 'channel' or 'input', got {gate!r}")

    def forward(self, stream: Tensor, fn: SublayerFn) -> Tensor:
        out = fn(stream)
        if self.gate_type == "input":
            g = torch.sigmoid(self.proj(stream))
        else:
            g = self.gate.to(out.dtype)
        return stream + g * out

    def extra_repr(self) -> str:
        return f"gate={self.gate_type}"


# ----------------------------------------------------------------------------- hyper-connections


class HyperStream(ResidualStream):
    """``n`` copies of the hidden state: init replicates, finalize sums (Zhu et al.)."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def init(self, x: Tensor) -> Tensor:
        return x.unsqueeze(-2).expand(*x.shape[:-1], self.n, x.shape[-1]).contiguous()

    def finalize(self, stream: Tensor) -> Tensor:
        return stream.sum(dim=-2)


class HyperConnectionModule(ResidualStrategy):
    """Dynamic hyper-connections (DHC).

    With stream ``H ∈ R^{n×D}`` and ``H̄ = RMSNorm(H)`` (per stream):

        A_m = A_m0 + s_a * tanh(H̄ w_m)       (n,)    width connection: layer input = A_m^T H
        A_r = A_r0 + s_a * tanh(H̄ W_r)       (n, n)  residual mixing
        B   = B_0  + s_b * tanh(H̄ w_b)       (n,)    depth connection: where the output goes
        H'  = A_r^T H + B^T f(A_m^T H)

    Static parts initialise to ``A_m0 = e_{layer mod n}``, ``A_r0 = I``, ``B_0 = 1`` and the
    dynamic projections to zero, so an n=1 network reduces to a standard pre-norm residual.
    """

    def __init__(self, dim: int, n: int, layer_id: int, dynamic: bool = True, scale_init: float = 0.01):
        super().__init__()
        self.n, self.dynamic = n, dynamic
        self.stream_key = ("hyper", n)
        am = torch.zeros(n)
        am[layer_id % n] = 1.0
        self.static_am = nn.Parameter(am)
        self.static_ar = nn.Parameter(torch.eye(n))
        self.static_b = nn.Parameter(torch.ones(n))
        if dynamic:
            self.norm_weight = nn.Parameter(torch.ones(dim))
            self.w_m = nn.Parameter(torch.zeros(dim, 1))
            self.w_r = nn.Parameter(torch.zeros(dim, n))
            self.w_b = nn.Parameter(torch.zeros(dim, 1))
            self.s_a = nn.Parameter(torch.tensor(scale_init))
            self.s_b = nn.Parameter(torch.tensor(scale_init))

    def forward(self, H: Tensor, fn: SublayerFn) -> Tensor:
        acc = acc_dtype(H)
        Hf = H.to(acc)
        am, ar, b = self.static_am, self.static_ar, self.static_b
        if self.dynamic:
            Hn = rms_norm(Hf, self.norm_weight, 1e-6)
            am = am + self.s_a * torch.tanh(Hn @ self.w_m).squeeze(-1)  # (..., n)
            ar = ar + self.s_a * torch.tanh(Hn @ self.w_r)  # (..., n, n): [i, j] from stream i to stream j
            b = b + self.s_b * torch.tanh(Hn @ self.w_b).squeeze(-1)
            h_in = torch.einsum("...i,...id->...d", am, Hf)
            mixed = torch.einsum("...ij,...id->...jd", ar, Hf)
        else:
            h_in = torch.einsum("i,...id->...d", am, Hf)
            mixed = torch.einsum("ij,...id->...jd", ar, Hf)
        out = fn(h_in.to(H.dtype)).to(acc)
        return (mixed + b.unsqueeze(-1) * out.unsqueeze(-2)).to(H.dtype)

    def extra_repr(self) -> str:
        return f"n={self.n}, dynamic={self.dynamic}"


def sinkhorn_knopp(logits: Tensor, iters: int = 20) -> Tensor:
    """Project ``exp(logits)`` (``(..., n, n)``) towards a doubly-stochastic matrix by
    alternating row and column normalisation."""
    M = torch.exp(logits - logits.amax(dim=(-1, -2), keepdim=True))
    for _ in range(iters):
        M = M / M.sum(dim=-1, keepdim=True)
        M = M / M.sum(dim=-2, keepdim=True)
    return M


class MHCModule(ResidualStrategy):
    """Manifold-constrained hyper-connections (mHC).

    For stream ``x ∈ R^{n×C}`` and ``x' = RMSNorm(vec(x)) ∈ R^{nC}``:

        H_pre  = sigmoid(a_pre  * (x' φ_pre)  + b_pre)            (1×n)
        H_post = 2 sigmoid(a_post * (x' φ_post) + b_post)         (1×n)
        H_res  = SinkhornKnopp(a_res * mat(x' φ_res) + b_res)     (n×n, doubly stochastic)
        x_next = H_res x + H_post^T f(H_pre x)

    Initialisation (our choice; see docs/residuals.md): ``φ = 0``, ``a = 0.01``,
    ``b_pre = logit(1/n)`` (layer input = stream mean), ``b_post = 0`` (``H_post = 1``),
    ``b_res`` = 0 on the diagonal and ``-res_offdiag_init`` elsewhere (near-identity mixing).
    """

    def __init__(self, dim: int, n: int, sinkhorn_iters: int = 20, alpha_init: float = 0.01,
                 res_offdiag_init: float = 8.0):
        super().__init__()
        self.n, self.iters = n, sinkhorn_iters
        self.stream_key = ("hyper", n)
        self.norm_weight = nn.Parameter(torch.ones(n * dim))
        self.phi = nn.Parameter(torch.zeros(n * dim, 2 * n + n * n))
        self.alpha = nn.Parameter(torch.full((3,), float(alpha_init)))
        self.b_pre = nn.Parameter(torch.full((n,), math.log((1 / n) / (1 - 1 / n)) if n > 1 else 10.0))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(-res_offdiag_init * (1 - torch.eye(n)))

    def coefficients(self, x: Tensor):
        n = self.n
        acc = acc_dtype(x)
        xf = x.to(acc).flatten(-2)  # (..., n*C)
        xn = rms_norm(xf, self.norm_weight, 1e-6)
        proj = xn @ self.phi
        p_pre, p_post, p_res = proj.split([n, n, n * n], dim=-1)
        h_pre = torch.sigmoid(self.alpha[0] * p_pre + self.b_pre)
        h_post = 2 * torch.sigmoid(self.alpha[1] * p_post + self.b_post)
        h_res = sinkhorn_knopp(self.alpha[2] * p_res.unflatten(-1, (n, n)) + self.b_res, self.iters)
        return h_pre, h_post, h_res

    def forward(self, x: Tensor, fn: SublayerFn) -> Tensor:
        h_pre, h_post, h_res = self.coefficients(x)
        acc = acc_dtype(x)
        xf = x.to(acc)
        h_in = torch.einsum("...j,...jd->...d", h_pre, xf)
        out = fn(h_in.to(x.dtype)).to(acc)
        mixed = torch.einsum("...ij,...jd->...id", h_res, xf)
        return (mixed + h_post.unsqueeze(-1) * out.unsqueeze(-2)).to(x.dtype)

    def extra_repr(self) -> str:
        return f"n={self.n}, sinkhorn_iters={self.iters}"


# ----------------------------------------------------------------------------- attention residuals


def _depth_attention(values: List[Tensor], query: Tensor) -> Tensor:
    acc = acc_dtype(values[0])
    V = torch.stack(values, dim=0).to(acc)  # (L, ..., D)
    K = rms_norm(V, None, 1e-6)
    logits = (K * query.to(acc)).sum(-1)  # (L, ...)
    alpha = torch.softmax(logits, dim=0)
    return (alpha.unsqueeze(-1) * V).sum(0)


class DepthStream(ResidualStream):
    """Stream = list of layer outputs ``[v_0 = embedding, v_1, ...]``; the final hidden state
    is a softmax-weighted mix of all of them with its own learned pseudo-query."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))

    def init(self, x: Tensor) -> List[Tensor]:
        return [x]

    def finalize(self, stream: List[Tensor]) -> Tensor:
        return _depth_attention(stream, self.query).to(stream[0].dtype)


class AttnResModule(ResidualStrategy):
    """Full attention residuals: ``h_l = sum_i softmax_i(w_l · RMSNorm(v_i)) v_i`` over all
    previous outputs ``v_i``; the sublayer output is appended to the stream.

    ``w_l`` is zero-initialised (uniform averaging at init). Keys use a parameter-free RMSNorm.
    """

    stream_key = ("depth",)

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))

    def forward(self, values: List[Tensor], fn: SublayerFn) -> List[Tensor]:
        h = _depth_attention(values, self.query).to(values[0].dtype)
        return values + [fn(h)]


# ----------------------------------------------------------------------------- specs


@dataclass(frozen=True, kw_only=True)
class ResidualSpec(Spec):
    kind = "residual"
    supports_post_norm = False

    def stream_key(self) -> Tuple:
        return ("tensor",)

    def build_stream(self, ctx: BuildContext) -> ResidualStream:
        return ResidualStream()


@dataclass(frozen=True, kw_only=True)
class Residual(ResidualSpec):
    """Standard additive residual ``x + f(x)``."""

    supports_post_norm = True

    def build(self, ctx: BuildContext) -> ResidualStrategy:
        return StandardResidual()


@dataclass(frozen=True, kw_only=True)
class GatedResidual(ResidualSpec):
    """Gated residual ``x + g ⊙ f(x)``; see :class:`GatedResidualModule`."""

    supports_post_norm = True
    gate: str = "channel"
    init: float = 1.0

    def build(self, ctx: BuildContext) -> ResidualStrategy:
        return GatedResidualModule(ctx.dim, self.gate, self.init)


@dataclass(frozen=True, kw_only=True)
class HyperConnections(ResidualSpec):
    """(Dynamic) hyper-connections with expansion rate ``n``."""

    n: int = 4
    dynamic: bool = True

    def stream_key(self) -> Tuple:
        return ("hyper", self.n)

    def build(self, ctx: BuildContext) -> ResidualStrategy:
        return HyperConnectionModule(ctx.dim, self.n, ctx.layer_idx or 0, self.dynamic)

    def build_stream(self, ctx: BuildContext) -> ResidualStream:
        return HyperStream(self.n)


@dataclass(frozen=True, kw_only=True)
class MHC(ResidualSpec):
    """Manifold-constrained hyper-connections with expansion rate ``n``."""

    n: int = 4
    sinkhorn_iters: int = 20
    alpha_init: float = 0.01

    def stream_key(self) -> Tuple:
        return ("hyper", self.n)

    def build(self, ctx: BuildContext) -> ResidualStrategy:
        return MHCModule(ctx.dim, self.n, self.sinkhorn_iters, self.alpha_init)

    def build_stream(self, ctx: BuildContext) -> ResidualStream:
        return HyperStream(self.n)


@dataclass(frozen=True, kw_only=True)
class AttnRes(ResidualSpec):
    """Full attention residuals over depth."""

    def stream_key(self) -> Tuple:
        return ("depth",)

    def build(self, ctx: BuildContext) -> ResidualStrategy:
        return AttnResModule(ctx.dim)

    def build_stream(self, ctx: BuildContext) -> ResidualStream:
        return DepthStream(ctx.dim)
