"""Position encodings: RoPE, ALiBi, learned absolute embeddings and NoPE.

Position encodings are not all the same kind of thing, and the library does not pretend
they are:

* ``RoPE`` rotates queries and keys inside an attention mixer (``kind = "rotary"``).
* ``ALiBi`` adds a distance-dependent bias to attention logits (``kind = "bias"``).
* ``LearnedPositions`` adds a learned vector to the token embedding at the model input
  (``kind = "absolute"``); it is configured on ``nn.LLM(position=...)``, not on a mixer.
* ``NoPE`` / ``None`` means no explicit position information.

Each mixer declares which kinds it accepts and raises a :class:`SpecError` otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from ...configs.spec import BuildContext, Spec, SpecError

__all__ = [
    "PositionSpec",
    "PositionEncoding",
    "RoPE",
    "ALiBi",
    "LearnedPositions",
    "NoPE",
    "RotaryEmbedding",
    "ALiBiBias",
    "LearnedPositionEmbedding",
    "apply_rotary",
    "check_position",
    "alibi_slopes",
]


class PositionEncoding(nn.Module):
    """Base class for position-encoding modules. ``kind`` is one of
    ``"rotary" | "bias" | "absolute" | "none"``."""

    kind: str = "none"


# ----------------------------------------------------------------------------- RoPE


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embedding (GPT-NeoX / Llama "rotate-half" layout).

    ``x``: ``(..., seq, head_dim)``; ``cos``/``sin``: broadcastable ``(seq, rotary_dim)``.
    If ``rotary_dim < head_dim`` only the first ``rotary_dim`` channels are rotated.
    """
    rd = cos.shape[-1]
    if rd == x.shape[-1]:
        return (x * cos + _rotate_half(x) * sin).to(x.dtype)
    x_rot, x_pass = x[..., :rd], x[..., rd:]
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat((x_rot.to(x.dtype), x_pass), dim=-1)


class RotaryEmbedding(PositionEncoding):
    """Rotary position embedding (Su et al., 2021). Caches cos/sin tables lazily."""

    kind = "rotary"

    def __init__(self, rotary_dim: int, theta: float = 10_000.0, max_seq_len: int = 2048):
        super().__init__()
        if rotary_dim % 2:
            raise SpecError(f"RoPE requires an even rotary dimension, got {rotary_dim}")
        self.rotary_dim, self.theta = rotary_dim, theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq.float(), persistent=False)
        self._cache: Optional[Tuple[Tensor, Tensor]] = None
        self._cached_len = 0
        self._max_seq_len = max_seq_len

    def _build_cache(self, length: int, device: torch.device) -> None:
        length = max(length, self._max_seq_len)
        # computed in float64 on CPU (MPS has no float64), then moved
        t = torch.arange(length, dtype=torch.float64)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float64) / self.rotary_dim))
        emb = torch.cat((torch.outer(t, inv_freq),) * 2, dim=-1)
        self._cache = (emb.cos(), emb.sin())  # float64 on CPU; cast/moved per call
        self._dev_cache = {}
        self._cached_len = length

    def cos_sin(self, offset: int, length: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        end = offset + length
        if self._cache is None or self._cached_len < end:
            self._build_cache(max(end, 2 * self._cached_len), device)
        key = (device, dtype)
        if key not in self._dev_cache:
            self._dev_cache[key] = (self._cache[0].to(device=device, dtype=dtype),  # type: ignore[index]
                                    self._cache[1].to(device=device, dtype=dtype))  # type: ignore[index]
        cos, sin = self._dev_cache[key]
        return cos[offset:end], sin[offset:end]

    def forward(self, q: Tensor, k: Tensor, offset: int = 0) -> Tuple[Tensor, Tensor]:
        """Rotate ``q`` and ``k`` of shape ``(batch, heads, seq, head_dim)``."""
        acc = torch.float64 if q.dtype == torch.float64 else torch.float32
        cos, sin = self.cos_sin(offset, q.shape[-2], q.device, acc)
        return apply_rotary(q.to(acc), cos, sin).to(q.dtype), apply_rotary(k.to(acc), cos, sin).to(k.dtype)

    def extra_repr(self) -> str:
        return f"rotary_dim={self.rotary_dim}, theta={self.theta}"


# ----------------------------------------------------------------------------- ALiBi


def alibi_slopes(n_heads: int) -> Tensor:
    """Head slopes from Press et al. (2022), including the non-power-of-two rule of the
    reference implementation."""

    def pow2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = pow2(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = pow2(closest) + pow2(2 * closest)[0::2][: n_heads - closest]
    return torch.tensor(slopes, dtype=torch.float32)


class ALiBiBias(PositionEncoding):
    """Attention with Linear Biases: adds ``-slope_h * (i - j)`` to logit ``(i, j)``."""

    kind = "bias"

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.register_buffer("slopes", alibi_slopes(n_heads), persistent=False)

    def bias(self, q_len: int, k_len: int, offset: int, device: torch.device) -> Tensor:
        """``(heads, q_len, k_len)`` float32 bias; query ``i`` sits at position ``offset+i``."""
        q_pos = torch.arange(offset, offset + q_len, device=device)
        k_pos = torch.arange(k_len, device=device)
        dist = (q_pos[:, None] - k_pos[None, :]).clamp(min=0).float()
        return -self.slopes.to(device)[:, None, None] * dist

    def extra_repr(self) -> str:
        return f"heads={self.n_heads}"


# ----------------------------------------------------------------------------- learned


class LearnedPositionEmbedding(PositionEncoding):
    """Learned absolute position embeddings added to token embeddings (GPT-2 style)."""

    kind = "absolute"

    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.weight = nn.Parameter(torch.zeros(max_seq_len, dim))

    def forward(self, position_ids: Tensor) -> Tensor:
        if position_ids.numel() and int(position_ids.max()) >= self.max_seq_len:
            raise ValueError(
                f"Position {int(position_ids.max())} exceeds LearnedPositions(max_seq_len={self.max_seq_len})"
            )
        return self.weight[position_ids]


# ----------------------------------------------------------------------------- specs


@dataclass(frozen=True, kw_only=True)
class PositionSpec(Spec):
    kind = "position"
    position_kind = "none"


@dataclass(frozen=True, kw_only=True)
class RoPE(PositionSpec):
    """Rotary position embedding.

    Args:
        theta: base frequency (10_000 in Llama-2, 500_000 in Llama-3).
        rotary_fraction: fraction of each head's channels that are rotated (partial RoPE).
    """

    position_kind = "rotary"
    theta: float = 10_000.0
    rotary_fraction: float = 1.0

    def build_rotary(self, head_dim: int, max_seq_len: int) -> RotaryEmbedding:
        rd = int(head_dim * self.rotary_fraction)
        rd -= rd % 2
        return RotaryEmbedding(rd, self.theta, max_seq_len)


@dataclass(frozen=True, kw_only=True)
class ALiBi(PositionSpec):
    """ALiBi linear attention biases (Press et al., 2022). Attention-only."""

    position_kind = "bias"

    def build_bias(self, n_heads: int) -> ALiBiBias:
        return ALiBiBias(n_heads)


@dataclass(frozen=True, kw_only=True)
class LearnedPositions(PositionSpec):
    """Learned absolute position embeddings. Model-level: pass to ``nn.LLM(position=...)``."""

    position_kind = "absolute"
    max_seq_len: Optional[int] = None

    def build(self, ctx: BuildContext) -> LearnedPositionEmbedding:
        return LearnedPositionEmbedding(self.max_seq_len or ctx.max_seq_len, ctx.dim)


@dataclass(frozen=True, kw_only=True)
class NoPE(PositionSpec):
    """No position encoding (causal masking alone provides positional signal)."""

    position_kind = "none"


def check_position(position: Optional[PositionSpec], allowed: Tuple[str, ...], mixer: str) -> str:
    """Validate that ``position`` is acceptable for ``mixer``; return its kind."""
    if position is None:
        return "none"
    if not isinstance(position, PositionSpec):
        raise SpecError(f"{mixer}: position must be a position spec (e.g. nn.RoPE()), got {position!r}")
    kind = position.position_kind
    if kind == "absolute":
        raise SpecError(
            f"{mixer}: {type(position).__name__} is a model-level (input) embedding. "
            f"Pass it as nn.LLM(position=nn.LearnedPositions(...)) instead of to the mixer."
        )
    if kind not in allowed:
        raise SpecError(
            f"{mixer} does not support {type(position).__name__} (kind '{kind}'). "
            f"Supported position kinds for {mixer}: {list(allowed)}."
        )
    return kind
