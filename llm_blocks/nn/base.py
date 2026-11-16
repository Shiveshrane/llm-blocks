"""Core interfaces shared by all building blocks.

The interfaces are deliberately small. Each family of component agrees on *what goes in
and what comes out*; how the computation (and any inference-time state) is organised is
left to the component.

``SequenceMixer``
    Mixes information across the sequence dimension: attention, MLA, Mamba-2, Gated
    DeltaNet... ``forward(x, state=None, padding_mask=None) -> y`` with ``x, y`` of shape
    ``(batch, seq, dim)``. Training passes ``state=None`` and the mixer must be causal.
    For incremental decoding the caller first obtains a mixer-specific state object with
    ``init_state(...)`` and then feeds chunks of tokens (a prompt, then one token at a
    time); the mixer reads and updates its own state *in place*. Attention keeps a KV
    cache, MLA a compressed latent cache, Mamba-2 a convolution + SSM state, Gated
    DeltaNet a convolution + fast-weight matrix. Nothing outside the mixer looks inside.

``FeedForward``
    Position-wise transform ``(batch, seq, dim) -> (batch, seq, dim)``. Receives an
    optional :class:`ForwardContext` so that components like MoE can report auxiliary
    losses and metrics without hidden module state.

``Normalization`` / ``PositionEncoding`` / ``ResidualStrategy``
    See their respective modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn

__all__ = [
    "ForwardContext",
    "MixerState",
    "SequenceMixer",
    "FeedForward",
]


@dataclass
class ForwardContext:
    """Per-forward-call scratch space for auxiliary outputs.

    A fresh context is created for every model forward. Components append auxiliary
    losses (already multiplied by their coefficient) and may record scalar metrics.
    """

    aux_losses: List[Tensor] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_aux_loss(self, loss: Tensor) -> None:
        self.aux_losses.append(loss)

    def total_aux_loss(self) -> Optional[Tensor]:
        if not self.aux_losses:
            return None
        total = self.aux_losses[0]
        for t in self.aux_losses[1:]:
            total = total + t
        return total

    def log(self, key: str, value: Any) -> None:
        self.metrics.setdefault(key, []).append(value)


class MixerState:
    """Base class for incremental-decoding state of a sequence mixer.

    Subclasses own whatever buffers they need. ``offset`` is the number of tokens that
    have already been absorbed into the state (it is also the absolute position of the
    next token for position encodings).
    """

    offset: int = 0

    def nbytes(self) -> int:
        """Total size of tensors held by this state (for memory reporting)."""
        total = 0
        for v in vars(self).values():
            if isinstance(v, Tensor):
                total += v.numel() * v.element_size()
        return total

    def select(self, index: Tensor) -> None:  # pragma: no cover - optional
        """Reorder / select batch entries in place (beam search). Optional."""
        for k, v in vars(self).items():
            if isinstance(v, Tensor) and v.dim() > 0:
                setattr(self, k, v.index_select(0, index))


class SequenceMixer(nn.Module):
    """Abstract base class for all sequence mixers."""

    #: whether ``padding_mask`` is honoured (all built-in mixers do).
    supports_padding_mask: bool = True

    def forward(
        self,
        x: Tensor,
        *,
        state: Optional[MixerState] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:  # pragma: no cover - abstract
        """Args:
            x: ``(batch, seq, dim)`` input.
            state: optional decoding state from :meth:`init_state`; updated in place.
            padding_mask: optional ``(batch, seq)`` bool tensor, ``True`` for real tokens.
                Padding is supported on the left (for batched generation) and on the right
                (for training on variable-length batches); padded outputs are unspecified.
        """
        raise NotImplementedError

    def init_state(
        self,
        batch_size: int,
        max_seq_len: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> MixerState:  # pragma: no cover - abstract
        raise NotImplementedError

    def flops_per_token(self, seq_len: int) -> float:
        """Forward matmul FLOPs per token at context length ``seq_len`` (see docs/flops.md)."""
        return 2.0 * sum(p.numel() for n, p in self.named_parameters() if p.dim() >= 2)

    def extra_summary(self) -> str:
        return ""


class FeedForward(nn.Module):
    """Abstract base class for position-wise feed-forward networks."""

    def forward(self, x: Tensor, ctx: Optional[ForwardContext] = None) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def flops_per_token(self) -> float:
        return 2.0 * sum(p.numel() for p in self.parameters() if p.dim() >= 2)

    def active_parameters(self) -> int:
        """Parameters touched by a single token (differs from total for MoE)."""
        return sum(p.numel() for p in self.parameters())


def mark_residual_output(linear: nn.Module) -> nn.Module:
    """Tag a projection that writes into the residual stream (gets scaled init)."""
    linear._llm_blocks_residual_out = True  # type: ignore[attr-defined]
    return linear


# ----------------------------------------------------------------------------- spec bases
from dataclasses import dataclass as _dataclass  # noqa: E402

from ..configs.spec import BuildContext, Spec  # noqa: E402


@_dataclass(frozen=True, kw_only=True)
class MixerSpec(Spec):
    """Spec for a :class:`SequenceMixer`."""

    kind = "mixer"

    def build(self, ctx: BuildContext) -> SequenceMixer:  # pragma: no cover
        raise NotImplementedError


@_dataclass(frozen=True, kw_only=True)
class FFNSpec(Spec):
    """Spec for a :class:`FeedForward`."""

    kind = "ffn"

    def build(self, ctx: BuildContext) -> FeedForward:  # pragma: no cover
        raise NotImplementedError

    def build_experts(self, ctx: BuildContext, num_experts: int) -> nn.Module:
        """Build ``num_experts`` copies with stacked weights for batched MoE execution."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot be used as an MoE expert (no batched expert implementation)."
        )
