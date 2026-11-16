"""A residual block: sequence mixer + (optional) feed-forward, each wrapped in a norm and
a residual strategy."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from torch import nn

from ...configs.spec import BuildContext, Spec, SpecError
from ..base import FeedForward, FFNSpec, ForwardContext, MixerSpec, MixerState, SequenceMixer
from ..ffn import SwiGLU
from ..norm import NormSpec, RMSNorm
from ..residual import Residual, ResidualSpec

__all__ = ["BlockModule", "Block"]


class BlockModule(nn.Module):
    """Pre-norm:  ``s = R1(s, lambda h: mixer(norm1(h)))``, ``s = R2(s, lambda h: ffn(norm2(h)))``.

    Post-norm: ``x = norm1(R1(x, mixer))``, ``x = norm2(R2(x, ffn))`` (original Transformer;
    only valid for tensor-stream residuals).
    """

    def __init__(self, mixer: SequenceMixer, ffn: Optional[FeedForward], norm1: nn.Module, norm2: Optional[nn.Module],
                 res1: nn.Module, res2: Optional[nn.Module], norm_position: str, spec: "Block", layer_idx: int):
        super().__init__()
        self.mixer, self.ffn = mixer, ffn
        self.norm1, self.norm2 = norm1, norm2
        self.res1, self.res2 = res1, res2
        self.norm_position = norm_position
        self.spec = spec
        self.layer_idx = layer_idx

    def forward(self, stream, *, state: Optional[MixerState] = None, padding_mask=None,
                ctx: Optional[ForwardContext] = None):
        if self.norm_position == "pre":
            stream = self.res1(stream, lambda h: self.mixer(self.norm1(h), state=state, padding_mask=padding_mask))
            if self.ffn is not None:
                stream = self.res2(stream, lambda h: self.ffn(self.norm2(h), ctx))
        else:
            stream = self.norm1(self.res1(stream, lambda h: self.mixer(h, state=state, padding_mask=padding_mask)))
            if self.ffn is not None:
                stream = self.norm2(self.res2(stream, lambda h: self.ffn(h, ctx)))
        return stream

    def init_state(self, batch_size, max_seq_len, **kw) -> MixerState:
        return self.mixer.init_state(batch_size, max_seq_len, **kw)


@dataclass(frozen=True, kw_only=True)
class Block(Spec):
    """Spec for one residual block.

    Args:
        mixer: sequence mixer spec (``nn.GQA``, ``nn.MLA``, ``nn.Mamba2``, ``nn.GatedDeltaNet``...).
        ffn: feed-forward spec (default ``nn.SwiGLU()``; ``nn.MoE``...), or ``None`` for mixer-only
            blocks (e.g. pure Mamba stacks).
        norm: normalization spec used around both sublayers.
        residual: residual strategy spec.
        norm_position: ``"pre"`` (default) or ``"post"``.
    """

    kind = "block"
    mixer: MixerSpec
    ffn: Optional[FFNSpec] = field(default_factory=SwiGLU)
    norm: NormSpec = field(default_factory=RMSNorm)
    residual: ResidualSpec = field(default_factory=Residual)
    norm_position: str = "pre"

    def validate(self) -> None:
        if not isinstance(self.mixer, MixerSpec):
            raise SpecError(f"Block.mixer must be a mixer spec (e.g. nn.GQA(...)), got {type(self.mixer).__name__}")
        if self.ffn is not None and not isinstance(self.ffn, FFNSpec):
            raise SpecError(f"Block.ffn must be a feed-forward spec (e.g. nn.SwiGLU(...)), got {type(self.ffn).__name__}")
        if not isinstance(self.norm, NormSpec):
            raise SpecError(f"Block.norm must be a norm spec (e.g. nn.RMSNorm()), got {type(self.norm).__name__}")
        if not isinstance(self.residual, ResidualSpec):
            raise SpecError(f"Block.residual must be a residual spec, got {type(self.residual).__name__}")
        if self.norm_position not in ("pre", "post"):
            raise SpecError(f"Block.norm_position must be 'pre' or 'post', got {self.norm_position!r}")
        if self.norm_position == "post" and not self.residual.supports_post_norm:
            raise SpecError(
                f"Post-norm is only defined for tensor-stream residuals (Residual, GatedResidual), "
                f"not {type(self.residual).__name__}"
            )

    def build(self, ctx: BuildContext) -> BlockModule:
        self.validate()
        mixer = self.mixer.build(ctx)
        ffn = self.ffn.build(ctx) if self.ffn is not None else None
        norm1 = self.norm.build(ctx)
        norm2 = self.norm.build(ctx) if ffn is not None else None
        # each sublayer counts as one "layer" for residual mechanisms (e.g. hyper-connection init)
        li = ctx.layer_idx or 0
        res1 = self.residual.build(dataclasses.replace(ctx, layer_idx=2 * li))
        res2 = self.residual.build(dataclasses.replace(ctx, layer_idx=2 * li + 1)) if ffn is not None else None
        return BlockModule(mixer, ffn, norm1, norm2, res1, res2, self.norm_position, self, ctx.layer_idx or 0)
