"""Top-level model configuration with ablation-friendly editing."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ...configs.spec import BuildContext, Spec, SpecError
from ..base import FFNSpec, MixerSpec
from ..blocks import Block
from ..ffn import SwiGLU
from ..norm import NormSpec, RMSNorm
from ..position import PositionSpec
from ..residual import Residual, ResidualSpec

__all__ = ["LLMConfig", "every", "LayerSelector"]

LayerSelector = Union[int, slice, Iterable[int], Callable[[int], bool]]

_BLOCK_FIELDS = {"mixer", "ffn", "norm", "residual", "norm_position"}


def every(n: int, start: int = 0) -> Callable[[int], bool]:
    """Layer selector: layers ``start, start+n, start+2n, ...``.

    ``every(4, start=3)`` selects every fourth layer (3, 7, 11, ...).
    """
    if n <= 0:
        raise ValueError("every(n): n must be positive")
    return lambda i: i >= start and (i - start) % n == 0


def _select(where: LayerSelector, depth: int) -> List[int]:
    if isinstance(where, int):
        idx = [where]
    elif isinstance(where, slice):
        idx = list(range(depth))[where]
    elif callable(where):
        idx = [i for i in range(depth) if where(i)]
    else:
        idx = list(where)
    for i in idx:
        if not -depth <= i < depth:
            raise SpecError(f"Layer index {i} out of range for depth {depth}")
    return sorted({i % depth for i in idx})


@dataclass(frozen=True, kw_only=True)
class LLMConfig(Spec):
    """Complete, serializable description of a causal language model.

    Two equivalent ways to describe the layer stack:

    * **uniform + overrides**: ``depth``, ``mixer``, ``ffn``, ``norm``, ``residual`` give the
      default block; ``overrides`` replace fields in selected layers (see
      :meth:`replace_layers`). ``replace(mixer=...)`` then changes every non-overridden layer.
    * **explicit**: ``layers=[nn.Block(...), ...]``.

    Args:
        vocab_size, dim: vocabulary size and residual width.
        position: optional model-level position embedding (``nn.LearnedPositions()``).
        final_norm: normalization before the LM head (``None`` disables it).
        tie_embeddings: share input embedding and output projection weights.
        max_seq_len: maximum context (sizes RoPE tables / learned positions).
        init_std: std of the normal weight init; ``scaled_init`` divides it by
            ``sqrt(2 * depth)`` for projections that write into the residual stream (GPT-2).
        seed: seed for parameter initialization (``None`` = use the global RNG).
    """

    kind = "model"
    vocab_size: int
    dim: int
    depth: Optional[int] = None
    mixer: Optional[MixerSpec] = None
    ffn: Optional[FFNSpec] = field(default_factory=SwiGLU)
    norm: NormSpec = field(default_factory=RMSNorm)
    residual: ResidualSpec = field(default_factory=Residual)
    norm_position: str = "pre"
    layers: Optional[Tuple[Block, ...]] = None
    overrides: Tuple[Dict[str, Any], ...] = ()
    position: Optional[PositionSpec] = None
    final_norm: Optional[NormSpec] = field(default_factory=RMSNorm)
    tie_embeddings: bool = False
    max_seq_len: int = 2048
    init_std: float = 0.02
    scaled_init: bool = True
    seed: Optional[int] = 0

    def __post_init__(self) -> None:
        if self.layers is not None and not isinstance(self.layers, tuple):
            object.__setattr__(self, "layers", tuple(self.layers))
        if not isinstance(self.overrides, tuple):
            object.__setattr__(self, "overrides", tuple(self.overrides))
        if self.layers is not None:
            if self.depth is not None and self.depth != len(self.layers):
                raise SpecError(f"depth={self.depth} conflicts with {len(self.layers)} explicit layers")
            if self.mixer is not None:
                raise SpecError("Pass either explicit `layers` or a uniform `mixer` (+ depth), not both")
            if self.overrides:
                raise SpecError("`overrides` only apply to uniform configs; edit `layers` directly instead")
        else:
            if self.depth is None or self.mixer is None:
                raise SpecError("LLMConfig needs either `layers=[...]` or both `depth` and `mixer`")
        if self.vocab_size <= 0 or self.dim <= 0:
            raise SpecError("vocab_size and dim must be positive")

    # -- layers -------------------------------------------------------------------------
    @property
    def num_layers(self) -> int:
        return len(self.layers) if self.layers is not None else int(self.depth)  # type: ignore[arg-type]

    def resolve_layers(self) -> List[Block]:
        """The concrete list of block specs, with overrides applied."""
        if self.layers is not None:
            return list(self.layers)
        base = Block(mixer=self.mixer, ffn=self.ffn, norm=self.norm, residual=self.residual,  # type: ignore[arg-type]
                     norm_position=self.norm_position)
        blocks = [base] * self.num_layers
        for ov in self.overrides:
            changes = {k: v for k, v in ov.items() if k != "layers"}
            for i in ov["layers"]:
                blocks[i] = blocks[i].replace(**changes)
        return blocks

    def replace_layers(self, where: LayerSelector, **changes: Any) -> "LLMConfig":
        """Return a config where the selected layers have some block fields replaced.

        ``where`` is an index, a slice, an iterable of indices, or a predicate on the index.

        >>> hybrid = base.replace_layers(every(4, start=3), mixer=nn.Mamba2())
        """
        bad = set(changes) - _BLOCK_FIELDS
        if bad:
            raise SpecError(f"replace_layers: unknown block field(s) {sorted(bad)}; valid: {sorted(_BLOCK_FIELDS)}")
        idx = _select(where, self.num_layers)
        if self.layers is not None:
            layers = list(self.layers)
            for i in idx:
                layers[i] = layers[i].replace(**changes)
            return self.replace(layers=tuple(layers))
        return self.replace(overrides=self.overrides + ({"layers": tuple(idx), **changes},))

    # -- construction -------------------------------------------------------------------
    def build_context(self, layer_idx: Optional[int] = None) -> BuildContext:
        return BuildContext(dim=self.dim, layer_idx=layer_idx, num_layers=self.num_layers,
                            max_seq_len=self.max_seq_len, init_std=self.init_std)

    def build(self, ctx: Optional[BuildContext] = None, **kwargs: Any):
        from .llm import LLM

        return LLM(config=self, **kwargs)

    # -- reproducibility ----------------------------------------------------------------
    def fingerprint(self) -> str:
        """Short stable hash of the full configuration."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LLMConfig":
        return cls.from_json(Path(path).read_text())
