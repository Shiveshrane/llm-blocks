"""The causal language model assembled from block specs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ...configs.spec import SpecError
from ...kernels._dtype import acc_dtype
from ..base import ForwardContext, MixerState
from ..blocks import Block, BlockModule
from .config import LLMConfig

__all__ = ["LLM", "LMOutput", "ModelState"]


@dataclass
class LMOutput:
    """Model output. ``loss = lm_loss + aux_loss`` when labels are given."""

    logits: Tensor
    loss: Optional[Tensor] = None
    lm_loss: Optional[Tensor] = None
    aux_loss: Optional[Tensor] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelState:
    """Incremental-decoding state: one mixer-specific state per layer.

    Each layer owns its state type (KV cache, latent cache, SSM state, fast weights...);
    the model only tracks how many positions have been consumed.
    """

    layers: List[MixerState]
    offset: int = 0
    real_tokens: Optional[Tensor] = None  # (B,) number of non-padding tokens seen (learned positions)

    def nbytes(self) -> int:
        return sum(s.nbytes() for s in self.layers)


class LLM(nn.Module):
    """Decoder-only causal language model.

    Build it either from block specs::

        model = nn.LLM(vocab_size=32000, dim=768, layers=[nn.Block(mixer=nn.GQA(heads=12, kv_heads=4))] * 12)

    or from a config::

        model = nn.LLM(config=nn.LLMConfig(vocab_size=32000, dim=768, depth=12, mixer=nn.GQA(heads=12, kv_heads=4)))

    Any other ``LLMConfig`` field can be passed as a keyword (``tie_embeddings``,
    ``position``, ``max_seq_len``, ``seed``...).
    """

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        dim: Optional[int] = None,
        layers: Optional[List[Block]] = None,
        *,
        config: Optional[LLMConfig] = None,
        gradient_checkpointing: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        **config_kwargs: Any,
    ):
        super().__init__()
        if config is None:
            if vocab_size is None or dim is None:
                raise SpecError("nn.LLM needs vocab_size and dim (or config=...)")
            config = LLMConfig(vocab_size=vocab_size, dim=dim,
                               layers=tuple(layers) if layers is not None else None, **config_kwargs)
        elif vocab_size is not None or dim is not None or layers is not None or config_kwargs:
            raise SpecError("Pass either config=... or model arguments, not both")
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing
        blocks = config.resolve_layers()
        if not blocks:
            raise SpecError("A model needs at least one block")
        for i, b in enumerate(blocks):
            if not isinstance(b, Block):
                raise SpecError(f"layers[{i}] must be nn.Block(...), got {type(b).__name__}")
            b.validate()
        keys = {b.residual.stream_key() for b in blocks}
        if len(keys) > 1:
            raise SpecError(
                f"All blocks must share one residual stream type, found {sorted(map(str, keys))}. "
                "Hyper-connection / attention-residual variants change the shape of the residual "
                "stream and cannot be mixed with other stream types in one model."
            )
        if config.position is not None and config.position.position_kind != "absolute":
            raise SpecError(
                f"nn.LLM(position=...) is for input embeddings such as nn.LearnedPositions(); "
                f"{type(config.position).__name__} belongs on the mixer (e.g. nn.GQA(position=...))"
            )

        fork = torch.random.fork_rng(devices=[]) if config.seed is not None else _nullcontext()
        with fork:
            if config.seed is not None:
                torch.manual_seed(config.seed)
            ctx = config.build_context()
            self.embed = nn.Embedding(config.vocab_size, config.dim)
            self.pos_embed = config.position.build(ctx) if config.position is not None else None
            self.stream = blocks[0].residual.build_stream(ctx)
            self.layers = nn.ModuleList([b.build(config.build_context(i)) for i, b in enumerate(blocks)])
            self.final_norm = config.final_norm.build(ctx) if config.final_norm is not None else nn.Identity()
            self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
            if config.tie_embeddings:
                self.lm_head.weight = self.embed.weight
            self._init_weights()
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    # -- initialization -------------------------------------------------------------------
    def _init_weights(self) -> None:
        cfg = self.config
        std = cfg.init_std
        out_std = std / math.sqrt(2 * cfg.num_layers) if cfg.scaled_init else std
        for m in self.modules():
            if isinstance(m, nn.Linear) and not getattr(m, "_llm_blocks_skip_init", False):
                nn.init.normal_(m.weight, std=out_std if getattr(m, "_llm_blocks_residual_out", False) else std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=std)
        for m in self.modules():
            hook = getattr(m, "_llm_blocks_init", None)
            if hook is not None:
                hook(std, out_std)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed.weight, std=std)

    def gradient_checkpointing_enable(self, enabled: bool = True) -> "LLM":
        self.gradient_checkpointing = enabled
        return self

    # -- forward --------------------------------------------------------------------------
    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        *,
        attention_mask: Optional[Tensor] = None,
        state: Optional[ModelState] = None,
        logits_to_keep: Optional[int] = None,
    ) -> LMOutput:
        """Args:
            input_ids: ``(batch, seq)`` token ids.
            labels: optional ``(batch, seq)`` targets aligned with ``input_ids``; the model
                predicts ``labels[:, t+1]`` from position ``t`` (shift is internal).
                ``-100`` is ignored.
            attention_mask: optional ``(batch, seq)`` 1/0 padding mask (left or right padding).
            state: decoding state from :meth:`init_state`, updated in place.
            logits_to_keep: only compute logits for the last ``n`` positions.
        """
        B, T = input_ids.shape
        x = self.embed(input_ids)
        padding_mask = attention_mask.bool() if attention_mask is not None else None
        if self.pos_embed is not None:
            x = x + self.pos_embed(self._position_ids(B, T, padding_mask, state, x.device)).to(x.dtype)
        stream = self.stream.init(x)
        ctx = ForwardContext()
        use_ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for i, layer in enumerate(self.layers):
            st = state.layers[i] if state is not None else None
            if use_ckpt:
                stream = checkpoint(layer, stream, state=st, padding_mask=padding_mask, ctx=ctx, use_reentrant=False)
            else:
                stream = layer(stream, state=st, padding_mask=padding_mask, ctx=ctx)
        h = self.final_norm(self.stream.finalize(stream))
        if state is not None:
            state.offset += T
        if logits_to_keep is not None:
            h = h[:, -logits_to_keep:]
        logits = self.lm_head(h)

        out = LMOutput(logits=logits, metrics=ctx.metrics)
        out.aux_loss = ctx.total_aux_loss()
        if labels is not None:
            if logits_to_keep is not None:
                raise ValueError("labels cannot be combined with logits_to_keep")
            out.lm_loss = F.cross_entropy(
                logits[:, :-1].to(acc_dtype(logits)).reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100
            )
            out.loss = out.lm_loss + out.aux_loss if out.aux_loss is not None else out.lm_loss
        return out

    def _position_ids(self, B, T, padding_mask, state, device) -> Tensor:
        """Absolute positions counting only real (non-padding) tokens."""
        m = padding_mask.long() if padding_mask is not None else torch.ones(B, T, dtype=torch.long, device=device)
        pos = (m.cumsum(-1) - 1).clamp(min=0)
        if state is not None:
            if state.real_tokens is None:
                state.real_tokens = torch.zeros(B, dtype=torch.long, device=device)
            pos = pos + state.real_tokens[:, None]
            state.real_tokens = state.real_tokens + m.sum(-1)
        return pos

    # -- inference ------------------------------------------------------------------------
    def init_state(
        self,
        batch_size: int,
        max_seq_len: Optional[int] = None,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> ModelState:
        """Allocate per-layer decoding states (KV caches are pre-allocated to ``max_seq_len``)."""
        max_seq_len = max_seq_len or self.config.max_seq_len
        layers = [blk.init_state(batch_size, max_seq_len, device=device, dtype=dtype) for blk in self.layers]  # type: ignore[union-attr]
        return ModelState(layers=layers)

    def generate(self, input_ids: Tensor, max_new_tokens: int = 32, **kwargs: Any) -> Tensor:
        """Autoregressive generation; see :func:`llm_blocks.inference.generate`."""
        from ...inference import generate

        return generate(self, input_ids, max_new_tokens=max_new_tokens, **kwargs)

    # -- introspection --------------------------------------------------------------------
    def num_parameters(self, *, exclude_embeddings: bool = False, trainable_only: bool = False) -> int:
        params = {id(p): p for p in self.parameters() if (p.requires_grad or not trainable_only)}
        if exclude_embeddings:
            params.pop(id(self.embed.weight), None)
            if self.pos_embed is not None:
                params.pop(id(self.pos_embed.weight), None)
        return sum(p.numel() for p in params.values())

    def num_active_parameters(self) -> int:
        """Parameters used per token (MoE counts only ``top_k`` routed experts)."""
        from ...utils.summary import active_parameters

        return active_parameters(self)

    def summary(self, seq_len: Optional[int] = None) -> str:
        from ...utils.summary import model_summary

        return model_summary(self, seq_len=seq_len)

    def architecture(self) -> str:
        from ...utils.summary import architecture_tree

        return architecture_tree(self)

    def parameter_breakdown(self) -> Dict[str, int]:
        from ...utils.summary import parameter_breakdown

        return parameter_breakdown(self)

    def flops_per_token(self, seq_len: Optional[int] = None) -> float:
        from ...utils.flops import forward_flops_per_token

        return forward_flops_per_token(self, seq_len or self.config.max_seq_len)

    def visualize(self, path: Optional[Union[str, Path]] = None, fmt: str = "svg") -> str:
        """Block diagram of the architecture as SVG (default) or Mermaid source; optionally saved."""
        from ...visualization import to_mermaid, to_svg

        text = to_svg(self) if fmt == "svg" else to_mermaid(self)
        if path is not None:
            Path(path).write_text(text)
        return text

    # -- persistence ----------------------------------------------------------------------
    def save_pretrained(self, directory: Union[str, Path]) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.config.save(d / "config.json")
        torch.save(self.state_dict(), d / "model.pt")

    @classmethod
    def from_pretrained(cls, directory: Union[str, Path], map_location: Union[str, torch.device] = "cpu") -> "LLM":
        d = Path(directory)
        model = cls(config=LLMConfig.load(d / "config.json"))
        model.load_state_dict(torch.load(d / "model.pt", map_location=map_location, weights_only=True))
        return model


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
