"""Human-readable model summaries, parameter breakdowns and architecture trees."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from torch import nn

from ..nn.base import FeedForward

if TYPE_CHECKING:  # pragma: no cover
    from ..nn.models.llm import LLM

__all__ = ["format_count", "active_parameters", "parameter_breakdown", "model_summary", "architecture_tree"]


def format_count(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return str(int(n))


def _params(m: Optional[nn.Module]) -> int:
    return sum(p.numel() for p in m.parameters()) if m is not None else 0


def active_parameters(model: "LLM") -> int:
    total = model.num_parameters()
    for layer in model.layers:
        ffn = layer.ffn  # type: ignore[union-attr]
        if isinstance(ffn, FeedForward):
            total -= _params(ffn) - ffn.active_parameters()
    return total


def parameter_breakdown(model: "LLM") -> Dict[str, int]:
    """Parameter counts by component family."""
    out: Dict[str, int] = {"embedding": model.embed.weight.numel()}
    if model.pos_embed is not None:
        out["position_embedding"] = model.pos_embed.weight.numel()
    mixer = ffn = norm = res = 0
    for layer in model.layers:
        mixer += _params(layer.mixer)  # type: ignore[union-attr]
        ffn += _params(layer.ffn)  # type: ignore[union-attr]
        norm += _params(layer.norm1) + _params(layer.norm2)  # type: ignore[union-attr]
        res += _params(layer.res1) + _params(layer.res2)  # type: ignore[union-attr]
    out.update(mixers=mixer, ffns=ffn, norms=norm + _params(model.final_norm), residual=res + _params(model.stream))
    out["lm_head"] = 0 if model.config.tie_embeddings else model.lm_head.weight.numel()
    out["total"] = model.num_parameters()
    return out


def _layer_rows(model: "LLM") -> List[List[str]]:
    rows = []
    for i, layer in enumerate(model.layers):
        spec = layer.spec  # type: ignore[union-attr]
        ffn = layer.ffn  # type: ignore[union-attr]
        active = _params(layer) - (_params(ffn) - ffn.active_parameters() if isinstance(ffn, FeedForward) else 0)
        rows.append([
            str(i),
            spec.mixer.short_repr(),
            spec.ffn.short_repr() if spec.ffn is not None else "-",
            type(spec.residual).__name__ + ("" if spec.norm_position == "pre" else " (post-norm)"),
            format_count(_params(layer)),
            format_count(active),
        ])
    return rows


def _table(header: List[str], rows: List[List[str]], max_width: int = 48) -> str:
    def clip(s: str) -> str:
        return s if len(s) <= max_width else s[: max_width - 1] + "…"

    rows = [[clip(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[j]) for r in rows)) if rows else len(h) for j, h in enumerate(header)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)
    return f"{line}\n{sep}\n{body}"


def model_summary(model: "LLM", seq_len: Optional[int] = None) -> str:
    from .flops import forward_flops_per_token

    cfg = model.config
    seq_len = seq_len or cfg.max_seq_len
    total = model.num_parameters()
    active = active_parameters(model)
    non_emb = model.num_parameters(exclude_embeddings=True)
    fpt = forward_flops_per_token(model, seq_len)
    parts = [
        f"LLM  vocab={cfg.vocab_size}  dim={cfg.dim}  layers={cfg.num_layers}  max_seq_len={cfg.max_seq_len}  "
        f"config={cfg.fingerprint()}",
        "",
        _table(["#", "mixer", "ffn", "residual", "params", "active"], _layer_rows(model)),
        "",
    ]
    bd = parameter_breakdown(model)
    parts.append("Parameters: " + ", ".join(f"{k}={format_count(v)}" for k, v in bd.items()))
    parts.append(f"Total: {total:,}  |  non-embedding: {non_emb:,}  |  active per token: {active:,}")
    parts.append(
        f"Forward FLOPs/token @ seq_len={seq_len}: {format_count(fpt)}  "
        f"(training ≈ 3x = {format_count(3 * fpt)}; matmul-only estimate, see docs/flops.md)"
    )
    return "\n".join(parts)


def architecture_tree(model: "LLM") -> str:
    """Compact tree; runs of identical blocks are collapsed."""
    cfg = model.config
    blocks = cfg.resolve_layers()
    lines = [f"LLM(vocab_size={cfg.vocab_size}, dim={cfg.dim}, params={format_count(model.num_parameters())})"]
    lines.append(f"├─ embed: Embedding({cfg.vocab_size}, {cfg.dim})")
    if cfg.position is not None:
        lines.append(f"├─ position: {cfg.position.short_repr()}")
    stream = type(model.stream).__name__
    if stream != "ResidualStream":
        lines.append(f"├─ residual stream: {stream}")
    i = 0
    while i < len(blocks):
        j = i
        while j + 1 < len(blocks) and blocks[j + 1] == blocks[i]:
            j += 1
        b = blocks[i]
        label = f"layers[{i}]" if i == j else f"layers[{i}-{j}] × {j - i + 1}"
        lines.append(f"├─ {label}: Block({b.norm_position}-norm {type(b.norm).__name__})")
        lines.append(f"│    ├─ mixer:    {b.mixer.short_repr()}")
        lines.append(f"│    ├─ ffn:      {b.ffn.short_repr() if b.ffn is not None else 'None'}")
        lines.append(f"│    └─ residual: {b.residual.short_repr()}")
        i = j + 1
    lines.append(f"├─ final_norm: {cfg.final_norm.short_repr() if cfg.final_norm is not None else 'None'}")
    tied = " (tied to embed)" if cfg.tie_embeddings else ""
    lines.append(f"└─ lm_head: Linear({cfg.dim} → {cfg.vocab_size}){tied}")
    return "\n".join(lines)
