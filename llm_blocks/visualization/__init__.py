"""Dependency-free architecture diagrams (SVG and Mermaid)."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import List, Tuple, Union

from ..nn.blocks import Block
from ..nn.models.config import LLMConfig

__all__ = ["to_svg", "to_mermaid", "save_svg"]

_COLORS = {
    "MHA": "#4C78A8", "GQA": "#4C78A8", "MQA": "#4C78A8", "MLA": "#72B7B2",
    "Mamba2": "#F58518", "GatedDeltaNet": "#E45756",
    "MoE": "#B279A2", "_ffn": "#54A24B", "_other": "#9D9D9D",
}


def _config(obj) -> LLMConfig:
    return obj if isinstance(obj, LLMConfig) else obj.config


def _runs(blocks: List[Block]) -> List[Tuple[int, int, Block]]:
    runs, i = [], 0
    while i < len(blocks):
        j = i
        while j + 1 < len(blocks) and blocks[j + 1] == blocks[i]:
            j += 1
        runs.append((i, j, blocks[i]))
        i = j + 1
    return runs


def to_svg(model_or_config, width: int = 760) -> str:
    """Render the layer stack as an SVG block diagram (bottom = input, top = output)."""
    cfg = _config(model_or_config)
    runs = _runs(cfg.resolve_layers())
    row_h, gap, pad = 64, 14, 20
    items = [("LM head", f"Linear {cfg.dim}→{cfg.vocab_size}", "_other")]
    if cfg.final_norm is not None:
        items.append(("Final norm", type(cfg.final_norm).__name__, "_other"))
    for i, j, b in reversed(runs):
        label = f"layer {i}" if i == j else f"layers {i}–{j}  (×{j - i + 1})"
        items.append((label, b, "block"))
    if cfg.position is not None:
        items.append(("Position", cfg.position.short_repr(), "_other"))
    items.append(("Embedding", f"{cfg.vocab_size} × {cfg.dim}", "_other"))

    n_blocks = sum(1 for it in items if it[2] == "block")
    height = pad * 2 + n_blocks * row_h + (len(items) - n_blocks) * (row_h - 20) + (len(items) - 1) * gap + 30
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
           f'<rect width="100%" height="100%" fill="#ffffff"/>',
           f'<text x="{pad}" y="{pad + 4}" font-size="14" font-weight="bold">'
           f'{escape(f"LLM  dim={cfg.dim}  layers={cfg.num_layers}")}</text>']
    y = pad + 20
    inner_w = width - 2 * pad
    for title, body, kind in items:
        if kind == "block":
            b: Block = body  # type: ignore[assignment]
            mname = type(b.mixer).__name__
            fname = type(b.ffn).__name__ if b.ffn is not None else None
            out.append(f'<rect x="{pad}" y="{y}" width="{inner_w}" height="{row_h}" rx="8" fill="#f4f4f4" stroke="#888"/>')
            out.append(f'<text x="{pad + 8}" y="{y + 16}" font-weight="bold">{escape(title)}</text>')
            out.append(f'<text x="{pad + inner_w - 8}" y="{y + 16}" text-anchor="end" fill="#555">'
                       f'{escape(b.norm_position + "-norm " + type(b.norm).__name__ + " · " + type(b.residual).__name__)}</text>')
            half = (inner_w - 24) / 2
            max_chars = int((half - 12) / 7.0)
            bx = pad + 8
            out.append(f'<rect x="{bx}" y="{y + 24}" width="{half}" height="{row_h - 32}" rx="5" '
                       f'fill="{_COLORS.get(mname, _COLORS["_other"])}"/>')
            out.append(f'<text x="{bx + 6}" y="{y + 44}" fill="#fff">{escape(_clip(b.mixer.short_repr(), max_chars))}</text>')
            if fname is not None:
                fx = bx + half + 8
                color = _COLORS.get(fname, _COLORS["_ffn"])
                out.append(f'<rect x="{fx}" y="{y + 24}" width="{half}" height="{row_h - 32}" rx="5" fill="{color}"/>')
                out.append(f'<text x="{fx + 6}" y="{y + 44}" fill="#fff">{escape(_clip(b.ffn.short_repr(), max_chars))}</text>')  # type: ignore[union-attr]
        else:
            out.append(f'<rect x="{pad}" y="{y}" width="{inner_w}" height="{row_h - 20}" rx="8" fill="#e8e8e8" stroke="#888"/>')
            out.append(f'<text x="{pad + 8}" y="{y + 27}"><tspan font-weight="bold">{escape(title)}</tspan>'
                       f'  {escape(str(body))}</text>')
            y -= 20
        y += row_h + gap
    out.append("</svg>")
    return "\n".join(out)


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def save_svg(model_or_config, path: Union[str, Path]) -> Path:
    p = Path(path)
    p.write_text(to_svg(model_or_config))
    return p


def to_mermaid(model_or_config) -> str:
    """Mermaid flowchart source (renders on GitHub / in many Markdown viewers)."""
    cfg = _config(model_or_config)
    lines = ["flowchart BT", f'  emb["Embedding {cfg.vocab_size}×{cfg.dim}"]']
    prev = "emb"
    for k, (i, j, b) in enumerate(_runs(cfg.resolve_layers())):
        node = f"L{k}"
        label = f"layer {i}" if i == j else f"layers {i}-{j} ×{j - i + 1}"
        ffn = b.ffn.short_repr() if b.ffn is not None else "no FFN"
        text = f"{label}<br/>{b.mixer.short_repr()}<br/>{ffn}".replace('"', "'")
        lines.append(f'  {node}["{text}"]')
        lines.append(f"  {prev} --> {node}")
        prev = node
    lines.append(f'  head["final norm + LM head"]')
    lines.append(f"  {prev} --> head")
    return "\n".join(lines)
