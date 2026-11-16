"""Regenerate the figures embedded in README.md.

    python examples/ablation.py --steps 300     # writes runs/ablation/*.json
    python benchmarks/make_readme_figures.py    # writes docs/assets/*.svg
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from llm_blocks import nn

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"

STYLE = """<style>
  .surface { fill: #fcfcfb; } .bar { fill: #2a78d6; } .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; }
  .grid { stroke: #e4e3df; } .axis { stroke: #b9b8b2; } .ref { stroke: #52514e; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; } .bar { fill: #3987e5; } .t1 { fill: #f0efec; } .t2 { fill: #c3c2b7; }
    .grid { stroke: #2e2e2c; } .axis { stroke: #57564f; } .ref { stroke: #c3c2b7; }
  }
  text { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; }
</style>"""


def ablation_chart(results: list[dict]) -> str:
    rows = sorted(results, key=lambda r: r["final_val_loss"])
    W, left, right, top, row, bar = 760, 190, 70, 64, 34, 20
    H = top + row * len(rows) + 46
    vmax = 2.2
    x = lambda v: left + (W - left - right) * v / vmax
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
           f'aria-label="Final validation loss per ablation variant">', STYLE,
           f'<rect class="surface" width="{W}" height="{H}" rx="8"/>',
           f'<text class="t1" x="20" y="28" font-size="16" font-weight="600">Final validation loss by variant (lower is better)</text>',
           f'<text class="t2" x="20" y="47" font-size="12">One component changed per variant · 300 steps, 1.2M tokens, single seed · nats/byte</text>']
    for v in (0, 0.5, 1.0, 1.5, 2.0):
        out.append(f'<line class="grid" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top - 6}" y2="{H - 34}" stroke-width="1"/>')
        out.append(f'<text class="t2" x="{x(v):.1f}" y="{H - 18}" font-size="11" text-anchor="middle">{v:.1f}</text>')
    for i, r in enumerate(rows):
        y = top + i * row + (row - bar) / 2
        w = x(r["final_val_loss"]) - left
        rad = 4
        # square at the baseline, 4px rounded data end
        out.append(f'<path class="bar" d="M{left},{y} h{w - rad:.1f} a{rad},{rad} 0 0 1 {rad},{rad} v{bar - 2 * rad} '
                   f'a{rad},{rad} 0 0 1 -{rad},{rad} h-{w - rad:.1f} z"/>')
        label = r["name"].replace("_", " ") + (" (baseline)" if r["name"] == "baseline_gqa" else "")
        weight = ' font-weight="600"' if r["name"] == "baseline_gqa" else ""
        out.append(f'<text class="t1" x="{left - 10}" y="{y + bar / 2 + 4}" font-size="12.5" text-anchor="end"{weight}>{escape(label)}</text>')
        out.append(f'<text class="t2" x="{left + w + 6:.1f}" y="{y + bar / 2 + 4}" font-size="12">{r["final_val_loss"]:.3f}</text>')
    out.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top - 6}" y2="{H - 34}" stroke-width="1"/>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    runs = ROOT / "runs" / "ablation"
    files = sorted(p for p in runs.glob("*.json"))
    if files:
        results = [json.loads(p.read_text()) for p in files]
        (ASSETS / "ablation_val_loss.svg").write_text(ablation_chart(results))
    else:
        print(f"no ablation results in {runs}; skipping ablation chart")

    hybrid = nn.LLMConfig(
        vocab_size=32000, dim=1024, depth=8, mixer=nn.Mamba2(), ffn=nn.SwiGLU(hidden_dim=2816),
    ).replace_layers(nn.every(4, start=3), mixer=nn.GQA(heads=16, kv_heads=4),
                     ffn=nn.MoE(expert=nn.SwiGLU(hidden_dim=1024), num_experts=16, top_k=2, shared_experts=1))
    nn.LLM(config=hybrid).visualize(ASSETS / "hybrid_architecture.svg")
    print("wrote", sorted(p.name for p in ASSETS.glob("*.svg")))


if __name__ == "__main__":
    main()
