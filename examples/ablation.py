"""Ablation study: swap one component at a time and train every variant identically.

Answers questions like "what happens if I replace GQA with MLA?", "...every other layer with
Mamba-2?", "...the dense FFN with MoE?", "...the residual mechanism?".

    python examples/ablation.py --steps 300
"""

from __future__ import annotations

import argparse
import sysconfig
from pathlib import Path

from llm_blocks import nn
from llm_blocks.training import ByteTokenizer, TokenDataset, TrainConfig, load_text_corpus, results_table, run_ablation


def variants(dim: int, depth: int):
    base = nn.LLMConfig(
        vocab_size=ByteTokenizer.vocab_size, dim=dim, depth=depth,
        mixer=nn.GQA(heads=6, kv_heads=2, position=nn.RoPE()),
        ffn=nn.SwiGLU(hidden_dim=512), norm=nn.RMSNorm(), max_seq_len=1024, seed=0,
    )
    return {
        "baseline_gqa": base,
        "mla": base.replace(mixer=nn.MLA(heads=6)),
        "hybrid_mamba2_every2": base.replace_layers(nn.every(2, start=1), mixer=nn.Mamba2(d_state=32, headdim=32)),
        "all_gated_deltanet": base.replace(mixer=nn.GatedDeltaNet(head_dim=32)),
        "moe_8x_top2": base.replace(ffn=nn.MoE(expert=nn.SwiGLU(hidden_dim=256), num_experts=8, top_k=2)),
        "gated_residual": base.replace(residual=nn.GatedResidual(gate="channel", init=1.0)),
        "mhc_n4": base.replace(residual=nn.MHC(n=4)),
        "attn_res": base.replace(residual=nn.AttnRes()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/ablation")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    tok = ByteTokenizer()
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    data = TokenDataset(tok.encode(load_text_corpus(stdlib.glob("*.py"), max_bytes=6_000_000)), seq_len=256)
    train, val = data.split(0.05)
    tcfg = TrainConfig(max_steps=args.steps, batch_size=16, seq_len=256, lr=2e-3, warmup_steps=args.steps // 20,
                       precision="bf16", device=args.device, eval_every=args.steps // 3, eval_batches=16,
                       log_every=max(1, args.steps // 10), seed=0)
    vs = variants(args.dim, args.depth)
    if args.only:
        vs = {k: v for k, v in vs.items() if k in args.only}
    results = run_ablation(vs, train, val, tcfg, out_dir=args.out)
    print()
    print(results_table(results))


if __name__ == "__main__":
    main()
