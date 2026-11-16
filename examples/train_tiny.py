"""Train a tiny byte-level LM end to end, then verify checkpoint reload and generation.

Corpus: the Python standard library sources of the running interpreter (available offline).

    python examples/train_tiny.py --steps 600
    python examples/train_tiny.py --mixer mamba2 --steps 300
"""

from __future__ import annotations

import argparse
import json
import sysconfig
import time
from pathlib import Path

import torch

from llm_blocks import nn
from llm_blocks.training import ByteTokenizer, TokenDataset, TrainConfig, Trainer, load_text_corpus

MIXERS = {
    "gqa": lambda: nn.GQA(heads=8, kv_heads=2, position=nn.RoPE()),
    "mla": lambda: nn.MLA(heads=8),
    "mamba2": lambda: nn.Mamba2(d_state=32, headdim=32),
    "gdn": lambda: nn.GatedDeltaNet(head_dim=32),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixer", default="gqa", choices=list(MIXERS))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--out", default="runs/tiny")
    ap.add_argument("--corpus-bytes", type=int, default=8_000_000)
    args = ap.parse_args()

    tok = ByteTokenizer()
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    text = load_text_corpus(stdlib.glob("*.py"), max_bytes=args.corpus_bytes)
    data = TokenDataset(tok.encode(text), seq_len=args.seq_len)
    train, val = data.split(0.05)
    print(f"corpus: {len(text):,} bytes from {stdlib}  train={len(train):,} val={len(val):,} tokens")

    cfg = nn.LLMConfig(vocab_size=tok.vocab_size, dim=args.dim, depth=args.depth, mixer=MIXERS[args.mixer](),
                       ffn=nn.SwiGLU(), max_seq_len=args.seq_len * 2, seed=0)
    model = nn.LLM(config=cfg)
    print(model.summary(seq_len=args.seq_len))

    out = Path(args.out) / args.mixer
    tcfg = TrainConfig(max_steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len, lr=2e-3,
                       warmup_steps=max(1, args.steps // 20), precision=args.precision, device=args.device,
                       eval_every=max(1, args.steps // 6), eval_batches=8, log_every=max(1, args.steps // 30),
                       checkpoint_dir=str(out))
    trainer = Trainer(model, train, val, tcfg)
    t0 = time.time()
    history = trainer.fit()
    print(f"trained {args.steps} steps in {time.time() - t0:.1f}s on {trainer.device}")

    # --- verify checkpoint reload -----------------------------------------------------
    reloaded = Trainer.load_model(out / "last.pt").to(trainer.device).eval()
    model.eval()
    probe = val.fixed_batches(1, 4, trainer.device)[0]
    with torch.no_grad():
        diff = (reloaded(probe).logits - model(probe).logits).abs().max().item()
    print(f"checkpoint reload: max |Δlogits| = {diff:.3g}")
    assert diff == 0.0, "reloaded model differs from trained model"

    # --- generation -------------------------------------------------------------------
    prompt = "def "
    ids = torch.tensor([tok.encode(prompt)], device=trainer.device)
    greedy = reloaded.generate(ids, max_new_tokens=200, do_sample=False)
    nocache = reloaded.generate(ids, max_new_tokens=200, do_sample=False, use_cache=False)
    sampled = reloaded.generate(ids, max_new_tokens=200, temperature=0.8, top_p=0.9,
                                generator=torch.Generator().manual_seed(0))
    same = torch.equal(greedy, nocache)
    print(f"cached vs uncached greedy decoding identical: {same}")
    print("---- greedy ----\n" + tok.decode(greedy[0].tolist()))
    print("---- sampled (T=0.8, top_p=0.9) ----\n" + tok.decode(sampled[0].tolist()))

    summary = {
        "mixer": args.mixer, "params": model.num_parameters(), "steps": args.steps, "device": str(trainer.device),
        "precision": args.precision, "first_val_loss": next(h["val_loss"] for h in history if "val_loss" in h),
        "final_val_loss": [h["val_loss"] for h in history if "val_loss" in h][-1],
        "final_train_loss": [h["lm_loss"] for h in history if "lm_loss" in h][-1],
        "tokens_per_sec": history[-1].get("tokens_per_sec"), "reload_max_abs_diff": diff,
        "cached_equals_uncached": same,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
