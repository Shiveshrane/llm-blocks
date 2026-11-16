"""Benchmark llm_blocks against a hand-written PyTorch Llama with identical weights.

Checks numerical equivalence first, then measures training-step throughput and
prefill / decode latency. Results are printed and written to benchmarks/results/.

    python benchmarks/compare_reference.py --device mps --precision bf16
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from reference_llama import RefLlama, load_from_llm_blocks  # noqa: E402

from llm_blocks import nn  # noqa: E402
from llm_blocks.training import autocast_context, resolve_device  # noqa: E402
from llm_blocks.utils.benchmark import synchronize  # noqa: E402


def timed(fn, dev, steps, warmup):
    for _ in range(warmup):
        fn()
    synchronize(dev)
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        synchronize(dev)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    dev = resolve_device(args.device)
    hidden = int(8 * args.dim / 3 // 64 * 64)

    model = nn.LLM(vocab_size=args.vocab, dim=args.dim, max_seq_len=4096, layers=[
        nn.Block(mixer=nn.GQA(heads=args.heads, kv_heads=args.kv_heads, position=nn.RoPE()),
                 ffn=nn.SwiGLU(hidden_dim=hidden), norm=nn.RMSNorm(eps=1e-6))] * args.depth).to(dev)
    ref = RefLlama(args.vocab, args.dim, args.depth, args.heads, args.kv_heads, hidden).to(dev)
    load_from_llm_blocks(ref, model)

    # ---- equivalence ------------------------------------------------------------------
    x = torch.randint(0, args.vocab, (2, 64), device=dev)
    with torch.no_grad():
        a = model(x).logits.float()
        b = ref(x)[0].float()
    max_diff = (a - b).abs().max().item()
    print(f"max |logits diff| fp32 = {max_diff:.3g}")

    results = {"device": dev.type, "precision": args.precision, "torch": torch.__version__,
               "machine": platform.machine(), "processor": platform.processor(), "config": vars(args),
               "params": model.num_parameters(), "fp32_max_logit_diff": max_diff, "rows": []}

    # ---- training step ----------------------------------------------------------------
    tokens = torch.randint(0, args.vocab, (args.batch, args.seq), device=dev)
    for name, m in (("llm_blocks", model), ("reference", ref)):
        m.train()
        fwd = torch.compile(m) if args.compile else m
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

        def step():
            with autocast_context(dev, args.precision):
                out = fwd(tokens, labels=tokens)
                loss = out.loss if hasattr(out, "loss") else out[1]
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        t = timed(step, dev, args.steps, 3)
        results["rows"].append({"impl": name, "bench": "train_step", "ms": t * 1e3,
                                "tokens_per_sec": args.batch * args.seq / t})

    # ---- inference --------------------------------------------------------------------
    prompt = torch.randint(0, args.vocab, (1, 128), device=dev)
    model.eval(), ref.eval()

    @torch.inference_mode()
    def ours_prefill():
        st = model.init_state(1, 128 + args.decode_tokens + 1)
        with autocast_context(dev, args.precision):
            model(prompt, state=st, logits_to_keep=1)
        return st

    @torch.inference_mode()
    def ours_decode():
        st = ours_prefill()
        nxt = prompt[:, -1:]
        for _ in range(args.decode_tokens):
            with autocast_context(dev, args.precision):
                nxt = model(nxt, state=st).logits[:, -1].argmax(-1, keepdim=True)

    @torch.inference_mode()
    def ref_prefill():
        caches = [dict() for _ in range(args.depth)]
        with autocast_context(dev, args.precision):
            ref(prompt, caches=caches)
        return caches

    @torch.inference_mode()
    def ref_decode():
        caches = ref_prefill()
        nxt = prompt[:, -1:]
        for i in range(args.decode_tokens):
            with autocast_context(dev, args.precision):
                nxt = ref(nxt, caches=caches, pos=128 + i)[0][:, -1].argmax(-1, keepdim=True)

    for name, pre, dec in (("llm_blocks", ours_prefill, ours_decode), ("reference", ref_prefill, ref_decode)):
        tp = timed(pre, dev, args.steps, 2)
        td = timed(dec, dev, max(3, args.steps // 3), 1)
        results["rows"].append({"impl": name, "bench": "prefill_128", "ms": tp * 1e3})
        results["rows"].append({"impl": name, "bench": f"decode_{args.decode_tokens}_tok", "ms": td * 1e3,
                                "ms_per_token": (td - tp) * 1e3 / args.decode_tokens})

    print(f"\n{'impl':<12} {'bench':<16} {'ms':>9} {'extra':>20}")
    for r in results["rows"]:
        extra = f"{r['tokens_per_sec']:,.0f} tok/s" if "tokens_per_sec" in r else (
            f"{r['ms_per_token']:.2f} ms/tok" if "ms_per_token" in r else "")
        print(f"{r['impl']:<12} {r['bench']:<16} {r['ms']:>9.2f} {extra:>20}")
    rows = {(r["impl"], r["bench"]): r["ms"] for r in results["rows"]}
    for bench in sorted({r["bench"] for r in results["rows"]}):
        ratio = rows[("llm_blocks", bench)] / rows[("reference", bench)]
        results.setdefault("ratio_llm_blocks_over_reference", {})[bench] = ratio
        print(f"ratio llm_blocks/reference {bench}: {ratio:.3f}")
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    tag = f"{dev.type}_{args.precision}{'_compile' if args.compile else ''}"
    (out / f"compare_reference_{tag}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
