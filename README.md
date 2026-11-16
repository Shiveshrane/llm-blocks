# llm-blocks

Composable building blocks for LLM architecture research in PyTorch. Describe a model as a stack
of interchangeable parts — sequence mixer, feed-forward, normalization, position encoding,
residual mechanism — and swap any one of them without touching the rest of the architecture or
the training/generation code.

```python
from llm_blocks import nn

model = nn.LLM(
    vocab_size=32000,
    dim=768,
    layers=[
        nn.Block(
            mixer=nn.GQA(heads=12, kv_heads=4, position=nn.RoPE()),
            ffn=nn.SwiGLU(hidden_dim=2048),
            norm=nn.RMSNorm(),
        )
        for _ in range(12)
    ],
)
print(model.summary())          # per-layer table, parameter breakdown, active params, FLOPs/token
print(model.num_parameters())   # 124,668,672
print(model.architecture())     # collapsed tree of the layer stack
model.visualize("arch.svg")     # SVG block diagram (or fmt="mermaid")
```

Swap one component:

```python
mixer=nn.MLA(heads=12)                                   # multi-head latent attention
mixer=nn.Mamba2()                                        # SSM
mixer=nn.GatedDeltaNet()                                 # linear attention w/ delta rule
ffn=nn.MoE(expert=nn.SwiGLU(hidden_dim=1024), num_experts=16, top_k=2, shared_experts=1)
residual=nn.GatedResidual() | nn.HyperConnections(n=4) | nn.MHC(n=4) | nn.AttnRes()
```

Ablations from a base config:

```python
base = nn.LLMConfig(vocab_size=32000, dim=1024, depth=12,
                    mixer=nn.GQA(heads=16, kv_heads=4), ffn=nn.SwiGLU(hidden_dim=4096))

experiment_a = base.replace(mixer=nn.MLA(heads=16))
experiment_b = base.replace(ffn=nn.MoE(expert=nn.SwiGLU(hidden_dim=1024), num_experts=8, top_k=2))
experiment_c = base.replace_layers(nn.every(4, start=3), mixer=nn.Mamba2())   # every 4th layer -> Mamba-2
experiment_d = base.replace(residual=nn.GatedResidual())

experiment_c.save("c.json"); nn.LLMConfig.load("c.json") == experiment_c   # True
experiment_c.fingerprint()                                                 # stable hash for bookkeeping
```

and train / compare them under identical conditions:

```python
from llm_blocks.training import TrainConfig, run_ablation, results_table
results = run_ablation({"gqa": base, "mla": experiment_a, "moe": experiment_b}, train_data, val_data,
                       TrainConfig(max_steps=2000, batch_size=32, seq_len=512))
print(results_table(results))
```

Generation works with any mix of mixers — each layer keeps its own decoding state (KV cache, MLA
latent cache, Mamba-2 SSM state, DeltaNet fast weights):

```python
tokens = model.generate(input_ids, max_new_tokens=128, temperature=0.8, top_p=0.9)
```

## Install

```bash
pip install -e .            # requires torch>=2.5
pip install -e ".[dev]"     # pytest
pip install -e ".[reference]"  # transformers, for the reference-implementation tests
```

## What's in the box

| Family | Components |
|---|---|
| Sequence mixers | `MHA`, `MQA`, `GQA`, `MLA`, `Mamba2`, `GatedDeltaNet` (+ your own: [examples/custom_mixer.py](examples/custom_mixer.py)) |
| Feed-forward | `FFN` (`GeluFFN`, `ReluFFN`), `GLU`, `SwiGLU`, `GeGLU`, `MoE` (top-k, softmax/sigmoid routers, shared experts, capacity or dropless dispatch, balance/z losses, aux-loss-free bias) |
| Normalization | `RMSNorm`, `LayerNorm`, `QKNorm` (`qk_norm=True`), pre-/post-norm blocks |
| Positions | `RoPE` (incl. partial), `ALiBi`, `LearnedPositions` (model-level), `NoPE` — incompatible combinations raise `SpecError` |
| Residuals | `Residual`, `GatedResidual`, `HyperConnections`, `MHC`, `AttnRes` |
| Training | `Trainer`: AdamW, warmup + cosine, grad accumulation, clipping, BF16 autocast, compile, activation checkpointing, exact resume |
| Research tools | `summary()`, `parameter_breakdown()`, `num_active_parameters()`, `flops_per_token()`, benchmarks, `run_ablation`, JSON configs |

Accurate, detailed status (what is tested, what is partial, what is missing): **[docs/STATUS.md](docs/STATUS.md)**.
Design and interfaces: [docs/design.md](docs/design.md). Residual math: [docs/residuals.md](docs/residuals.md).
FLOP assumptions: [docs/flops.md](docs/flops.md).

## Correctness

`pytest` runs 248 tests in ~12 s (CPU + MPS). Highlights:

* **Reference implementations** (HuggingFace transformers, float64 on both sides): a 3-layer
  GQA/RoPE/SwiGLU model reproduces `LlamaForCausalLM` logits; `nn.MLA` reproduces
  `DeepseekV3Attention` (full and absorbed-decoding paths); `nn.Mamba2` reproduces `Mamba2Mixer`;
  the gated delta rule reproduces Qwen3-Next's chunked and recurrent kernels. Agreement is
  ≤1e-6 (bounded by float32 pieces inside the references).
* **Incremental decoding == full causal pass** for every mixer under arbitrary chunkings, and for a
  hybrid GQA + Mamba-2 + GatedDeltaNet + MLA + MoE model; cached and uncached greedy generation
  produce identical tokens; left-padded batched generation equals per-prompt generation.
* Causality, left/right padding, float64 gradchecks, BF16 vs FP32, CPU vs MPS, gradient
  checkpointing equivalence, exact checkpoint resume, gradient accumulation equivalence,
  serialization round-trips, error messages.

## Measured results

All numbers below are from one Apple M5 laptop (MPS / CPU), PyTorch 2.15. Nothing has been run on
CUDA yet.

### Overhead vs. a hand-written model

[benchmarks/compare_reference.py](benchmarks/compare_reference.py) builds a GQA/RoPE/SwiGLU model
with llm-blocks and a monolithic hand-written PyTorch Llama ([benchmarks/reference_llama.py](benchmarks/reference_llama.py)),
copies the weights across, checks the logits match (**max diff 0** on MPS in fp32), then times them
(54.5M params, dim 512, 8 layers, vocab 32k, batch 8 × 512 tokens, medians):

| MPS | llm-blocks | hand-written | ratio |
|---|---:|---:|---:|
| train step, bf16 autocast | 467 ms (8.8k tok/s) | 475 ms (8.6k tok/s) | 0.98 |
| train step, bf16 + `torch.compile` | 245 ms (16.8k tok/s) | 246 ms (16.7k tok/s) | 0.99 |
| train step, fp32 | 735 ms | 736 ms | 1.00 |
| prefill 128 tokens, bf16 | 5.8 ms | 5.8 ms | 1.01 |
| decode, bf16 | 3.83 ms/token | 3.75 ms/token | 1.02 |
| decode, fp32 | 2.43 ms/token | 2.46 ms/token | 0.99 |

The abstraction adds no measurable training overhead and ≈0–6% per-token decode overhead (Python
bookkeeping of the state objects at batch size 1). Raw results: `benchmarks/results/*.json`.

### Tiny end-to-end run

[examples/train_tiny.py](examples/train_tiny.py): byte-level LM (4.36M params, 6 × GQA + SwiGLU,
dim 256) on 4.3 MB of Python standard-library source, 600 steps × 8,192 tokens, bf16 on MPS
(146 s, ~34k tok/s). Validation loss 2.00 → 1.26 nats/byte. The reloaded checkpoint reproduces the
trained model's logits exactly, and cached / uncached greedy decoding agree.

### Ablation

[examples/ablation.py](examples/ablation.py) trains 8 variants of one base config (6 layers, dim 192,
GQA 6/2 heads + SwiGLU, byte-level, stdlib Python corpus), each changing **one** component, with
identical data order, optimizer, schedule and init seed: 300 steps × 16 × 256 tokens (1.2M tokens),
bf16 on MPS. Validation on 16 fixed batches.

| variant | params | active | GFLOPs/tok (fwd) | train loss | val loss | best val | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_gqa | 2.46M | 2.46M | 0.0054 | 1.8610 | 1.7352 | 1.7352 | 44,133 |
| mla | 2.89M | 2.89M | 0.0064 | 1.9339 | 1.9980 | 1.9980 | 40,692 |
| hybrid_mamba2_every2 | 2.88M | 2.88M | 0.0062 | 1.4679 | 1.3772 | 1.3772 | 29,207 |
| all_gated_deltanet | 3.07M | 3.07M | 0.0064 | 1.3963 | 1.3294 | 1.3294 | 25,340 |
| moe_8x_top2 | 7.78M | 2.47M | 0.0054 | 1.8470 | 1.7416 | 1.7416 | 19,003 |
| gated_residual | 2.46M | 2.46M | 0.0054 | 1.8531 | 1.7370 | 1.7370 | 41,366 |
| mhc_n4 | 2.69M | 2.69M | 0.0054 | 1.8597 | 1.7245 | 1.7245 | 14,770 |
| attn_res | 2.46M | 2.46M | 0.0054 | 1.8214 | 1.6974 | 1.6974 | 14,816 |

How to read this: it is a **single-seed, 1.2M-token smoke test**, not evidence about which
architecture is better. Variants are not parameter-matched (MLA, Mamba-2 and Gated DeltaNet layers
are larger at these widths; MoE has 3× total but equal active parameters). At this tiny scale the
recurrent mixers' short convolutions are a strong local bias for byte-level text, which likely
explains their much faster early loss drop. MLA at `heads=6` with derived dims was the worst here;
it was not tuned. Throughput reflects pure-PyTorch kernels on MPS: the Mamba-2 / DeltaNet chunk
loops, the dropless MoE expert loop and the extra mixing of mHC / AttnRes have no fused kernels,
so the tok/s column says nothing about how they would perform with optimized kernels on GPU.
Per-variant JSON (configs, fingerprints, full loss curves) lands in `runs/ablation/`.

## Layout

```
llm_blocks/
  configs/spec.py         Spec base class, BuildContext, JSON registry
  nn/base.py              SequenceMixer, FeedForward, MixerState, ForwardContext, spec bases
  nn/attention/           MHA/GQA/MQA (+KVCache), MLA (+latent cache)
  nn/mixers/              Mamba-2, Gated DeltaNet, causal conv
  nn/ffn/  nn/moe/        dense / gated FFNs, stacked experts, MoE
  nn/norm/ nn/position/   norms, RoPE / ALiBi / learned / NoPE
  nn/residual/            Residual, GatedResidual, HyperConnections, MHC, AttnRes
  nn/blocks/ nn/models/   Block, LLMConfig, LLM
  kernels/                pure-PyTorch SSD and gated-delta-rule kernels (chunked + recurrent)
  training/               data, Trainer, ablation runner
  inference/              generate / sampling
  utils/                  summary, FLOPs, benchmarks
  visualization/          SVG / Mermaid diagrams
examples/  benchmarks/  tests/  docs/
```

## Limitations (short version)

No CUDA validation yet; no Triton / fused kernels for Mamba-2 or Gated DeltaNet (pure-PyTorch
chunked algorithms instead); no distributed training; dropless MoE loops over experts and breaks
`torch.compile` graphs (capacity mode does not); mHC and AttnRes are implemented from their papers
without comparison against official code. Not production-ready. Full list in
[docs/STATUS.md](docs/STATUS.md).

Attributions: [NOTICE](NOTICE).
