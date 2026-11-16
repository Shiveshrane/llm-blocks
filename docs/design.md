# Design

## Specs vs. modules

Everything a user composes (`nn.GQA`, `nn.SwiGLU`, `nn.RMSNorm`, `nn.Block`, `nn.LLMConfig`...) is a
**spec**: a frozen, keyword-only dataclass that describes a component and holds no parameters.
`spec.build(ctx)` turns it into an ordinary `torch.nn.Module`, where `ctx` (`BuildContext`) supplies
what the user should not have to repeat: model width, layer index, depth, max sequence length,
init std.

Why:

* `nn.GQA(heads=12, kv_heads=4)` needs no `dim` — the model provides it. Swapping a mixer is a
  one-argument change.
* The same spec can be reused for many layers; every `build` creates fresh parameters.
* Specs compare by value, serialize to JSON (`to_dict` / `spec_from_dict`, registry keyed by class
  name), and are edited with `replace(...)`. That is what makes ablations reproducible: a run is
  identified by `config.fingerprint()`.
* Subclassing a spec base (`MixerSpec`, `FFNSpec`, ...) registers it — user-defined components
  serialize like built-in ones (see `examples/custom_mixer.py`).

The module classes (`Attention`, `MultiHeadLatentAttention`, `Mamba2Mixer`, `GatedDeltaNetMixer`,
`MoELayer`, ...) are plain PyTorch and can be used without specs.

## Interfaces

### `SequenceMixer`

```python
y = mixer(x, state=None, padding_mask=None)        # x, y: (B, T, D)
state = mixer.init_state(batch, max_len, device=..., dtype=...)
```

* **Training / full sequence**: `state=None`; the mixer must be causal.
* **Incremental decoding**: the caller allocates a state once and then feeds any chunking of the
  sequence (whole prompt, then one token at a time). The mixer reads and updates **its own** state
  in place. Nothing outside the mixer inspects it:

  | mixer | state | size per sequence |
  |---|---|---|
  | MHA / GQA / MQA | `KVCache`: K and V, pre-allocated | `2 · kv_heads · head_dim · T` |
  | MLA | `MLACache`: normalized latent `c_kv` + shared rotary key | `(kv_lora_rank + rope_dim) · T` |
  | Mamba-2 | `Mamba2State`: conv window + SSM state (fp32) | `O(1)` in T |
  | Gated DeltaNet | `GatedDeltaNetState`: conv window + fast-weight matrix (fp32) | `O(1)` in T |

  No mixer is forced into a KV-cache abstraction; the model-level `ModelState` is just a list of
  per-layer states.
* **Padding**: `padding_mask` (`True` = real token) supports left padding (batched generation) and
  right padding (variable-length training). Attention masks padded keys; recurrent mixers zero the
  pre-convolution inputs and set the decay/write strength to identity at padded steps, so padded
  positions leave the state untouched. Left-padded results equal unpadded results exactly
  (tested for every mixer).
* **Positions**: position encodings are owned by the mixer that uses them. Attention accepts
  `RoPE` / `ALiBi` / none, MLA accepts (decoupled) `RoPE` / none, Mamba-2 and Gated DeltaNet accept
  none. Learned absolute embeddings are model-level (`nn.LLM(position=nn.LearnedPositions())`).
  Anything else raises `SpecError` with an explanation.

### `FeedForward`

`ffn(x, ctx)`. `ctx` is a per-forward `ForwardContext` that collects auxiliary losses and metrics,
so MoE needs no hidden module state; `LLM.forward` adds them to the loss and returns them in
`LMOutput`. FFN specs that can be MoE experts implement `build_experts(ctx, E)`, returning a
stacked-weight module with `forward_batched` (one `bmm` for all experts) and `forward_expert`.

### Residual strategies

A residual mechanism has a model-level **stream type** (`ResidualStream.init/finalize`) and a
per-sublayer **strategy** (`strategy(stream, fn) -> stream`, where `fn` is the pre-normalized
sublayer). The stream is a `(B,T,D)` tensor for standard/gated residuals, `(B,T,n,D)` for
hyper-connections / mHC, and a list of all previous layer outputs for attention residuals. Mixing
stream types in one model is rejected. See [residuals.md](residuals.md).

### Block

`Block(mixer, ffn=SwiGLU(), norm=RMSNorm(), residual=Residual(), norm_position="pre")`.
Pre-norm: `s = R(s, λh: mixer(norm(h)))`, then the same for the FFN. Post-norm (`norm(x + f(x))`) is
supported for tensor-stream residuals only. `ffn=None` gives mixer-only blocks (e.g. pure Mamba
stacks).

## Performance choices

* Attention: fused QKV projection; `F.scaled_dot_product_attention` with `is_causal` (no materialized
  mask) for training and prefill, no mask for single-token decode, native GQA (`enable_gqa`, no K/V
  repetition). A boolean / additive mask is only built for padding, ALiBi or chunked prefill.
  On CUDA, SDPA dispatches to FlashAttention / memory-efficient kernels.
* SwiGLU/GeGLU: gate and up projections share one GEMM.
* RMSNorm: `F.rms_norm` (fused op) in fp32.
* KV caches are pre-allocated (no `torch.cat` growth).
* MoE: router in fp32 outside autocast; dropless dispatch sorts tokens by expert (one host sync per
  layer); capacity dispatch has static shapes and no host sync.
* Mamba-2 / Gated DeltaNet: chunked algorithms (matmul-heavy, parallel across the sequence within a
  chunk) for training and prefill, O(1) recurrent steps for decoding. Pure PyTorch — no Triton.
* The abstraction itself costs nothing measurable against a hand-written model with the same
  weights (see the benchmarks in the README).
