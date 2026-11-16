# Residual mechanisms

All formulas are for one sublayer `f` (mixer or FFN, including its pre-norm). Each sublayer counts
as one "layer" for mechanisms that index layers.

## `Residual()` — standard
`x' = x + f(x)`.

## `GatedResidual(gate, init)`
`x' = x + g ⊙ f(x)`

* `gate="scalar"`: one learned scalar, initialised to `init`.
* `gate="channel"`: learned per-channel vector (LayerScale; ReZero with `init=0`).
* `gate="input"`: `g = sigmoid(W x + b)`, `W = 0`, `b = logit(init)` at initialisation.

## `HyperConnections(n, dynamic=True)` — Zhu et al., *Hyper-Connections*, ICLR 2025 (arXiv:2409.19606)
Stream `H ∈ ℝ^{n×D}` per token (embedding replicated `n` times; the final hidden state is the sum
of the streams). With `H̄ = RMSNorm(H)`:

```
A_m = A_m0 + s_α · tanh(H̄ w_m)      ∈ ℝ^n       layer input  h = A_mᵀ H
A_r = A_r0 + s_α · tanh(H̄ W_r)      ∈ ℝ^{n×n}   residual mix
B   = B_0  + s_β · tanh(H̄ w_b)      ∈ ℝ^n       output distribution
H'  = A_rᵀ H + Bᵀ f(h)
```
Init: `A_m0 = e_{layer mod n}`, `A_r0 = I`, `B_0 = 1`, dynamic weights `0`, `s_α = s_β = 0.01`.
`dynamic=False` keeps only the static parts. Tested: with `n=1` the model is exactly a standard
pre-norm residual network; with `n=4` at init it produces the same logits as the standard model.

## `MHC(n, sinkhorn_iters=20)` — DeepSeek-AI, *mHC: Manifold-Constrained Hyper-Connections* (arXiv:2512.24880)
```
x' = RMSNorm(vec(x))                                   x ∈ ℝ^{n×C}
H_pre  = σ(α_pre  · x'φ_pre  + b_pre)                  ∈ ℝ^{1×n}
H_post = 2σ(α_post · x'φ_post + b_post)                ∈ ℝ^{1×n}
H_res  = SinkhornKnopp(α_res · mat(x'φ_res) + b_res)    ∈ ℝ^{n×n}
x_next = H_res x + H_postᵀ f(H_pre x)
```
Sinkhorn-Knopp exponentiates and alternately normalises rows and columns (`sinkhorn_iters`
iterations, columns last), so `H_res` is (approximately) doubly stochastic; this is the constraint
that distinguishes mHC from unconstrained HC.

Initialisation is **our choice** (we could not verify the paper's exact values): `φ = 0`,
`α = 0.01`, `b_pre = logit(1/n)` (input = mean of streams), `b_post = 0` (`H_post = 1`),
`b_res = 0` on the diagonal and `−8` off-diagonal (near-identity mixing). There is no official
open-source implementation to compare against; tests check the constraints and that the forward
pass equals the formula above.

## `AttnRes()` — Kimi Team, *Attention Residuals* (arXiv:2603.15031), "Full AttnRes" variant
The stream is the list of all previous outputs `[v_0 = embedding, v_1, ..., v_{l-1}]`.
```
h_l = Σ_i softmax_i( w_l · RMSNorm(v_i) ) v_i         w_l ∈ ℝ^D learned pseudo-query
v_l = f_l(h_l)                                          appended to the stream
```
The final hidden state is the same depth-attention with its own pseudo-query. `w = 0` at init
(uniform averaging). Keys use a parameter-free RMSNorm (our choice). The memory-saving *Block*
AttnRes variant is not implemented. Implemented from the paper's description; an official
implementation exists (github.com/MoonshotAI/Attention-Residuals) but this version has **not** been
compared against it yet — treat details (key normalization, final aggregation) as unverified.

## Constraints
* All blocks in one model must use the same stream type (`tensor`, `hyper(n)`, `depth`).
* Post-norm is only defined for tensor-stream residuals.
