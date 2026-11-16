# FLOP estimates

`model.flops_per_token(seq_len)` and the FLOPs line of `model.summary()` are **analytical
estimates of forward-pass FLOPs per token**. Assumptions:

1. Only matrix multiplications are counted, at 2 FLOPs per multiply-accumulate. Norms,
   activations, softmax, rotary embedding, routing top-k, Sinkhorn iterations, residual adds and
   hyper-connection mixing are ignored.
2. Every parameter matrix a token passes through costs `2 × (#params)`.
3. Causal attention at context `T`: a token attends on average to `(T + 1) / 2` keys; `QKᵀ` and
   `PV` are both counted: `2 · 2 · heads · head_dim · (T+1)/2` (MLA uses `qk_head_dim + v_head_dim`).
4. Mamba-2 and Gated DeltaNet are counted in their *recurrent* form (≈3 and ≈4 MACs per state
   element per token respectively, plus projections and the short convolution). The chunked
   training algorithms do more arithmetic than this (intra-chunk quadratic terms), so the estimate
   is a lower bound for training.
5. MoE: router + `top_k` routed experts + shared experts.
6. The embedding lookup is free; the LM head is a `dim × vocab` matmul.
7. Training ≈ 3 × forward (`llm_blocks.utils.flops.training_flops`), 4 × with full activation
   recomputation.

`tests/test_config.py::test_flops_estimate_matches_hand_count` checks the formula on a small model.
