"""Pure-PyTorch gated delta rule (Gated DeltaNet, Yang, Kautz & Hatamizadeh, ICLR 2025).

Per head, with fast-weight state ``S`` of shape ``(d_k, d_v)``:

    S_t = exp(g_t) * S_{t-1}
    S_t = S_t + k_t ⊗ beta_t * (v_t - S_t^T k_t)        # delta rule correction
    o_t = S_t^T q_t

i.e. ``S_t = exp(g_t) (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T``.

The chunked form uses the WY / UT-transform representation (Yang et al., 2024, "Parallelizing
Linear Transformers with the Delta Rule over Sequence Length"): inside a chunk the pseudo-values
``u = T (beta*v - beta*exp(G) * K S_0)`` with ``T = (I + A)^{-1}`` for the strictly lower
triangular ``A_ts = beta_t exp(G_t - G_s) k_t·k_s`` are obtained from one triangular solve,
after which outputs and the next state are plain matmuls. Chunks are scanned sequentially.

Inputs are expected L2-normalised (q, k) with ``q`` already scaled; math runs in float32.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ._dtype import acc_dtype

__all__ = ["gated_delta_rule_chunked", "gated_delta_rule_recurrent", "gated_delta_rule_step"]


def gated_delta_rule_step(state: Tensor, q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor):
    """One token. ``state (b,h,dk,dv)``, ``q,k (b,h,dk)``, ``v (b,h,dv)``, ``g,beta (b,h)``."""
    acc = acc_dtype(v)
    state = state * torch.exp(g.to(acc))[..., None, None]
    kv_mem = torch.einsum("bhkv,bhk->bhv", state, k.to(acc))
    delta = (v.to(acc) - kv_mem) * beta.to(acc)[..., None]
    state = state + k.to(acc)[..., :, None] * delta[..., None, :]
    o = torch.einsum("bhkv,bhk->bhv", state, q.to(acc))
    return o, state


def gated_delta_rule_recurrent(q, k, v, g, beta, initial_state: Optional[Tensor] = None):
    """Token-by-token reference. ``q,k (b,l,h,dk)``, ``v (b,l,h,dv)``, ``g,beta (b,l,h)``."""
    b, L, h, dk = k.shape
    dv = v.shape[-1]
    acc = acc_dtype(v)
    state = (torch.zeros(b, h, dk, dv, device=q.device, dtype=acc)
             if initial_state is None else initial_state.to(acc))
    outs = []
    for t in range(L):
        o, state = gated_delta_rule_step(state, q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t])
        outs.append(o)
    return torch.stack(outs, dim=1).to(v.dtype), state


def gated_delta_rule_chunked(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
    chunk_size: int = 64, initial_state: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Chunk-parallel gated delta rule; same contract as :func:`gated_delta_rule_recurrent`."""
    dtype = v.dtype
    acc = acc_dtype(v)
    b, L, h, dk = k.shape
    dv = v.shape[-1]
    # -> (b, h, l, d) float32
    q, k, v = (x.transpose(1, 2).to(acc) for x in (q, k, v))
    g, beta = (x.transpose(1, 2).to(acc) for x in (g, beta))
    pad = (chunk_size - L % chunk_size) % chunk_size
    if pad:  # zero k/v/beta and zero log-decay: padding leaves the state untouched
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = (F.pad(x, (0, pad)) for x in (g, beta))
    n = q.shape[2] // chunk_size
    q, k, v = (x.reshape(b, h, n, chunk_size, x.shape[-1]) for x in (q, k, v))
    g, beta = (x.reshape(b, h, n, chunk_size) for x in (g, beta))

    G = g.cumsum(-1)  # (b,h,n,c) cumulative log-decay within the chunk
    incl = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).tril()
    # decay[t, s] = exp(G_t - G_s) for s <= t (masked before exp to avoid overflow)
    decay = (G[..., :, None] - G[..., None, :]).masked_fill(~incl, float("-inf")).exp()

    k_beta = k * beta[..., None]
    A = (k_beta @ k.transpose(-1, -2)) * decay  # diagonal ignored by unitriangular solve
    u0 = torch.linalg.solve_triangular(A, v * beta[..., None], upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(A, k_beta * G.exp()[..., None], upper=False, unitriangular=True)
    attn = (q @ k.transpose(-1, -2)) * decay  # inclusive causal intra-chunk scores
    q_dec = q * G.exp()[..., None]
    k_dec = k * (G[..., -1:] - G).exp()[..., None]
    chunk_decay = G[..., -1].exp()[..., None, None]  # (b,h,n,1,1)

    S = (torch.zeros(b, h, dk, dv, device=q.device, dtype=acc)
         if initial_state is None else initial_state.to(acc))
    outs = []
    for i in range(n):
        u = u0[:, :, i] - w[:, :, i] @ S
        outs.append(q_dec[:, :, i] @ S + attn[:, :, i] @ u)
        S = S * chunk_decay[:, :, i] + k_dec[:, :, i].transpose(-1, -2) @ u
    o = torch.stack(outs, dim=2).reshape(b, h, n * chunk_size, dv)[:, :, :L]
    return o.transpose(1, 2).to(dtype), S
