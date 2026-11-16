"""Pure-PyTorch state-space-duality (SSD) scan for Mamba-2.

The chunked algorithm follows ``ssd_minimal_discrete`` from the official Mamba repository
(state-spaces/mamba, ``mamba_ssm/modules/ssd_minimal.py``, Apache-2.0, Tri Dao & Albert Gu),
rewritten without einops and extended with an optional initial state. It computes, for each
head, the selective SSM

    h_t = exp(a_t) * h_{t-1} + x_t ⊗ B_t        (h: headdim × d_state)
    y_t = h_t · C_t

with ``x_t`` already multiplied by ``dt`` and ``a_t = A * dt``. All math runs in float32.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ._dtype import acc_dtype

__all__ = ["segsum", "ssd_chunked", "ssd_recurrent", "ssd_step"]


def segsum(x: Tensor) -> Tensor:
    """Stable segment sum: ``out[..., i, j] = sum_{k=j+1..i} x[..., k]`` for ``j <= i``, ``-inf`` above."""
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    return x_segsum.masked_fill(~mask, float("-inf"))


def ssd_chunked(
    X: Tensor, A: Tensor, B: Tensor, C: Tensor, chunk_size: int, initial_state: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """Chunked SSD.

    Args:
        X: ``(b, l, h, p)`` inputs (already scaled by dt).
        A: ``(b, l, h)`` log-decays (``A * dt``, non-positive).
        B, C: ``(b, l, h, n)``.
        initial_state: optional ``(b, h, p, n)``.
    Returns:
        ``Y (b, l, h, p)`` and the final state ``(b, h, p, n)``.
    """
    dtype = X.dtype
    acc = acc_dtype(X)
    X, A, B, C = X.to(acc), A.to(acc), B.to(acc), C.to(acc)
    b, L, h, p = X.shape
    n = B.shape[-1]
    pad = (chunk_size - L % chunk_size) % chunk_size
    if pad:
        # Zero inputs with zero log-decay leave the state unchanged, so padding is exact.
        X = F.pad(X, (0, 0, 0, 0, 0, pad))
        A = F.pad(A, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
    c = X.shape[1] // chunk_size
    X = X.reshape(b, c, chunk_size, h, p)
    B = B.reshape(b, c, chunk_size, h, n)
    C = C.reshape(b, c, chunk_size, h, n)
    A = A.reshape(b, c, chunk_size, h).permute(0, 3, 1, 2)  # (b, h, c, l)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. intra-chunk (diagonal blocks)
    Lmat = torch.exp(segsum(A))  # (b, h, c, l, s)
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, Lmat, X)

    # 2. state at the end of each chunk, from inputs inside the chunk
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)  # (b, h, c, l)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. inter-chunk recurrence on chunk states
    if initial_state is None:
        initial_state = torch.zeros_like(states[:, :1])
    else:
        initial_state = initial_state.to(acc).unsqueeze(1)
    states = torch.cat([initial_state, states], dim=1)  # (b, c+1, h, p, n)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[..., -1], (1, 0))))  # (b, h, c+1, c+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. contribution of the carried-in state to each position
    state_decay_out = torch.exp(A_cumsum)  # (b, h, c, l)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    Y = (Y_diag + Y_off).reshape(b, c * chunk_size, h, p)[:, :L]
    return Y.to(dtype), final_state


def ssd_step(state: Tensor, x: Tensor, a: Tensor, B: Tensor, C: Tensor) -> Tuple[Tensor, Tensor]:
    """One recurrent step. ``state (b,h,p,n)``, ``x (b,h,p)``, ``a (b,h)``, ``B,C (b,h,n)``."""
    acc = acc_dtype(x)
    state = state * torch.exp(a.to(acc))[..., None, None] + x.to(acc)[..., None] * B.to(acc)[:, :, None, :]
    y = torch.einsum("bhpn,bhn->bhp", state, C.to(acc))
    return y, state


def ssd_recurrent(X: Tensor, A: Tensor, B: Tensor, C: Tensor, initial_state: Optional[Tensor] = None):
    """Token-by-token reference implementation (slow; used for testing)."""
    b, L, h, p = X.shape
    n = B.shape[-1]
    acc = acc_dtype(X)
    state = torch.zeros(b, h, p, n, device=X.device, dtype=acc) if initial_state is None else initial_state.to(acc)
    ys = []
    for t in range(L):
        y, state = ssd_step(state, X[:, t], A[:, t], B[:, t], C[:, t])
        ys.append(y)
    return torch.stack(ys, dim=1).to(X.dtype), state
