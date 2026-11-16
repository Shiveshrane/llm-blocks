"""Analytical FLOP estimates.

Assumptions (also in docs/flops.md):

* Only matrix multiplications are counted, at 2 FLOPs per multiply-accumulate. Norms,
  activations, softmax, rotary embedding, routing top-k and residual adds are ignored.
* Causal attention at context ``seq_len`` is averaged over positions: a token attends to
  ``(seq_len + 1) / 2`` keys on average (QK^T and PV both counted).
* Recurrent mixers (Mamba-2, Gated DeltaNet) are counted in their recurrent form
  (state-size work per token), which is a lower bound on the chunked training algorithm.
* MoE counts the router plus ``top_k`` routed experts (plus shared experts).
* Training FLOPs ≈ 3 × forward (backward ≈ 2 × forward); activation recomputation from
  gradient checkpointing adds another forward.
* The embedding lookup is free; the LM head is a matmul.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..nn.models.llm import LLM

__all__ = ["forward_flops_per_token", "training_flops"]


def forward_flops_per_token(model: "LLM", seq_len: int) -> float:
    total = 2.0 * model.config.dim * model.config.vocab_size  # LM head
    for layer in model.layers:
        total += layer.mixer.flops_per_token(seq_len)  # type: ignore[union-attr]
        if layer.ffn is not None:  # type: ignore[union-attr]
            total += layer.ffn.flops_per_token()  # type: ignore[union-attr]
    return total


def training_flops(model: "LLM", seq_len: int, tokens: int, recompute: bool = False) -> float:
    """Approximate total training FLOPs for ``tokens`` tokens at context ``seq_len``."""
    mult = 4.0 if recompute else 3.0
    return mult * forward_flops_per_token(model, seq_len) * tokens
