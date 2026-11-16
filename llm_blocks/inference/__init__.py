"""Autoregressive generation that works with any mix of sequence mixers.

The prompt is processed in one forward pass that fills each layer's own decoding state
(KV cache, MLA latent cache, Mamba-2 SSM state, DeltaNet fast weights); subsequent tokens
are fed one at a time. No layer type is assumed to use a KV cache.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

__all__ = ["generate", "sample_next_token"]


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample from ``(batch, vocab)`` logits. ``temperature == 0`` means greedy."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature
    if top_k is not None and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(-1)
        # remove tokens once the cumulative probability *before* them exceeds top_p
        remove = (probs.cumsum(-1) - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = logits.softmax(-1)
    if probs.device.type == "mps" and generator is not None:  # MPS generators can't drive multinomial on CPU RNG
        return torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(-1).to(probs.device)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.inference_mode()
def generate(
    model,
    input_ids: Tensor,
    max_new_tokens: int = 32,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    do_sample: bool = True,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    attention_mask: Optional[Tensor] = None,
    use_cache: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Generate ``max_new_tokens`` tokens after ``input_ids`` (``(batch, prompt_len)``).

    Batched prompts of different lengths must be **left**-padded with ``attention_mask``.
    ``do_sample=False`` or ``temperature=0`` gives greedy decoding. Sequences that emit
    ``eos_token_id`` are padded with ``pad_token_id`` (default: ``eos_token_id``) afterwards.
    ``use_cache=False`` recomputes the full sequence every step (reference / testing).

    Returns ``(batch, prompt_len + new_tokens)``.
    """
    was_training = model.training
    model.eval()
    try:
        temp = temperature if do_sample else 0.0
        B, T = input_ids.shape
        device = input_ids.device
        pad = pad_token_id if pad_token_id is not None else eos_token_id
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        out = input_ids

        if use_cache:
            state = model.init_state(B, T + max_new_tokens, device=device)
            logits = model(input_ids, attention_mask=attention_mask, state=state, logits_to_keep=1).logits[:, -1]
        mask = attention_mask
        for step in range(max_new_tokens):
            if not use_cache:
                logits = model(out, attention_mask=mask, logits_to_keep=1).logits[:, -1]
            nxt = sample_next_token(logits, temperature=temp, top_k=top_k, top_p=top_p, generator=generator)
            if eos_token_id is not None:
                nxt = torch.where(finished, torch.full_like(nxt, pad), nxt)
                finished = finished | (nxt == eos_token_id)
            out = torch.cat([out, nxt[:, None]], dim=1)
            if mask is not None:
                mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
            if use_cache and step + 1 < max_new_tokens:
                logits = model(nxt[:, None], state=state).logits[:, -1]
        return out
    finally:
        model.train(was_training)
