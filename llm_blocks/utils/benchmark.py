"""Throughput, latency and memory measurement helpers.

All timings synchronize the device before reading the clock and exclude warm-up steps.
Memory: on CUDA the peak allocated memory is reported; on MPS PyTorch exposes no peak
counter, so the driver-allocated memory after the run is reported instead (an upper bound
that includes the caching allocator); on CPU memory is not reported.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
from torch import nn

from ..training.trainer import autocast_context, resolve_device

__all__ = ["synchronize", "benchmark_training", "benchmark_inference", "memory_snapshot_mb"]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _reset_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        torch.mps.empty_cache()


def memory_snapshot_mb(device: torch.device) -> Optional[float]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return None


def benchmark_training(
    model: nn.Module,
    *,
    batch_size: int = 8,
    seq_len: int = 512,
    steps: int = 10,
    warmup: int = 3,
    precision: str = "bf16",
    device: str = "auto",
    compile: bool = False,
    vocab_size: Optional[int] = None,
    loss_fn=None,
) -> Dict[str, Any]:
    """Time forward + backward + AdamW step on random tokens."""
    dev = resolve_device(device)
    model = model.to(dev).train()
    vocab = vocab_size or model.config.vocab_size  # type: ignore[attr-defined]
    fwd = torch.compile(model) if compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=(dev.type == "cuda") or None)
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(0, vocab, (batch_size, seq_len), generator=gen).to(dev)

    def step():
        with autocast_context(dev, precision):
            out = fwd(x, labels=x)
            loss = out.loss if loss_fn is None else loss_fn(out, x)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(warmup):
        step()
    synchronize(dev)
    _reset_memory(dev)
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    synchronize(dev)
    dt = (time.perf_counter() - t0) / steps
    return {
        "device": dev.type,
        "precision": precision,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "ms_per_step": dt * 1e3,
        "tokens_per_sec": batch_size * seq_len / dt,
        "memory_mb": memory_snapshot_mb(dev),
    }


@torch.inference_mode()
def benchmark_inference(
    model,
    *,
    batch_size: int = 1,
    prompt_len: int = 128,
    new_tokens: int = 64,
    precision: str = "bf16",
    device: str = "auto",
    warmup: int = 1,
) -> Dict[str, Any]:
    """Measure prefill latency and per-token decode latency with the model's own decoding states."""
    dev = resolve_device(device)
    model = model.to(dev).eval()
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(0, model.config.vocab_size, (batch_size, prompt_len), generator=gen).to(dev)

    def run():
        state = model.init_state(batch_size, prompt_len + new_tokens + 1, device=dev)
        synchronize(dev)
        t0 = time.perf_counter()
        with autocast_context(dev, precision):
            logits = model(x, state=state, logits_to_keep=1).logits
        synchronize(dev)
        t1 = time.perf_counter()
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(new_tokens):
            with autocast_context(dev, precision):
                nxt = model(nxt, state=state).logits[:, -1].argmax(-1, keepdim=True)
        synchronize(dev)
        t2 = time.perf_counter()
        return t1 - t0, (t2 - t1) / new_tokens, state.nbytes()

    for _ in range(warmup):
        run()
    prefill, decode, state_bytes = run()
    return {
        "device": dev.type,
        "precision": precision,
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "prefill_ms": prefill * 1e3,
        "decode_ms_per_token": decode * 1e3,
        "decode_tokens_per_sec": batch_size / decode,
        "state_mb": state_bytes / 2**20,
    }
