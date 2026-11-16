"""A small but complete causal-LM training loop.

Features: AdamW with decoupled weight decay (no decay on norms, biases and SSM scalars),
warmup + cosine/linear/constant LR schedules, gradient accumulation, gradient clipping,
BF16/FP16 autocast, optional ``torch.compile`` and activation checkpointing, periodic
validation, and checkpoints that capture model, optimizer, scheduler, data-sampler and RNG
state so a resumed run continues exactly where it stopped.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from ..nn.models.config import LLMConfig
from ..nn.models.llm import LLM
from .data import TokenDataset

__all__ = ["TrainConfig", "Trainer", "set_seed", "resolve_device", "autocast_context"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_PRECISIONS = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def autocast_context(device: torch.device, precision: str):
    if precision not in _PRECISIONS:
        raise ValueError(f"precision must be one of {list(_PRECISIONS)}, got {precision!r}")
    dtype = _PRECISIONS[precision]
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


@dataclass
class TrainConfig:
    """Training hyper-parameters (JSON-serializable)."""

    max_steps: int = 1000
    batch_size: int = 16
    seq_len: int = 256
    grad_accum_steps: int = 1
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    schedule: str = "cosine"  # cosine | linear | constant
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: Optional[float] = 1.0
    precision: str = "bf16"  # fp32 | bf16 | fp16
    device: str = "auto"
    seed: int = 0
    eval_every: int = 100
    eval_batches: int = 8
    log_every: int = 10
    checkpoint_dir: Optional[str] = None
    checkpoint_every: Optional[int] = None
    compile: bool = False
    gradient_checkpointing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        d = dict(d)
        if "betas" in d:
            d["betas"] = tuple(d["betas"])
        return cls(**d)


def lr_lambda(cfg: TrainConfig) -> Callable[[int], float]:
    def f(step: int) -> float:
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        if cfg.schedule == "constant":
            return 1.0
        decay_steps = max(1, cfg.max_steps - cfg.warmup_steps)
        progress = min(1.0, (step - cfg.warmup_steps) / decay_steps)
        if cfg.schedule == "cosine":
            mult = 0.5 * (1 + math.cos(math.pi * progress))
        elif cfg.schedule == "linear":
            mult = 1 - progress
        else:
            raise ValueError(f"Unknown schedule {cfg.schedule!r}")
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * mult

    return f


def param_groups(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    decay, no_decay, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.dim() < 2 or getattr(p, "_no_weight_decay", False):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


class Trainer:
    """Train an :class:`~llm_blocks.nn.LLM` on a :class:`TokenDataset`.

    >>> trainer = Trainer(model, train_data, val_data, TrainConfig(max_steps=200))
    >>> history = trainer.fit()
    """

    def __init__(self, model: LLM, train_data: TokenDataset, val_data: Optional[TokenDataset] = None,
                 config: Optional[TrainConfig] = None, log_fn: Optional[Callable[[Dict[str, Any]], None]] = print):
        self.cfg = config or TrainConfig()
        self.device = resolve_device(self.cfg.device)
        self.model = model.to(self.device)
        self.model.gradient_checkpointing_enable(self.cfg.gradient_checkpointing)
        self.train_data, self.val_data = train_data, val_data
        if train_data.seq_len != self.cfg.seq_len:
            raise ValueError(f"train_data.seq_len={train_data.seq_len} != TrainConfig.seq_len={self.cfg.seq_len}")
        fused = self.device.type == "cuda"
        self.optimizer = torch.optim.AdamW(param_groups(model, self.cfg.weight_decay), lr=self.cfg.lr,
                                           betas=self.cfg.betas, eps=self.cfg.eps, fused=fused or None)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda(self.cfg))
        self.scaler = torch.amp.GradScaler(self.device.type) if self.cfg.precision == "fp16" else None
        if self.scaler is not None and self.device.type != "cuda":
            raise ValueError("fp16 training with loss scaling is only supported on CUDA; use bf16")
        self.data_gen = torch.Generator().manual_seed(self.cfg.seed)
        self.step = 0
        self.history: List[Dict[str, Any]] = []
        self.log_fn = log_fn
        self._forward = torch.compile(self.model) if self.cfg.compile else self.model
        set_seed(self.cfg.seed)

    # -- core -----------------------------------------------------------------------------
    def train_step(self) -> Dict[str, Any]:
        self.model.train()
        cfg = self.cfg
        total = torch.zeros((), device=self.device)
        lm = torch.zeros((), device=self.device)
        aux = torch.zeros((), device=self.device)
        for _ in range(cfg.grad_accum_steps):
            batch = self.train_data.get_batch(cfg.batch_size, self.data_gen, self.device)
            with autocast_context(self.device, cfg.precision):
                out = self._forward(batch, labels=batch)
            loss = out.loss / cfg.grad_accum_steps
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            total += loss.detach()
            lm += out.lm_loss.detach() / cfg.grad_accum_steps
            if out.aux_loss is not None:
                aux += out.aux_loss.detach() / cfg.grad_accum_steps
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        grad_norm = (torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                     if cfg.grad_clip else None)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return {"loss": total, "lm_loss": lm, "aux_loss": aux, "grad_norm": grad_norm}

    @torch.no_grad()
    def evaluate(self, data: Optional[TokenDataset] = None, num_batches: Optional[int] = None) -> float:
        data = data or self.val_data
        if data is None:
            raise ValueError("No validation data")
        self.model.eval()
        losses = []
        for batch in data.fixed_batches(num_batches or self.cfg.eval_batches, self.cfg.batch_size, self.device):
            with autocast_context(self.device, self.cfg.precision):
                losses.append(self.model(batch, labels=batch).lm_loss.float())
        self.model.train()
        return torch.stack(losses).mean().item()

    def fit(self) -> List[Dict[str, Any]]:
        cfg = self.cfg
        tokens_per_step = cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps
        t0 = time.perf_counter()
        last_t, last_step = t0, self.step
        while self.step < cfg.max_steps:
            stats = self.train_step()
            record: Optional[Dict[str, Any]] = None
            if self.step % cfg.log_every == 0 or self.step == cfg.max_steps:
                _sync(self.device)
                now = time.perf_counter()
                record = {
                    "step": self.step,
                    "loss": stats["loss"].item(),
                    "lm_loss": stats["lm_loss"].item(),
                    "aux_loss": stats["aux_loss"].item(),
                    "grad_norm": float(stats["grad_norm"]) if stats["grad_norm"] is not None else None,
                    "lr": self.scheduler.get_last_lr()[0],
                    "tokens_per_sec": tokens_per_step * (self.step - last_step) / max(now - last_t, 1e-9),
                    "elapsed_sec": now - t0,
                }
                last_t, last_step = now, self.step
            if self.val_data is not None and (self.step % cfg.eval_every == 0 or self.step == cfg.max_steps):
                record = record or {"step": self.step}
                record["val_loss"] = self.evaluate()
            if record is not None:
                self.history.append(record)
                if self.log_fn is not None:
                    self.log_fn(_fmt(record))
            if cfg.checkpoint_dir and cfg.checkpoint_every and self.step % cfg.checkpoint_every == 0:
                self.save_checkpoint(Path(cfg.checkpoint_dir) / f"step_{self.step:07d}.pt")
        if cfg.checkpoint_dir:
            self.save_checkpoint(Path(cfg.checkpoint_dir) / "last.pt")
        return self.history

    # -- checkpoints ----------------------------------------------------------------------
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model_config": self.model.config.to_dict(),
            "train_config": self.cfg.to_dict(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "data_gen": self.data_gen.get_state(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "history": self.history,
        }
        if self.scaler is not None:
            ckpt["scaler"] = self.scaler.state_dict()
        tmp = path.with_suffix(".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if LLMConfig.from_dict(ckpt["model_config"]) != self.model.config:
            raise ValueError("Checkpoint model config does not match this trainer's model")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step = ckpt["step"]
        self.data_gen.set_state(ckpt["data_gen"])
        torch.set_rng_state(ckpt["torch_rng"])
        random.setstate(ckpt["python_rng"])
        self.history = list(ckpt.get("history", []))
        if self.scaler is not None and "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])

    @staticmethod
    def load_model(path: Union[str, Path], map_location: Union[str, torch.device] = "cpu") -> LLM:
        """Rebuild a model from a training checkpoint."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = LLM(config=LLMConfig.from_dict(ckpt["model_config"]))
        model.load_state_dict(ckpt["model"])
        return model


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _fmt(record: Dict[str, Any]) -> str:
    parts = []
    for k, v in record.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        elif v is not None:
            parts.append(f"{k}={v}")
    return " ".join(parts)
