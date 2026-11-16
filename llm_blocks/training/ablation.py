"""Run several architectures under identical training conditions and compare them.

"Identical conditions" means: the same ``TrainConfig`` (optimizer, schedule, batch size,
steps, precision), the same data stream (the sampler is seeded from ``TrainConfig.seed``
and is independent of the model), the same validation windows, and per-variant
deterministic initialization from ``LLMConfig.seed``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import torch

from ..nn.models.config import LLMConfig
from ..nn.models.llm import LLM
from ..utils.summary import format_count
from .data import TokenDataset
from .trainer import TrainConfig, Trainer

__all__ = ["AblationResult", "run_ablation", "results_table"]


@dataclass
class AblationResult:
    name: str
    fingerprint: str
    params: int
    active_params: int
    flops_per_token: float
    final_train_loss: float
    final_val_loss: Optional[float]
    best_val_loss: Optional[float]
    tokens_per_sec: float
    wall_time_sec: float
    history: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


def run_ablation(
    variants: Mapping[str, LLMConfig],
    train_data: TokenDataset,
    val_data: Optional[TokenDataset],
    train_config: TrainConfig,
    out_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> List[AblationResult]:
    """Train each variant from scratch with the same ``train_config`` and data."""
    results = []
    for name, cfg in variants.items():
        if verbose:
            print(f"=== {name} ({cfg.fingerprint()}) ===")
        model = LLM(config=cfg)
        trainer = Trainer(model, train_data, val_data, train_config, log_fn=print if verbose else None)
        history = trainer.fit()
        train_recs = [h for h in history if "loss" in h]
        val = [h["val_loss"] for h in history if "val_loss" in h]
        tps = [h["tokens_per_sec"] for h in train_recs[1:]] or [h["tokens_per_sec"] for h in train_recs]
        res = AblationResult(
            name=name,
            fingerprint=cfg.fingerprint(),
            params=model.num_parameters(),
            active_params=model.num_active_parameters(),
            flops_per_token=model.flops_per_token(train_config.seq_len),
            final_train_loss=train_recs[-1]["lm_loss"],
            final_val_loss=val[-1] if val else None,
            best_val_loss=min(val) if val else None,
            tokens_per_sec=sum(tps) / len(tps),
            wall_time_sec=train_recs[-1]["elapsed_sec"],
            history=history,
            config=cfg.to_dict(),
        )
        results.append(res)
        if out_dir is not None:
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{name}.json").write_text(json.dumps(asdict(res), indent=2))
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if out_dir is not None:
        (Path(out_dir) / "summary.md").write_text(results_table(results))
    return results


def results_table(results: List[AblationResult]) -> str:
    header = "| variant | params | active | GFLOPs/tok (fwd) | train loss | val loss | best val | tok/s |\n"
    header += "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    rows = []
    for r in results:
        fv = f"{r.final_val_loss:.4f}" if r.final_val_loss is not None else "-"
        bv = f"{r.best_val_loss:.4f}" if r.best_val_loss is not None else "-"
        rows.append(
            f"| {r.name} | {format_count(r.params)} | {format_count(r.active_params)} | "
            f"{r.flops_per_token / 1e9:.4f} | {r.final_train_loss:.4f} | {fv} | {bv} | {r.tokens_per_sec:,.0f} |"
        )
    return header + "\n".join(rows) + "\n"
