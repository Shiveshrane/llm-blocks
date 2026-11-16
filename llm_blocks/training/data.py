"""Minimal, deterministic data utilities for causal LM experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor

__all__ = ["ByteTokenizer", "TokenDataset", "load_text_corpus"]


class ByteTokenizer:
    """UTF-8 byte tokenizer (vocab 256 + ``eos`` = 256). No dependencies, fully reversible."""

    vocab_size = 257
    eos_token_id = 256

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        ids = list(text.encode("utf-8"))
        return ids + [self.eos_token_id] if add_eos else ids

    def decode(self, ids: Iterable[int]) -> str:
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")


class TokenDataset:
    """A 1-D token stream sampled as random contiguous windows.

    ``get_batch`` draws window offsets from an explicit ``torch.Generator`` so data order is
    reproducible and checkpointable. ``fixed_batches`` returns evenly spaced windows for
    deterministic evaluation.
    """

    def __init__(self, tokens: Union[Tensor, List[int]], seq_len: int):
        self.tokens = torch.as_tensor(tokens, dtype=torch.long).flatten()
        if self.tokens.numel() <= seq_len:
            raise ValueError(f"Dataset has {self.tokens.numel()} tokens, need more than seq_len={seq_len}")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.tokens.numel()

    def get_batch(self, batch_size: int, generator: torch.Generator, device=None) -> Tensor:
        hi = self.tokens.numel() - self.seq_len
        starts = torch.randint(0, hi, (batch_size,), generator=generator)
        batch = torch.stack([self.tokens[s : s + self.seq_len] for s in starts.tolist()])
        return batch.to(device) if device is not None else batch

    def fixed_batches(self, num_batches: int, batch_size: int, device=None) -> List[Tensor]:
        hi = self.tokens.numel() - self.seq_len
        n = num_batches * batch_size
        starts = torch.linspace(0, hi - 1, n).long().tolist()
        out = []
        for b in range(num_batches):
            rows = [self.tokens[s : s + self.seq_len] for s in starts[b * batch_size : (b + 1) * batch_size]]
            t = torch.stack(rows)
            out.append(t.to(device) if device is not None else t)
        return out

    def split(self, val_fraction: float = 0.1) -> Tuple["TokenDataset", "TokenDataset"]:
        n = int(self.tokens.numel() * (1 - val_fraction))
        return TokenDataset(self.tokens[:n], self.seq_len), TokenDataset(self.tokens[n:], self.seq_len)


def load_text_corpus(paths: Iterable[Union[str, Path]], max_bytes: Optional[int] = None) -> str:
    """Concatenate text files (sorted for determinism) up to ``max_bytes``."""
    chunks, total = [], 0
    for p in sorted(Path(x) for x in paths):
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        chunks.append(text)
        total += len(text)
        if max_bytes is not None and total >= max_bytes:
            break
    corpus = "\n".join(chunks)
    return corpus[:max_bytes] if max_bytes is not None else corpus
