"""Short causal depthwise convolution with an incremental-decoding state."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise causal conv over ``(batch, seq, channels)``.

    With ``conv_state`` (``(batch, channels, kernel_size - 1)``, the last inputs seen) the
    convolution continues seamlessly across calls; the updated state is returned.
    """

    def __init__(self, channels: int, kernel_size: int, bias: bool = True):
        super().__init__()
        self.channels, self.kernel_size = channels, kernel_size
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None
        bound = 1.0 / kernel_size**0.5  # PyTorch Conv1d default (fan_in = kernel_size)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor, conv_state: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        xt = x.transpose(1, 2)  # (B, C, T)
        K = self.kernel_size
        if conv_state is None:
            inp = F.pad(xt, (K - 1, 0))
        else:
            inp = torch.cat([conv_state.to(xt.dtype), xt], dim=-1)
        new_state = inp[..., -(K - 1):] if K > 1 else None
        if x.shape[1] == 1 and conv_state is not None:
            out = (inp * self.weight.squeeze(1)).sum(-1, keepdim=True)
            if self.bias is not None:
                out = out + self.bias[:, None]
        else:
            out = F.conv1d(inp, self.weight, self.bias, groups=self.channels)
        return out.transpose(1, 2), new_state
