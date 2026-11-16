"""Reference (pure PyTorch) kernels for non-attention mixers.

Attention uses ``torch.nn.functional.scaled_dot_product_attention``, which dispatches to
FlashAttention / memory-efficient kernels on CUDA. Optional fused kernels from
``mamba_ssm`` / ``flash-linear-attention`` are not wired in yet (see docs/STATUS.md).
"""

from .delta_rule import gated_delta_rule_chunked, gated_delta_rule_recurrent, gated_delta_rule_step
from .ssd import ssd_chunked, ssd_recurrent, ssd_step

__all__ = [
    "ssd_chunked", "ssd_recurrent", "ssd_step",
    "gated_delta_rule_chunked", "gated_delta_rule_recurrent", "gated_delta_rule_step",
]
