"""Adding a new sequence mixer without touching library code.

A mixer is two small classes: a ``SequenceMixer`` module (the computation + its own decoding
state) and a ``MixerSpec`` (a frozen dataclass that builds it). Subclassing ``MixerSpec``
registers the spec, so configs that use it serialize to / from JSON automatically.

Here: a causal long-convolution mixer ("H3/Hyena-lite"): y = W_o( conv_causal(W_i x) * gate ).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn as tnn

from llm_blocks import nn
from llm_blocks.nn import BuildContext, MixerSpec, MixerState, SequenceMixer


class ConvState(MixerState):
    def __init__(self, window: Tensor):
        self.window = window  # last (kernel-1) projected inputs
        self.offset = 0


class CausalConvMixer(SequenceMixer):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.k = kernel_size
        self.in_proj = tnn.Linear(dim, 2 * dim, bias=False)
        self.filter = tnn.Parameter(torch.randn(dim, kernel_size) / kernel_size)
        self.out_proj = tnn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, *, state: Optional[ConvState] = None, padding_mask: Optional[Tensor] = None) -> Tensor:
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        if padding_mask is not None:
            u = u * padding_mask[..., None].to(u.dtype)
        prev = state.window if state is not None else u.new_zeros(x.shape[0], self.k - 1, x.shape[-1])
        seq = torch.cat([prev.to(u.dtype), u], dim=1)  # (B, k-1+T, D)
        y = F.conv1d(seq.transpose(1, 2), self.filter.flip(-1).unsqueeze(1).to(u.dtype), groups=u.shape[-1])
        if state is not None:
            state.window = seq[:, -(self.k - 1):]
            state.offset += x.shape[1]
        return self.out_proj(y.transpose(1, 2) * F.silu(gate))

    def init_state(self, batch_size, max_seq_len, *, device=None, dtype=None) -> ConvState:
        p = self.in_proj.weight
        return ConvState(torch.zeros(batch_size, self.k - 1, p.shape[1], device=device or p.device, dtype=dtype or p.dtype))


@dataclass(frozen=True, kw_only=True)
class CausalConv(MixerSpec):
    kernel_size: int = 64

    def build(self, ctx: BuildContext) -> CausalConvMixer:
        return CausalConvMixer(ctx.dim, self.kernel_size)


if __name__ == "__main__":
    cfg = nn.LLMConfig(vocab_size=100, dim=64, depth=4, mixer=nn.GQA(heads=4, kv_heads=2))
    cfg = cfg.replace_layers([1, 2], mixer=CausalConv(kernel_size=16))
    model = nn.LLM(config=cfg)
    print(model.architecture())

    # config round-trips through JSON, including the user-defined spec
    assert nn.LLMConfig.from_json(cfg.to_json()) == cfg

    # works with training, incremental decoding and generation like built-in mixers
    x = torch.randint(0, 100, (2, 20))
    model(x, labels=x).loss.backward()
    model.eval()
    with torch.no_grad():
        full = model(x).logits
        st = model.init_state(2, 32)
        inc = torch.cat([model(x[:, :9], state=st).logits] + [model(x[:, t:t + 1], state=st).logits for t in range(9, 20)], 1)
    print("incremental == full:", torch.allclose(full, inc, atol=1e-5))
    print(model.generate(x[:, :4], max_new_tokens=8, do_sample=False).shape)
