"""Mixture-of-Experts feed-forward layer.

Routing
    ``router="softmax"``: probabilities ``p = softmax(W_r x)``, pick top-k; with
    ``normalize_topk=True`` the selected weights are renormalised (Mixtral / Qwen-MoE),
    otherwise the raw probabilities are used (Switch / GShard).
    ``router="sigmoid"``: affinities ``s = sigmoid(W_r x)``, top-k on ``s + bias`` and
    normalised selected affinities as weights (DeepSeek-V3). ``bias`` is the optional
    auxiliary-loss-free load-balancing bias, updated outside of autograd by
    ``bias_update_rate * sign(mean_load - load)`` after each training forward.

Auxiliary losses (added to the model loss through :class:`ForwardContext`)
    * load-balancing loss (Switch Transformer, Fedus et al. 2021):
      ``E * sum_i f_i * P_i`` with ``f_i`` the fraction of routed assignments that went to
      expert ``i`` and ``P_i`` its mean routing probability; equals 1 at perfect balance.
    * router z-loss (ST-MoE, Zoph et al. 2022): ``mean(logsumexp(logits)^2)``.

Dispatch
    ``capacity_factor=None`` (default) is *dropless*: assignments are sorted by expert and
    each expert processes exactly its tokens (one host sync to read the per-expert counts).
    ``capacity_factor=c`` uses fixed-capacity GShard-style dispatch: each expert gets
    ``ceil(c * tokens * top_k / E)`` slots, overflow assignments are dropped (first choices
    have priority), and all experts run in a single batched ``bmm`` with static shapes —
    no host synchronisation, friendly to ``torch.compile`` and CUDA graphs. Capacity is
    only applied in training mode; ``eval()`` always dispatches droplessly.

Shared experts (DeepSeekMoE): ``shared_experts=n`` adds an always-active dense expert of
``n`` times the routed expert's hidden size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...configs.spec import BuildContext, SpecError
from ..base import FeedForward, FFNSpec, ForwardContext
from ..ffn import GLU, FFN

__all__ = ["MoELayer", "MoE"]


class MoELayer(FeedForward):
    def __init__(
        self,
        dim: int,
        experts: nn.Module,
        *,
        num_experts: int,
        top_k: int,
        router: str = "softmax",
        normalize_topk: bool = True,
        capacity_factor: Optional[float] = None,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.0,
        bias_update_rate: float = 0.0,
        shared: Optional[FeedForward] = None,
    ):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise SpecError(f"MoE: need 1 <= top_k <= num_experts, got top_k={top_k}, num_experts={num_experts}")
        if router not in ("softmax", "sigmoid"):
            raise SpecError(f"MoE: router must be 'softmax' or 'sigmoid', got {router!r}")
        self.dim, self.num_experts, self.top_k = dim, num_experts, top_k
        self.router_type, self.normalize_topk = router, normalize_topk
        self.capacity_factor = capacity_factor
        self.aux_loss_coef, self.z_loss_coef, self.bias_update_rate = aux_loss_coef, z_loss_coef, bias_update_rate
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = experts
        self.shared = shared
        self.register_buffer("expert_bias", torch.zeros(num_experts), persistent=True)

    # -- routing -------------------------------------------------------------------------
    def route(self, x: Tensor):
        """Return ``(weights (N,k), indices (N,k), probs (N,E) fp32, logits (N,E) fp32)``."""
        with torch.autocast(device_type=x.device.type, enabled=False):  # router in fp32
            logits = F.linear(x.float(), self.router.weight.float())
        if self.router_type == "softmax":
            probs = logits.softmax(dim=-1)
            sel = probs + self.expert_bias if self.bias_update_rate > 0 else probs
            _, idx = sel.topk(self.top_k, dim=-1)
            w = probs.gather(-1, idx)
            if self.normalize_topk and self.top_k > 1:
                w = w / w.sum(-1, keepdim=True)
        else:
            scores = torch.sigmoid(logits)
            _, idx = (scores + self.expert_bias).topk(self.top_k, dim=-1)
            w = scores.gather(-1, idx)
            if self.normalize_topk:
                w = w / (w.sum(-1, keepdim=True) + 1e-20)
            probs = scores / (scores.sum(-1, keepdim=True) + 1e-20)
        return w, idx, probs, logits

    # -- forward -------------------------------------------------------------------------
    def forward(self, x: Tensor, ctx: Optional[ForwardContext] = None) -> Tensor:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        N, E, k = x.shape[0], self.num_experts, self.top_k
        w, idx, probs, logits = self.route(x)

        # assignments per expert (static-shape scatter instead of bincount: no torch.compile graph break)
        counts = torch.zeros(E, dtype=torch.long, device=x.device).scatter_add_(
            0, idx.reshape(-1), torch.ones_like(idx.reshape(-1)))
        if self.training and ctx is not None:
            if self.aux_loss_coef > 0:
                f = counts.float() / (N * k)
                aux = E * (f * probs.mean(0)).sum()
                ctx.add_aux_loss(self.aux_loss_coef * aux)
            if self.z_loss_coef > 0:
                ctx.add_aux_loss(self.z_loss_coef * torch.logsumexp(logits, -1).square().mean())
        if self.training and self.bias_update_rate > 0:
            with torch.no_grad():
                load = counts.float()
                self.expert_bias.add_(self.bias_update_rate * torch.sign(load.mean() - load))

        if self.capacity_factor is None or not self.training:
            # Capacity limits are a training-time device; inference is dropless so that the
            # output for a token never depends on which other tokens share the batch.
            y = self._dropless(x, w, idx, counts)
        else:
            y = self._capacity(x, w, idx, ctx)
        if self.shared is not None:
            y = y + self.shared(x)
        return y.reshape(shape)

    def _dropless(self, x: Tensor, w: Tensor, idx: Tensor, counts: Tensor) -> Tensor:
        N, k = idx.shape
        flat_e = idx.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        token_of = order // k
        xs = x.index_select(0, token_of)
        outs = []
        start = 0
        for e, c in enumerate(counts.tolist()):  # single host sync
            if c:
                outs.append(self.experts.forward_expert(e, xs[start:start + c]))
                start += c
        out = torch.cat(outs, dim=0)
        out = out * w.reshape(-1)[order, None].to(out.dtype)
        return out.new_zeros(x.shape).index_add_(0, token_of, out)

    def _capacity(self, x: Tensor, w: Tensor, idx: Tensor, ctx: Optional[ForwardContext]) -> Tensor:
        N, k = idx.shape
        E, D = self.num_experts, self.dim
        cap = max(1, math.ceil(self.capacity_factor * N * k / E))  # type: ignore[operator]
        # priority: all first choices, then all second choices, ...
        flat_e = idx.t().reshape(-1)  # (k*N,)
        token_of = torch.arange(N, device=x.device).repeat(k)
        weight = w.t().reshape(-1)
        onehot = F.one_hot(flat_e, E)
        pos = (onehot.cumsum(0) * onehot).sum(-1) - 1  # position of each assignment in its expert queue
        keep = pos < cap
        slot = torch.where(keep, flat_e * cap + pos, torch.full_like(pos, E * cap))  # dropped -> dummy slot
        buf = x.new_zeros(E * cap + 1, D)
        buf.index_copy_(0, slot, x.index_select(0, token_of))
        out = self.experts.forward_batched(buf[:-1].view(E, cap, D)).reshape(E * cap, D)
        out = torch.cat([out, out.new_zeros(1, D)], dim=0)
        y_assign = out.index_select(0, slot) * (weight * keep).to(out.dtype)[:, None]
        if ctx is not None:
            ctx.log("moe_drop_fraction", (1.0 - keep.float().mean()).detach())
        return y_assign.new_zeros(x.shape).index_add_(0, token_of, y_assign)

    # -- accounting ----------------------------------------------------------------------
    def active_parameters(self) -> int:
        expert_params = sum(p.numel() for p in self.experts.parameters())
        active = expert_params * self.top_k // self.num_experts
        active += self.router.weight.numel()
        if self.shared is not None:
            active += sum(p.numel() for p in self.shared.parameters())
        return active

    def flops_per_token(self) -> float:
        routed = 2.0 * sum(p.numel() for p in self.experts.parameters()) * self.top_k / self.num_experts
        shared = self.shared.flops_per_token() if self.shared is not None else 0.0
        return routed + shared + 2.0 * self.router.weight.numel()

    def extra_repr(self) -> str:
        return (f"num_experts={self.num_experts}, top_k={self.top_k}, router={self.router_type}, "
                f"capacity_factor={self.capacity_factor}")


@dataclass(frozen=True, kw_only=True)
class MoE(FFNSpec):
    """Mixture-of-Experts spec.

    Args:
        expert: spec of a single expert (``nn.SwiGLU``, ``nn.GeGLU``, ``nn.FFN``...).
        num_experts: routed experts. top_k: experts per token.
        shared_experts: number of always-on shared experts (merged into one dense FFN).
        router: ``"softmax"`` or ``"sigmoid"``. normalize_topk: renormalise top-k weights.
        capacity_factor: ``None`` = dropless; a float enables fixed-capacity dispatch.
        aux_loss_coef / z_loss_coef: coefficients of the balance / z losses (0 disables).
        bias_update_rate: > 0 enables auxiliary-loss-free balancing bias (DeepSeek-V3).
    """

    expert: FFNSpec
    num_experts: int = 8
    top_k: int = 2
    shared_experts: int = 0
    router: str = "softmax"
    normalize_topk: bool = True
    capacity_factor: Optional[float] = None
    aux_loss_coef: float = 0.01
    z_loss_coef: float = 0.0
    bias_update_rate: float = 0.0

    def build(self, ctx: BuildContext) -> MoELayer:
        if isinstance(self.expert, MoE):
            raise SpecError("MoE: nested MoE experts are not supported")
        experts = self.expert.build_experts(ctx, self.num_experts)
        shared = None
        if self.shared_experts:
            if not isinstance(self.expert, (GLU, FFN)):
                raise SpecError("MoE: shared experts require a GLU/FFN expert spec")
            hidden = self.expert._hidden(ctx.dim) * self.shared_experts
            shared = self.expert.replace(hidden_dim=hidden).build(ctx)
        return MoELayer(
            ctx.dim, experts, num_experts=self.num_experts, top_k=self.top_k, router=self.router,
            normalize_topk=self.normalize_topk, capacity_factor=self.capacity_factor,
            aux_loss_coef=self.aux_loss_coef, z_loss_coef=self.z_loss_coef,
            bias_update_rate=self.bias_update_rate, shared=shared,
        )
