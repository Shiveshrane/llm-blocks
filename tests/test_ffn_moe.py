import pytest
import torch
import torch.nn.functional as F

from llm_blocks import nn
from llm_blocks.nn import BuildContext, ForwardContext

CTX = BuildContext(dim=32)


@pytest.mark.parametrize("spec, hidden", [
    (nn.FFN(), 128), (nn.GeluFFN(hidden_dim=40), 40), (nn.ReluFFN(), 128),
    (nn.SwiGLU(), 128), (nn.GeGLU(hidden_dim=48), 48), (nn.GLU(activation="relu"), 128),
])
def test_dense_ffn_shapes_and_math(spec, hidden):
    m = spec.build(CTX)
    x = torch.randn(2, 5, 32, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    y.sum().backward()
    if isinstance(spec, nn.GLU):
        gate, up = (x @ m.gate_up.weight.T).chunk(2, -1)
        assert gate.shape[-1] == hidden
        act = {"silu": F.silu, "gelu": lambda t: F.gelu(t, approximate="tanh"), "relu": F.relu}[spec.activation]
        ref = (act(gate) * up) @ m.down.weight.T
    else:
        h = x @ m.up.weight.T
        assert h.shape[-1] == hidden
        act = {"gelu": lambda t: F.gelu(t, approximate="tanh"), "relu": F.relu}[spec.activation]
        ref = act(h) @ m.down.weight.T
    torch.testing.assert_close(y, ref)


def test_glu_default_hidden_is_param_matched():
    assert nn.SwiGLU()._hidden(768) == 2048  # 8/3 * 768 rounded to 64


def naive_moe(moe, x):
    """Reference: loop over tokens and their chosen experts."""
    flat = x.reshape(-1, x.shape[-1])
    w, idx, _, _ = moe.route(flat)
    out = torch.zeros_like(flat)
    for t in range(flat.shape[0]):
        for j in range(moe.top_k):
            out[t] += w[t, j] * moe.experts.forward_expert(int(idx[t, j]), flat[t:t + 1])[0]
    if moe.shared is not None:
        out = out + moe.shared(flat)
    return out.reshape(x.shape)


@pytest.mark.parametrize("kw", [
    dict(router="softmax", normalize_topk=True),
    dict(router="softmax", normalize_topk=False, top_k=1),
    dict(router="sigmoid"),
    dict(shared_experts=2),
    dict(expert=nn.FFN(hidden_dim=24)),
])
def test_moe_dropless_matches_naive(kw):
    kw = {"expert": nn.SwiGLU(hidden_dim=24), "num_experts": 5, "top_k": 2, **kw}
    moe = nn.MoE(**kw).build(CTX)
    x = torch.randn(3, 7, 32)
    torch.testing.assert_close(moe(x), naive_moe(moe, x), atol=1e-5, rtol=1e-5)


def test_moe_capacity_with_ample_capacity_matches_dropless():
    torch.manual_seed(0)
    a = nn.MoE(expert=nn.SwiGLU(hidden_dim=24), num_experts=4, top_k=2, capacity_factor=4.0).build(CTX).train()
    x = torch.randn(2, 9, 32, requires_grad=True)
    ctx = ForwardContext()
    y = a(x, ctx)
    torch.testing.assert_close(y, naive_moe(a, x), atol=1e-5, rtol=1e-5)
    assert float(ctx.metrics["moe_drop_fraction"][0]) == 0.0
    y.sum().backward()
    assert a.experts.w_up.grad is not None and a.router.weight.grad is not None


def test_moe_capacity_drops_overflow_and_eval_is_dropless():
    moe = nn.MoE(expert=nn.SwiGLU(hidden_dim=16), num_experts=4, top_k=1, capacity_factor=0.25).build(CTX).train()
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0] = 1.0  # bias routing toward expert 0
    x = torch.rand(1, 32, 32) + 0.1
    ctx = ForwardContext()
    moe(x, ctx)
    cap = 2  # ceil(0.25 * 32 / 4)
    assert float(ctx.metrics["moe_drop_fraction"][0]) == pytest.approx(1 - cap / 32)
    moe.eval()
    torch.testing.assert_close(moe(x), naive_moe(moe, x), atol=1e-5, rtol=1e-5)


def test_moe_aux_loss_values():
    moe = nn.MoE(expert=nn.SwiGLU(hidden_dim=8), num_experts=4, top_k=1, aux_loss_coef=1.0, z_loss_coef=0.0).build(CTX).train()
    with torch.no_grad():
        moe.router.weight.zero_()  # uniform probabilities; topk picks expert 0 for every token
    ctx = ForwardContext()
    moe(torch.randn(1, 8, 32), ctx)
    # f = [1,0,0,0], P = [1/4]*4 -> E * sum f*P = 4 * 1/4 = 1
    assert ctx.total_aux_loss().item() == pytest.approx(1.0)


def test_moe_bias_update_balances_load():
    torch.manual_seed(0)
    moe = nn.MoE(expert=nn.SwiGLU(hidden_dim=8), num_experts=4, top_k=1, router="sigmoid",
                 bias_update_rate=0.05, aux_loss_coef=0.0).build(CTX).train()
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0, 0] = 5.0
    x = torch.randn(1, 64, 32).abs()
    def load():
        _, idx, _, _ = moe.route(x.reshape(-1, 32))
        return torch.bincount(idx.flatten(), minlength=4).float()
    before = load().max()
    for _ in range(100):
        moe(x)
    assert load().max() < before
    assert moe.expert_bias.abs().sum() > 0


def test_moe_inside_model_reports_aux_loss():
    cfg = nn.LLMConfig(vocab_size=50, dim=32, depth=2, mixer=nn.GQA(heads=4, kv_heads=2),
                       ffn=nn.MoE(expert=nn.SwiGLU(hidden_dim=16), num_experts=4, top_k=2, z_loss_coef=1e-3))
    m = nn.LLM(config=cfg).train()
    x = torch.randint(0, 50, (2, 10))
    out = m(x, labels=x)
    assert out.aux_loss is not None and out.aux_loss > 0
    torch.testing.assert_close(out.loss, out.lm_loss + out.aux_loss)
    m.eval()
    assert m(x, labels=x).aux_loss is None


def test_nested_moe_rejected():
    with pytest.raises(nn.SpecError):
        nn.MoE(expert=nn.MoE(expert=nn.SwiGLU())).build(CTX)
