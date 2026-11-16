import math

import pytest
import torch
import torch.nn.functional as F

from llm_blocks import nn
from llm_blocks.nn import BuildContext
from llm_blocks.nn.position import RotaryEmbedding, alibi_slopes
from llm_blocks.nn.residual import sinkhorn_knopp

from conftest import tiny_model


# ----------------------------------------------------------------------------- norms
def test_rmsnorm_matches_definition():
    m = nn.RMSNorm(eps=1e-6).build(BuildContext(dim=16))
    with torch.no_grad():
        m.weight.uniform_()
    x = torch.randn(3, 16)
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * m.weight
    torch.testing.assert_close(m(x), ref)
    assert m(x.bfloat16()).dtype == torch.bfloat16


def test_layernorm_matches_torch():
    m = nn.LayerNorm().build(BuildContext(dim=16))
    ref = torch.nn.LayerNorm(16)
    x = torch.randn(3, 16)
    torch.testing.assert_close(m(x), ref(x))


def test_qk_norm_normalizes_per_head():
    mod = nn.GQA(heads=4, kv_heads=2, qk_norm=True).build(BuildContext(dim=32))
    assert mod.q_norm.dim == 8 and mod.k_norm.dim == 8
    mod2 = nn.GQA(heads=4, kv_heads=2, qk_norm=nn.LayerNorm()).build(BuildContext(dim=32))
    assert type(mod2.q_norm).__name__ == "LayerNormModule"


# ----------------------------------------------------------------------------- positions
def test_rope_is_relative():
    rope = RotaryEmbedding(16)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    def score(i, j):
        qi, _ = rope(q, q, offset=i)
        _, kj = rope(k, k, offset=j)
        return (qi * kj).sum()
    torch.testing.assert_close(score(5, 2), score(105, 102), atol=1e-4, rtol=1e-4)
    rq, _ = rope(q, q, offset=7)
    torch.testing.assert_close(rq.norm(), q.norm())


def test_partial_rope_leaves_tail_untouched():
    rope = nn.RoPE(rotary_fraction=0.5).build_rotary(16, 64)
    q = torch.randn(1, 1, 3, 16)
    rq, _ = rope(q, q, offset=3)
    torch.testing.assert_close(rq[..., 8:], q[..., 8:])


def test_alibi_slopes_reference_values():
    torch.testing.assert_close(alibi_slopes(8), torch.tensor([2 ** -(i + 1) for i in range(8)], dtype=torch.float32))
    s6 = alibi_slopes(6)  # non power of two: 4 from pow2(4) + 2 interleaved from pow2(8)
    torch.testing.assert_close(s6, torch.tensor([0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125]))


def test_learned_positions_and_padding():
    m = tiny_model(nn.GQA(heads=4, kv_heads=2, position=None), position=nn.LearnedPositions())
    m.eval()
    assert m.pos_embed is not None
    x = torch.randint(0, 97, (1, 9))
    xp = torch.cat([torch.zeros(1, 3, dtype=torch.long), x], 1)
    mask = torch.cat([torch.zeros(1, 3), torch.ones(1, 9)], 1)
    with torch.no_grad():
        torch.testing.assert_close(m(xp, attention_mask=mask).logits[:, 3:], m(x).logits, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="exceeds"):
        m(torch.zeros(1, 200, dtype=torch.long))


# ----------------------------------------------------------------------------- residuals
def test_gated_residual_identity_at_init_and_zero_init():
    x = torch.randn(2, 3, 8)
    f = lambda h: h * 2
    torch.testing.assert_close(nn.GatedResidual(init=1.0).build(BuildContext(dim=8))(x, f), x + 2 * x)
    torch.testing.assert_close(nn.GatedResidual(init=0.0).build(BuildContext(dim=8))(x, f), x)
    g = nn.GatedResidual(gate="input", init=0.25).build(BuildContext(dim=8))
    torch.testing.assert_close(g(x, f), x + 0.25 * 2 * x)


def test_hyperconnections_n1_equals_standard_residual():
    std = tiny_model(nn.GQA(heads=4, kv_heads=2))
    hc = tiny_model(nn.GQA(heads=4, kv_heads=2), residual=nn.HyperConnections(n=1))
    # copy shared weights
    missing = hc.load_state_dict(std.state_dict(), strict=False)
    assert not missing.unexpected_keys
    x = torch.randint(0, 97, (2, 8))
    with torch.no_grad():
        torch.testing.assert_close(hc(x).logits, std(x).logits, atol=1e-5, rtol=1e-5)


def test_hyperconnections_n4_init_behaviour():
    """With static init (A_r = I, B = 1, replicated streams), every stream equals the standard
    residual stream, and the final sum is 4x it; after the final RMSNorm (scale invariant up to
    eps, so eps is made negligible here) logits are identical."""
    fn = nn.RMSNorm(eps=1e-12)
    std = tiny_model(nn.GQA(heads=4, kv_heads=2), depth=1, final_norm=fn)
    hc = tiny_model(nn.GQA(heads=4, kv_heads=2), depth=1, residual=nn.HyperConnections(n=4), final_norm=fn)
    hc.load_state_dict(std.state_dict(), strict=False)
    x = torch.randint(0, 97, (2, 8))
    with torch.no_grad():
        torch.testing.assert_close(hc(x).logits, std(x).logits, atol=1e-4, rtol=1e-4)


def test_sinkhorn_doubly_stochastic():
    M = sinkhorn_knopp(torch.randn(5, 4, 4), iters=50)
    torch.testing.assert_close(M.sum(-1), torch.ones(5, 4), atol=1e-4, rtol=0)
    torch.testing.assert_close(M.sum(-2), torch.ones(5, 4), atol=1e-5, rtol=0)  # last step normalises columns
    assert (M >= 0).all()


def test_mhc_coefficients_respect_constraints():
    mod = nn.MHC(n=4).build(BuildContext(dim=8))
    with torch.no_grad():
        mod.phi.normal_(std=0.05)
        mod.alpha.fill_(1.0)
    x = torch.randn(2, 5, 4, 8)
    pre, post, res = mod.coefficients(x)
    assert ((pre > 0) & (pre < 1)).all() and ((post > 0) & (post < 2)).all()
    # 20 Sinkhorn iterations: columns exact, rows approximately normalised
    torch.testing.assert_close(res.sum(-2), torch.ones(2, 5, 4), atol=1e-5, rtol=0)
    torch.testing.assert_close(res.sum(-1), torch.ones(2, 5, 4), atol=2e-2, rtol=0)
    # forward matches the written formula
    f = lambda h: torch.tanh(h)
    out = mod(x, f)
    h_in = (pre.unsqueeze(-1) * x).sum(-2)
    ref = torch.einsum("btij,btjd->btid", res, x) + post.unsqueeze(-1) * f(h_in).unsqueeze(-2)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_attnres_uniform_average_at_init():
    mod = nn.AttnRes().build(BuildContext(dim=8))
    vals = [torch.randn(2, 3, 8) for _ in range(3)]
    seen = {}
    def f(h):
        seen["h"] = h
        return h * 0
    out = mod(vals, f)
    torch.testing.assert_close(seen["h"], torch.stack(vals).mean(0), atol=1e-6, rtol=1e-6)
    assert len(out) == 4


@pytest.mark.parametrize("res", [nn.Residual(), nn.GatedResidual(), nn.GatedResidual(gate="input", init=0.5),
                                 nn.HyperConnections(n=2), nn.HyperConnections(n=3, dynamic=False), nn.MHC(n=2), nn.AttnRes()])
def test_residuals_train_and_decode(res):
    m = tiny_model(nn.GQA(heads=4, kv_heads=2), residual=res, depth=3)
    x = torch.randint(0, 97, (2, 12))
    out = m(x, labels=x)
    out.loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    m.eval()
    with torch.no_grad():
        full = m(x).logits
        st = m.init_state(2, 16)
        inc = torch.cat([m(x[:, :5], state=st).logits] + [m(x[:, t:t + 1], state=st).logits for t in range(5, 12)], 1)
    torch.testing.assert_close(inc, full, atol=1e-4, rtol=1e-4)


def test_post_norm_block():
    m = tiny_model(nn.MHA(heads=4), norm_position="post", final_norm=None)
    x = torch.randint(0, 97, (2, 6))
    m(x, labels=x).loss.backward()
