import pytest
import torch
import torch.nn.functional as F

from llm_blocks.kernels import (
    gated_delta_rule_chunked,
    gated_delta_rule_recurrent,
    ssd_chunked,
    ssd_recurrent,
)


def ssd_inputs(b=2, L=37, h=3, p=4, n=5, dtype=torch.float32):
    X = torch.randn(b, L, h, p, dtype=dtype)
    dt = F.softplus(torch.randn(b, L, h, dtype=dtype))
    A = -torch.rand(h, dtype=dtype) * 2
    return X * dt[..., None], A * dt, torch.randn(b, L, h, n, dtype=dtype), torch.randn(b, L, h, n, dtype=dtype)


@pytest.mark.parametrize("L,chunk", [(37, 8), (16, 16), (5, 8), (64, 4), (1, 4)])
def test_ssd_chunked_matches_recurrence(L, chunk):
    X, A, B, C = ssd_inputs(L=L)
    init = torch.randn(2, 3, 4, 5)
    y1, s1 = ssd_chunked(X, A, B, C, chunk, init)
    y2, s2 = ssd_recurrent(X, A, B, C, init)
    torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s1, s2, atol=1e-4, rtol=1e-4)


def test_ssd_state_carry_equals_full():
    X, A, B, C = ssd_inputs(L=40)
    y, s = ssd_chunked(X, A, B, C, 8)
    ya, sa = ssd_chunked(X[:, :17], A[:, :17], B[:, :17], C[:, :17], 8)
    yb, sb = ssd_chunked(X[:, 17:], A[:, 17:], B[:, 17:], C[:, 17:], 8, sa)
    torch.testing.assert_close(torch.cat([ya, yb], 1), y, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(sb, s, atol=1e-4, rtol=1e-4)


def test_ssd_gradcheck():
    X, A, B, C = ssd_inputs(b=1, L=6, h=2, p=2, n=3, dtype=torch.float64)
    for t in (X, A, B, C):
        t.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda *a: ssd_chunked(*a, 4)[0], (X, A, B, C))


def gdn_inputs(b=2, L=37, h=3, dk=8, dv=6, dtype=torch.float32):
    q = F.normalize(torch.randn(b, L, h, dk, dtype=dtype), dim=-1) * dk ** -0.5
    k = F.normalize(torch.randn(b, L, h, dk, dtype=dtype), dim=-1)
    v = torch.randn(b, L, h, dv, dtype=dtype)
    g = -F.softplus(torch.randn(b, L, h, dtype=dtype))
    beta = torch.sigmoid(torch.randn(b, L, h, dtype=dtype))
    return q, k, v, g, beta


@pytest.mark.parametrize("L,chunk", [(37, 8), (32, 16), (3, 8), (64, 64)])
def test_delta_rule_chunked_matches_recurrence(L, chunk):
    q, k, v, g, beta = gdn_inputs(L=L)
    init = torch.randn(2, 3, 8, 6) * 0.1
    o1, s1 = gated_delta_rule_chunked(q, k, v, g, beta, chunk, init)
    o2, s2 = gated_delta_rule_recurrent(q, k, v, g, beta, init)
    torch.testing.assert_close(o1, o2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s1, s2, atol=1e-4, rtol=1e-4)


def test_delta_rule_definition():
    """Check the recurrence against the matrix form S_t = a_t (I - b_t k k^T) S_{t-1} + b_t k v^T."""
    q, k, v, g, beta = gdn_inputs(b=1, L=5, h=1, dk=4, dv=3, dtype=torch.float64)
    S = torch.zeros(4, 3, dtype=torch.float64)
    outs = []
    for t in range(5):
        kt, vt, a, bt = k[0, t, 0], v[0, t, 0], g[0, t, 0].exp(), beta[0, t, 0]
        S = a * (torch.eye(4, dtype=torch.float64) - bt * torch.outer(kt, kt)) @ S + bt * torch.outer(kt, vt)
        outs.append(S.T @ q[0, t, 0])
    o, _ = gated_delta_rule_recurrent(q, k, v, g, beta)
    torch.testing.assert_close(o[0, :, 0], torch.stack(outs), atol=1e-12, rtol=1e-10)


def test_delta_rule_gradcheck():
    tensors = [t.requires_grad_(True) for t in gdn_inputs(b=1, L=6, h=1, dk=3, dv=2, dtype=torch.float64)]
    assert torch.autograd.gradcheck(lambda *a: gated_delta_rule_chunked(*a, chunk_size=4)[0], tuple(tensors))
