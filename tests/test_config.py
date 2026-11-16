import json

import pytest
import torch

from llm_blocks import nn
from llm_blocks.nn import SpecError, spec_from_dict


def base():
    return nn.LLMConfig(vocab_size=101, dim=64, depth=8, mixer=nn.GQA(heads=4, kv_heads=2),
                        ffn=nn.SwiGLU(hidden_dim=128), max_seq_len=64)


def test_spec_roundtrip_all_components():
    specs = [
        nn.GQA(heads=8, kv_heads=2, qk_norm=True, position=nn.RoPE(theta=5e5)),
        nn.MLA(heads=4, q_lora_rank=32), nn.Mamba2(d_state=16), nn.GatedDeltaNet(head_dim=32),
        nn.MoE(expert=nn.GeGLU(hidden_dim=64), num_experts=4, top_k=1, capacity_factor=1.5, shared_experts=1),
        nn.Block(mixer=nn.MQA(heads=4, position=nn.ALiBi()), ffn=None, norm=nn.LayerNorm(), residual=nn.MHC(n=2)),
        nn.QKNorm(norm=nn.LayerNorm()), nn.GatedResidual(gate="input", init=0.5), nn.AttnRes(),
    ]
    for s in specs:
        again = spec_from_dict(json.loads(s.to_json()))
        assert again == s, s


def test_config_roundtrip_with_overrides(tmp_path):
    cfg = base().replace_layers(nn.every(4, start=3), mixer=nn.Mamba2(d_state=16, headdim=16))
    path = tmp_path / "c.json"
    cfg.save(path)
    again = nn.LLMConfig.load(path)
    assert again == cfg
    assert again.fingerprint() == cfg.fingerprint()
    layers = again.resolve_layers()
    assert [type(b.mixer).__name__ for b in layers] == ["GQA"] * 3 + ["Mamba2"] + ["GQA"] * 3 + ["Mamba2"]


def test_replace_changes_base_but_keeps_overrides():
    cfg = base().replace_layers([0], ffn=nn.MoE(expert=nn.SwiGLU(hidden_dim=32), num_experts=4))
    cfg2 = cfg.replace(mixer=nn.MLA(heads=4))
    layers = cfg2.resolve_layers()
    assert all(isinstance(b.mixer, nn.MLA) for b in layers)
    assert isinstance(layers[0].ffn, nn.MoE) and isinstance(layers[1].ffn, nn.SwiGLU)


def test_replace_layers_on_explicit_layers_and_selectors():
    cfg = nn.LLMConfig(vocab_size=11, dim=32, layers=[nn.Block(mixer=nn.MHA(heads=2))] * 4)
    c2 = cfg.replace_layers(slice(1, 3), residual=nn.GatedResidual())
    assert [type(b.residual).__name__ for b in c2.resolve_layers()] == ["Residual", "GatedResidual", "GatedResidual", "Residual"]
    c3 = cfg.replace_layers(lambda i: i % 2 == 0, norm_position="post")
    assert [b.norm_position for b in c3.resolve_layers()] == ["post", "pre", "post", "pre"]
    with pytest.raises(SpecError):
        cfg.replace_layers(9, mixer=nn.MHA(heads=2))
    with pytest.raises(SpecError):
        cfg.replace_layers(0, heads=3)


def test_spec_replace_rejects_unknown_fields():
    with pytest.raises(SpecError, match="no field"):
        nn.GQA(heads=4, kv_heads=2).replace(kvheads=1)


@pytest.mark.parametrize("bad, match", [
    (lambda: nn.LLM(vocab_size=10, dim=30, layers=[nn.Block(mixer=nn.MHA(heads=4))]), "not divisible"),
    (lambda: nn.LLM(vocab_size=10, dim=32, layers=[nn.Block(mixer=nn.GQA(heads=4, kv_heads=3))]), "divisible by kv_heads"),
    (lambda: nn.LLM(vocab_size=10, dim=32, layers=[nn.Block(mixer=nn.Mamba2(position=nn.RoPE()))]), "does not support"),
    (lambda: nn.LLM(vocab_size=10, dim=32, layers=[nn.Block(mixer=nn.GatedDeltaNet(position=nn.ALiBi()))]), "does not support"),
    (lambda: nn.LLM(vocab_size=10, dim=32, layers=[nn.Block(mixer=nn.MLA(heads=4, position=nn.ALiBi()))]), "does not support"),
    (lambda: nn.LLM(vocab_size=10, dim=32, layers=[nn.Block(mixer=nn.GQA(heads=4, kv_heads=2, position=nn.LearnedPositions()))]), "model-level"),
    (lambda: nn.LLM(vocab_size=10, dim=32, layers=[nn.Block(mixer=nn.MHA(heads=4))], position=nn.RoPE()), "belongs on the mixer"),
    (lambda: nn.LLM(vocab_size=10, dim=32, layers=[nn.Block(mixer=nn.MHA(heads=4), residual=nn.MHC(n=2)),
                                                    nn.Block(mixer=nn.MHA(heads=4))]), "residual stream"),
    (lambda: nn.LLM(vocab_size=10, dim=32, layers=[nn.Block(mixer=nn.MHA(heads=4), residual=nn.AttnRes(), norm_position="post")]), "Post-norm"),
    (lambda: nn.LLM(vocab_size=10, dim=32, layers=[nn.Block(mixer=nn.SwiGLU())]), "mixer spec"),
    (lambda: nn.LLM(vocab_size=10, dim=32, layers=[nn.Block(mixer=nn.MHA(heads=4), ffn=nn.MoE(expert=nn.SwiGLU(), num_experts=2, top_k=3))]), "top_k"),
    (lambda: nn.LLMConfig(vocab_size=10, dim=32, depth=2), "needs either"),
])
def test_meaningful_errors(bad, match):
    with pytest.raises(SpecError, match=match):
        bad()


def test_deterministic_init():
    a, b = nn.LLM(config=base()), nn.LLM(config=base())
    for (n1, p1), (n2, p2) in zip(a.state_dict().items(), b.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2), n1
    c = nn.LLM(config=base().replace(seed=1))
    assert not torch.equal(a.embed.weight, c.embed.weight)
    # building must not perturb the global RNG
    torch.manual_seed(5)
    x = torch.rand(3)
    torch.manual_seed(5)
    nn.LLM(config=base())
    assert torch.equal(torch.rand(3), x)


def test_same_spec_reused_builds_independent_layers():
    blk = nn.Block(mixer=nn.MHA(heads=4), ffn=nn.SwiGLU(hidden_dim=64))
    m = nn.LLM(vocab_size=10, dim=32, layers=[blk] * 3)
    assert m.layers[0].mixer.qkv_proj.weight.data_ptr() != m.layers[1].mixer.qkv_proj.weight.data_ptr()
    assert not torch.equal(m.layers[0].mixer.qkv_proj.weight, m.layers[1].mixer.qkv_proj.weight)


def test_parameter_counting_and_summary():
    m = nn.LLM(config=base())
    expected = sum(p.numel() for p in m.parameters())
    assert m.num_parameters() == expected
    assert m.num_parameters(exclude_embeddings=True) == expected - 101 * 64
    tied = nn.LLM(config=base().replace(tie_embeddings=True))
    assert tied.num_parameters() == expected - 101 * 64
    bd = m.parameter_breakdown()
    assert bd["total"] == expected
    assert sum(v for k, v in bd.items() if k != "total") == expected
    s = m.summary()
    assert "GQA" in s and "Total" in s and "FLOPs" in s
    arch = m.architecture()
    assert "layers[0-7] × 8" in arch
    svg = m.visualize()
    assert svg.startswith("<svg") and "GQA" in svg
    assert "flowchart" in m.visualize(fmt="mermaid")


def test_moe_active_parameters():
    cfg = base().replace(ffn=nn.MoE(expert=nn.SwiGLU(hidden_dim=32), num_experts=8, top_k=2))
    m = nn.LLM(config=cfg)
    per_expert = 3 * 64 * 32
    total_experts = 8 * per_expert * 8  # 8 layers
    inactive = 8 * 6 * per_expert
    assert m.num_parameters() - m.num_active_parameters() == inactive
    assert total_experts > inactive


def test_flops_estimate_matches_hand_count():
    cfg = nn.LLMConfig(vocab_size=100, dim=64, depth=2, mixer=nn.MHA(heads=4), ffn=nn.FFN(hidden_dim=256))
    m = nn.LLM(config=cfg)
    T = 32
    attn = 2 * (64 * 3 * 64 + 64 * 64) + 4 * 64 * (T + 1) / 2
    mlp = 2 * (64 * 256 * 2)
    head = 2 * 64 * 100
    assert m.flops_per_token(T) == pytest.approx(2 * (attn + mlp) + head)
