import pytest
import torch

from llm_blocks import nn
from llm_blocks.inference import sample_next_token

from conftest import DEVICES, tiny_mixers, tiny_model

MIXERS = tiny_mixers()


def hybrid():
    return nn.LLM(vocab_size=97, dim=64, max_seq_len=128, layers=[
        nn.Block(mixer=nn.GQA(heads=4, kv_heads=2), ffn=nn.SwiGLU(hidden_dim=96)),
        nn.Block(mixer=nn.Mamba2(d_state=16, headdim=16, chunk_size=8), ffn=nn.SwiGLU(hidden_dim=96)),
        nn.Block(mixer=nn.GatedDeltaNet(heads=2, head_dim=16, chunk_size=8),
                 ffn=nn.MoE(expert=nn.SwiGLU(hidden_dim=32), num_experts=4, top_k=2)),
        nn.Block(mixer=nn.MLA(heads=4), ffn=nn.GeGLU(hidden_dim=96), residual=nn.GatedResidual()),
        nn.Block(mixer=nn.Mamba2(d_state=16, headdim=16, chunk_size=8), ffn=None),
    ])


def test_readme_example_builds():
    model = nn.LLM(
        vocab_size=32000, dim=768,
        layers=[nn.Block(mixer=nn.GQA(heads=12, kv_heads=4, position=nn.RoPE()),
                         ffn=nn.SwiGLU(hidden_dim=2048), norm=nn.RMSNorm()) for _ in range(12)],
    )
    # embed 24.6M + head 24.6M + 12 x (attn 1.57M + SwiGLU 4.72M)
    assert model.num_parameters() == 2 * 32000 * 768 + 12 * (768 * 20 * 64 + 768 * 768 + 3 * 768 * 2048 + 2 * 768) + 768
    assert "layers[0-11] × 12" in model.architecture()


@pytest.mark.parametrize("name", list(MIXERS))
def test_model_forward_backward_every_mixer(name):
    m = tiny_model(MIXERS[name])
    x = torch.randint(0, 97, (2, 19))
    out = m(x, labels=x)
    assert out.logits.shape == (2, 19, 97)
    assert abs(out.lm_loss.item() - torch.log(torch.tensor(97.0)).item()) < 0.5  # ~uniform at init
    out.loss.backward()
    assert all(p.grad is not None for p in m.parameters())


def test_hybrid_incremental_decoding_matches_full():
    m = hybrid().eval()
    x = torch.randint(0, 97, (3, 30))
    with torch.no_grad():
        full = m(x).logits
        st = m.init_state(3, 40)
        parts = [m(x[:, :11], state=st).logits] + [m(x[:, t:t + 1], state=st).logits for t in range(11, 30)]
    torch.testing.assert_close(torch.cat(parts, 1), full, atol=5e-5, rtol=5e-5)
    assert st.offset == 30
    types = {type(s).__name__ for s in st.layers}
    assert types == {"KVCache", "Mamba2State", "GatedDeltaNetState", "MLACache"}


def test_generate_cached_equals_uncached_greedy():
    m = hybrid()
    x = torch.randint(0, 97, (2, 7))
    a = m.generate(x, max_new_tokens=12, do_sample=False)
    b = m.generate(x, max_new_tokens=12, do_sample=False, use_cache=False)
    assert torch.equal(a, b) and a.shape == (2, 19)


def test_generate_left_padded_batch_matches_individual():
    m = hybrid()
    p1, p2 = torch.randint(0, 97, (1, 9)), torch.randint(0, 97, (1, 4))
    batch = torch.cat([p1, torch.cat([torch.zeros(1, 5, dtype=torch.long), p2], 1)])
    mask = torch.ones(2, 9, dtype=torch.long)
    mask[1, :5] = 0
    out = m.generate(batch, max_new_tokens=8, do_sample=False, attention_mask=mask)
    o1 = m.generate(p1, max_new_tokens=8, do_sample=False)
    o2 = m.generate(p2, max_new_tokens=8, do_sample=False)
    assert torch.equal(out[0, 9:], o1[0, 9:])
    assert torch.equal(out[1, 9:], o2[0, 4:])


def test_sampling_controls():
    logits = torch.tensor([[0.0, 1.0, 5.0, 2.0]])
    g = torch.Generator().manual_seed(0)
    assert sample_next_token(logits, temperature=0).item() == 2
    picks = {sample_next_token(logits, top_k=2, generator=g).item() for _ in range(200)}
    assert picks <= {2, 3}
    picks = {sample_next_token(logits, top_p=0.5, generator=g).item() for _ in range(50)}
    assert picks == {2}
    m = hybrid()
    x = torch.randint(0, 97, (1, 3))
    a = m.generate(x, 10, temperature=0.8, top_p=0.9, generator=torch.Generator().manual_seed(1))
    b = m.generate(x, 10, temperature=0.8, top_p=0.9, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


def test_generate_eos_stops_and_pads():
    m = hybrid()
    x = torch.randint(0, 97, (2, 4))
    greedy = m.generate(x, max_new_tokens=6, do_sample=False)
    eos = int(greedy[0, 5])  # second generated token of row 0
    out = m.generate(x, max_new_tokens=6, do_sample=False, eos_token_id=eos)
    first = (out[0, 4:] == eos).nonzero()[0].item()
    assert (out[0, 4 + first:] == eos).all()


def test_gradient_checkpointing_same_gradients():
    m1, m2 = hybrid(), hybrid()
    m2.gradient_checkpointing_enable()
    x = torch.randint(0, 97, (2, 16))
    o1, o2 = m1(x, labels=x), m2(x, labels=x)
    torch.testing.assert_close(o1.loss, o2.loss)
    o1.loss.backward()
    o2.loss.backward()
    for (n, p1), p2 in zip(m1.named_parameters(), m2.parameters()):
        torch.testing.assert_close(p1.grad, p2.grad, atol=1e-6, rtol=1e-5, msg=n)


def test_ignore_index_and_right_padding_training():
    m = tiny_model(MIXERS["GQA"])
    x = torch.randint(0, 97, (2, 10))
    labels = x.clone()
    labels[1, 6:] = -100
    mask = torch.ones(2, 10)
    mask[1, 6:] = 0
    out = m(x, labels=labels, attention_mask=mask)
    ref0 = m(x[:1], labels=x[:1]).lm_loss
    ref1 = m(x[1:, :6], labels=x[1:, :6]).lm_loss
    torch.testing.assert_close(out.lm_loss, (ref0 * 9 + ref1 * 5) / 14, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_autocast_bf16_training_step(device):
    m = hybrid().to(device)
    x = torch.randint(0, 97, (2, 16), device=device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        out = m(x, labels=x)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    with torch.no_grad():
        ref = m(x, labels=x).lm_loss
    assert abs(out.lm_loss.item() - ref.item()) < 0.05


def test_bf16_weights_inference():
    m = hybrid().to(torch.bfloat16).eval()
    x = torch.randint(0, 97, (1, 8))
    out = m.generate(x, max_new_tokens=5, do_sample=False)
    assert out.shape == (1, 13)


def test_save_load_pretrained(tmp_path):
    m = hybrid()
    m.save_pretrained(tmp_path / "ckpt")
    m2 = nn.LLM.from_pretrained(tmp_path / "ckpt")
    assert m2.config == m.config
    x = torch.randint(0, 97, (1, 9))
    with torch.no_grad():
        torch.testing.assert_close(m2(x).logits, m.eval()(x).logits)


def test_tied_embeddings_share_storage():
    m = tiny_model(MIXERS["MHA"], tie_embeddings=True)
    assert m.lm_head.weight is m.embed.weight


def test_swap_one_component_keeps_everything_else():
    base = nn.LLMConfig(vocab_size=97, dim=64, depth=4, mixer=nn.GQA(heads=4, kv_heads=2), ffn=nn.SwiGLU(hidden_dim=96))
    variants = {
        "mla": base.replace(mixer=nn.MLA(heads=4)),
        "moe": base.replace(ffn=nn.MoE(expert=nn.SwiGLU(hidden_dim=48), num_experts=4, top_k=2)),
        "hybrid": base.replace_layers(nn.every(4, start=3), mixer=nn.Mamba2(d_state=16, headdim=16)),
        "gated_res": base.replace(residual=nn.GatedResidual()),
        "mhc": base.replace(residual=nn.MHC(n=2)),
    }
    x = torch.randint(0, 97, (2, 12))
    for name, cfg in variants.items():
        m = nn.LLM(config=cfg)
        out = m(x, labels=x)
        out.loss.backward()
        assert torch.isfinite(out.loss), name
