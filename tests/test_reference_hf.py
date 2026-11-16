"""Numerical comparison against HuggingFace ``transformers`` implementations.

These are independent, widely used implementations; matching them with mapped weights checks
that our components follow the published architectures (not just that they are
self-consistent). Skipped if ``transformers`` is not installed.
"""

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")

from llm_blocks import nn  # noqa: E402
from llm_blocks.kernels import gated_delta_rule_chunked, gated_delta_rule_recurrent  # noqa: E402
from llm_blocks.nn import BuildContext  # noqa: E402

pytestmark = pytest.mark.reference

# Both sides run in float64, but the reference implementations deliberately compute some
# pieces (RMSNorm, RoPE tables, gated norm) in float32, which bounds agreement at ~1e-7.
# Any real formula difference shows up at 1e-2 or larger.
TOL = 1e-6


def causal_mask(T, dtype=torch.float64):
    return torch.full((T, T), torch.finfo(dtype).min, dtype=dtype).triu(1)[None, None]


def test_llama_full_model_matches():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=101, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6, max_position_embeddings=64,
                      rope_parameters={"rope_type": "default", "rope_theta": 10000.0}, tie_word_embeddings=False,
                      attn_implementation="eager")
    ref = LlamaForCausalLM(cfg).double().eval()
    ours = nn.LLM(vocab_size=101, dim=64, max_seq_len=64, layers=[
        nn.Block(mixer=nn.GQA(heads=4, kv_heads=2, position=nn.RoPE(theta=10000.0)),
                 ffn=nn.SwiGLU(hidden_dim=96), norm=nn.RMSNorm(eps=1e-6))] * 3,
        final_norm=nn.RMSNorm(eps=1e-6)).double().eval()
    sd = {"embed.weight": ref.model.embed_tokens.weight, "lm_head.weight": ref.lm_head.weight,
          "final_norm.weight": ref.model.norm.weight}
    for i, L in enumerate(ref.model.layers):
        a, mlp = L.self_attn, L.mlp
        sd[f"layers.{i}.mixer.qkv_proj.weight"] = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
        sd[f"layers.{i}.mixer.o_proj.weight"] = a.o_proj.weight
        sd[f"layers.{i}.ffn.gate_up.weight"] = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight])
        sd[f"layers.{i}.ffn.down.weight"] = mlp.down_proj.weight
        sd[f"layers.{i}.norm1.weight"] = L.input_layernorm.weight
        sd[f"layers.{i}.norm2.weight"] = L.post_attention_layernorm.weight
    ours.load_state_dict(sd, strict=True)
    x = torch.randint(0, 101, (2, 23))
    with torch.no_grad():
        torch.testing.assert_close(ours(x).logits, ref(x).logits, atol=TOL, rtol=TOL)


@pytest.mark.parametrize("q_lora_rank", [None, 24])
def test_deepseek_v3_mla_matches(q_lora_rank):
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention, DeepseekV3RotaryEmbedding

    cfg = DeepseekV3Config(hidden_size=64, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=q_lora_rank,
                           kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=12,
                           rope_interleave=False, attention_bias=False, max_position_embeddings=64,
                           rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
                           attn_implementation="eager")
    ref = DeepseekV3Attention(cfg, layer_idx=0).double().eval()
    ours = nn.MLA(heads=4, q_lora_rank=q_lora_rank, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8,
                  v_head_dim=12, position=nn.RoPE(theta=10000.0)).build(BuildContext(dim=64)).double().eval()
    sd = {"kv_a_proj.weight": ref.kv_a_proj_with_mqa.weight, "kv_a_norm.weight": ref.kv_a_layernorm.weight,
          "kv_b_proj.weight": ref.kv_b_proj.weight, "o_proj.weight": ref.o_proj.weight}
    if q_lora_rank:
        sd.update({"q_a_proj.weight": ref.q_a_proj.weight, "q_a_norm.weight": ref.q_a_layernorm.weight,
                   "q_b_proj.weight": ref.q_b_proj.weight})
    else:
        sd["q_proj.weight"] = ref.q_proj.weight
    ours.load_state_dict(sd, strict=True)
    with torch.no_grad():  # randomize norm weights so they are actually checked
        for w in (ours.kv_a_norm.weight, ref.kv_a_layernorm.weight):
            w.copy_(torch.linspace(0.5, 1.5, w.numel(), dtype=w.dtype))
    ref.kv_a_layernorm.variance_epsilon = ours.kv_a_norm.eps
    if q_lora_rank:
        ref.q_a_layernorm.variance_epsilon = ours.q_a_norm.eps
    T = 17
    x = torch.randn(2, T, 64, dtype=torch.float64)
    rope = DeepseekV3RotaryEmbedding(cfg).double()
    cos, sin = rope(x, torch.arange(T)[None].expand(2, T))
    with torch.no_grad():
        expected = ref(x, position_embeddings=(cos, sin), attention_mask=causal_mask(T))[0]
        torch.testing.assert_close(ours(x), expected, atol=TOL, rtol=TOL)
        st = ours.init_state(2, T)  # absorbed decoding path must match too
        inc = torch.cat([ours(x[:, :9], state=st)] + [ours(x[:, t:t + 1], state=st) for t in range(9, T)], 1)
        torch.testing.assert_close(inc, expected, atol=TOL, rtol=TOL)


@pytest.mark.parametrize("ngroups", [1, 2])
def test_mamba2_mixer_matches(ngroups):
    from transformers import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer

    cfg = Mamba2Config(hidden_size=64, state_size=16, expand=2, head_dim=16, num_heads=8, n_groups=ngroups,
                       conv_kernel=4, chunk_size=8, use_bias=False, use_conv_bias=True, layer_norm_epsilon=1e-5)
    ref = Mamba2Mixer(cfg, layer_idx=0).double().eval()
    ours = nn.Mamba2(d_state=16, expand=2, headdim=16, ngroups=ngroups, chunk_size=8).build(BuildContext(dim=64)).double().eval()
    with torch.no_grad():
        ref.norm.weight.copy_(torch.linspace(0.5, 1.5, 128, dtype=torch.float64))
    sd = {"in_proj.weight": ref.in_proj.weight, "conv1d.weight": ref.conv1d.weight, "conv1d.bias": ref.conv1d.bias,
          "dt_bias": ref.dt_bias, "A_log": ref.A_log, "D": ref.D, "norm.weight": ref.norm.weight,
          "out_proj.weight": ref.out_proj.weight}
    ours.load_state_dict(sd, strict=True)
    x = torch.randn(2, 29, 64, dtype=torch.float64)
    with torch.no_grad():
        expected = ref(x)  # eval mode, no CUDA kernels -> pure PyTorch path
        torch.testing.assert_close(ours(x), expected, atol=TOL, rtol=TOL)


@pytest.mark.parametrize("L,chunk", [(37, 8), (64, 16), (5, 64)])
def test_gated_delta_rule_matches_qwen3_next(L, chunk):
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )

    b, h, dk, dv = 2, 3, 16, 8
    q = F.normalize(torch.randn(b, L, h, dk, dtype=torch.float64), dim=-1)
    k = F.normalize(torch.randn(b, L, h, dk, dtype=torch.float64), dim=-1)
    v = torch.randn(b, L, h, dv, dtype=torch.float64)
    g = -F.softplus(torch.randn(b, L, h, dtype=torch.float64))
    beta = torch.sigmoid(torch.randn(b, L, h, dtype=torch.float64))
    init = torch.randn(b, h, dk, dv, dtype=torch.float64) * 0.1
    o_ref, s_ref = torch_chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk, initial_state=init,
                                                output_final_state=True)
    o_rec, s_rec = torch_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=init, output_final_state=True)
    # our kernels expect q pre-scaled by dk^-0.5 (the reference scales internally)
    o1, s1 = gated_delta_rule_chunked(q * dk ** -0.5, k, v, g, beta, chunk, init)
    o2, s2 = gated_delta_rule_recurrent(q * dk ** -0.5, k, v, g, beta, init)
    # reference keeps float32 internally
    for o, s in ((o1, s1), (o2, s2)):
        torch.testing.assert_close(o, o_rec.to(o.dtype), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(s, s_rec.to(s.dtype), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(o, o_ref.to(o.dtype), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(s, s_ref.to(s.dtype), atol=1e-5, rtol=1e-5)
