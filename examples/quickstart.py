"""Quickstart: build a model, inspect it, swap components, generate."""

import torch

from llm_blocks import nn

model = nn.LLM(
    vocab_size=32000,
    dim=768,
    layers=[
        nn.Block(
            mixer=nn.GQA(heads=12, kv_heads=4, position=nn.RoPE()),
            ffn=nn.SwiGLU(hidden_dim=2048),
            norm=nn.RMSNorm(),
        )
        for _ in range(12)
    ],
)
print(model.summary())
print(model.num_parameters())
print(model.architecture())

# The same architecture as a config, then single-component swaps
base = nn.LLMConfig(vocab_size=32000, dim=768, depth=12,
                    mixer=nn.GQA(heads=12, kv_heads=4), ffn=nn.SwiGLU(hidden_dim=2048))
experiments = {
    "mla": base.replace(mixer=nn.MLA(heads=12)),
    "mamba_every_4th": base.replace_layers(nn.every(4, start=3), mixer=nn.Mamba2()),
    "moe": base.replace(ffn=nn.MoE(expert=nn.SwiGLU(hidden_dim=1024), num_experts=16, top_k=2, shared_experts=1)),
    "gated_residual": base.replace(residual=nn.GatedResidual()),
}
for name, cfg in experiments.items():
    m = nn.LLM(config=cfg)
    print(f"{name:>16}: total={m.num_parameters():,}  active={m.num_active_parameters():,}  "
          f"fwd GFLOPs/token@2048={m.flops_per_token(2048) / 1e9:.2f}  config={cfg.fingerprint()}")

print(experiments["mamba_every_4th"].to_json()[:400], "...")

small = nn.LLM(config=base.replace(dim=128, depth=2, vocab_size=1000, mixer=nn.GQA(heads=4, kv_heads=2),
                                   ffn=nn.SwiGLU(hidden_dim=256)))
tokens = small.generate(torch.randint(0, 1000, (1, 8)), max_new_tokens=16, temperature=0.8, top_p=0.9)
print(tokens.shape)
model.visualize("architecture.svg")
