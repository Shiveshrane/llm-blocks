import json

import pytest
import torch

from llm_blocks import nn
from llm_blocks.training import ByteTokenizer, TokenDataset, TrainConfig, Trainer, results_table, run_ablation
from llm_blocks.training.trainer import lr_lambda


def corpus():
    tok = ByteTokenizer()
    text = "the quick brown fox jumps over the lazy dog. " * 200
    return TokenDataset(tok.encode(text), seq_len=32)


def model(**kw):
    cfg = nn.LLMConfig(vocab_size=257, dim=48, depth=2, mixer=nn.GQA(heads=4, kv_heads=2), ffn=nn.SwiGLU(hidden_dim=96), **kw)
    return nn.LLM(config=cfg)


def tcfg(**kw):
    base = dict(max_steps=30, batch_size=8, seq_len=32, lr=3e-3, warmup_steps=5, precision="fp32", device="cpu",
                eval_every=10, eval_batches=2, log_every=5)
    base.update(kw)
    return TrainConfig(**base)


def test_tokenizer_roundtrip():
    tok = ByteTokenizer()
    s = "héllo ✓"
    assert tok.decode(tok.encode(s, add_eos=True)) == s


def test_loss_decreases():
    train, val = corpus().split(0.2)
    tr = Trainer(model(), train, val, tcfg(max_steps=60), log_fn=None)
    hist = tr.fit()
    vals = [h["val_loss"] for h in hist if "val_loss" in h]
    assert vals[-1] < vals[0] - 1.0, vals


def test_checkpoint_resume_is_exact(tmp_path):
    data = corpus()
    a = Trainer(model(), data, None, tcfg(max_steps=20), log_fn=None)
    a.fit()
    b = Trainer(model(), data, None, tcfg(max_steps=20), log_fn=None)  # same schedule, stopped halfway
    for _ in range(10):
        b.train_step()
    b.save_checkpoint(tmp_path / "mid.pt")
    c = Trainer(model(), data, None, tcfg(max_steps=20), log_fn=None)
    c.load_checkpoint(tmp_path / "mid.pt")
    assert c.step == 10
    c.fit()
    for (n, p1), p2 in zip(a.model.named_parameters(), c.model.parameters()):
        torch.testing.assert_close(p1, p2, msg=n)
    loaded = Trainer.load_model(tmp_path / "mid.pt")
    torch.testing.assert_close(loaded.embed.weight, b.model.embed.weight)


def test_grad_accumulation_equivalent_to_large_batch():
    data = corpus()
    m1, m2 = model(), model()
    t1 = Trainer(m1, data, None, tcfg(batch_size=8, grad_accum_steps=1, max_steps=1, grad_clip=None), log_fn=None)
    t2 = Trainer(m2, data, None, tcfg(batch_size=4, grad_accum_steps=2, max_steps=1, grad_clip=None), log_fn=None)
    # same 8 sequences: draw them once and feed in two halves vs one batch
    batch = data.get_batch(8, torch.Generator().manual_seed(0))
    l1 = m1(batch, labels=batch).loss
    l1.backward()
    for half in (batch[:4], batch[4:]):
        (m2(half, labels=half).loss / 2).backward()
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(p1.grad, p2.grad, atol=1e-6, rtol=1e-4)
    # and the trainer path runs
    t2.train_step()
    assert t2.step == 1


def test_lr_schedule_shape():
    f = lr_lambda(TrainConfig(max_steps=100, warmup_steps=10, min_lr_ratio=0.1))
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0) and f(100) == pytest.approx(0.1)
    assert f(55) == pytest.approx(0.55)


def test_no_weight_decay_on_norms_and_ssm_scalars():
    m = nn.LLM(vocab_size=50, dim=32, layers=[nn.Block(mixer=nn.Mamba2(d_state=8, headdim=16))])
    tr = Trainer(m, TokenDataset(list(range(50)) * 10, 16), None, tcfg(seq_len=16), log_fn=None)
    nodecay = {id(p) for p in tr.optimizer.param_groups[1]["params"]}
    assert id(m.layers[0].mixer.A_log) in nodecay and id(m.layers[0].norm1.weight) in nodecay
    assert id(m.layers[0].mixer.in_proj.weight) not in nodecay


def test_bf16_training_runs():
    data = corpus()
    tr = Trainer(model(), data, None, tcfg(max_steps=5, precision="bf16"), log_fn=None)
    hist = tr.fit()
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in hist if "loss" in h)


def test_ablation_runner(tmp_path):
    train, val = corpus().split(0.2)
    base = model().config
    res = run_ablation({"gqa": base, "mla": base.replace(mixer=nn.MLA(heads=4))}, train, val,
                       tcfg(max_steps=10), out_dir=tmp_path, verbose=False)
    assert [r.name for r in res] == ["gqa", "mla"]
    table = results_table(res)
    assert "| gqa |" in table and "| mla |" in table
    assert json.loads((tmp_path / "mla.json").read_text())["fingerprint"] == base.replace(mixer=nn.MLA(heads=4)).fingerprint()
