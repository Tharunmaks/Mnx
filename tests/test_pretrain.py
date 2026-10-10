import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from mnx.budget import budget
from mnx.configs import get_config
from mnx.model import MNXCoder
from mnx.prepare import prepare
from mnx.pretrain import PretrainArgs, ShardedTokens, export, latest_checkpoint, make_rng, pretrain
from mnx.train import load_checkpoint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def shards(tmp_path_factory, tokenizer, sample_dir):
    root = tmp_path_factory.mktemp("shards")
    tok_path = root / "tok.json"
    tokenizer.save(tok_path)
    out = root / "data"
    prepare([str(sample_dir)], str(tok_path), str(out), shard_tokens=2000, val_fraction=0.1, fim_rate=0.0)
    return out


def _args(shards, out, steps, **kw):
    base = dict(
        config="tiny", data=str(shards), out=str(out), steps=steps, micro_batch=2, seq_len=64,
        lr=3e-3, warmup=2, eval_interval=steps, eval_iters=1, log_interval=1, save_interval=steps,
        device="cpu", dtype="fp32",
    )
    base.update(kw)
    return PretrainArgs(**base)


def test_prepare_writes_lossless_shards(shards, tokenizer, sample_dir):
    meta = json.loads((shards / "meta.json").read_text())
    assert meta["vocab_size"] == tokenizer.vocab_size
    assert len(meta["train"]) > 1 and meta["val"]
    stream = np.concatenate([np.fromfile(shards / s["file"], dtype=np.uint16) for s in meta["train"] + meta["val"]])
    assert len(stream) == meta["train_tokens"] + meta["val_tokens"]
    # Every document is framed as <|bos|> ... <|eos|> and decodes back exactly.
    assert (stream == tokenizer.bos_id).sum() == meta["documents"] == (stream == tokenizer.eos_id).sum()
    decoded = tokenizer.decode(stream.tolist(), skip_special=True)
    for f in list(sample_dir.glob("*.py"))[:5]:
        assert f.read_text() in decoded


def test_prepare_parallel_matches_serial(tmp_path, tokenizer, sample_dir):
    tok_path = tmp_path / "tok.json"
    tokenizer.save(tok_path)
    a = prepare([str(sample_dir)], str(tok_path), str(tmp_path / "a"), fim_rate=0.5, workers=1)
    b = prepare([str(sample_dir)], str(tok_path), str(tmp_path / "b"), fim_rate=0.5, workers=2)
    assert a["train_tokens"] == b["train_tokens"] and a["val_tokens"] == b["val_tokens"]
    ta = np.fromfile(tmp_path / "a" / "train_00000.bin", dtype=np.uint16)
    tb = np.fromfile(tmp_path / "b" / "train_00000.bin", dtype=np.uint16)
    assert np.array_equal(ta, tb)


def test_prepare_reads_jsonl(tmp_path, tokenizer):
    tok_path = tmp_path / "tok.json"
    tokenizer.save(tok_path)
    src = tmp_path / "docs.jsonl"
    src.write_text("\n".join(json.dumps({"content": f"def f{i}():\n    return {i}\n"}) for i in range(20)))
    meta = prepare([str(src)], str(tok_path), str(tmp_path / "out"), val_fraction=0.0, fim_rate=0.0)
    assert meta["documents"] == 20 and meta["val_tokens"] == 0


def test_sharded_tokens_windows(shards):
    ds = ShardedTokens(shards, "train", seq_len=32)
    x, y = ds.batch(4, make_rng(0, 0, 0))
    assert x.shape == y.shape == (4, 32) and x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])
    x2, _ = ds.batch(4, make_rng(0, 0, 0))
    assert torch.equal(x, x2)


def test_meta_init_7b_without_memory():
    with torch.device("meta"):
        model = MNXCoder(get_config("max"))
    assert model.num_parameters() == 7_113_805_824
    assert model.lm_head.weight is model.tok_emb.weight


def test_to_empty_keeps_tied_embeddings():
    with torch.device("meta"):
        model = MNXCoder(get_config("tiny"))
    model.to_empty(device="cpu")
    model.init_weights()
    assert model.lm_head.weight is model.tok_emb.weight
    assert model.num_parameters() == MNXCoder(get_config("tiny")).num_parameters()
    assert torch.isfinite(model.tok_emb.weight).all()


def test_gradient_checkpointing_matches(tokenizer):
    cfg = get_config("tiny", vocab_size=tokenizer.vocab_size)
    torch.manual_seed(0)
    a = MNXCoder(cfg)
    b = MNXCoder(cfg)
    b.load_state_dict(a.state_dict())
    b.gradient_checkpointing = True
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    for m in (a, b):
        m.train()
        _, loss, _ = m(x, x)
        loss.backward()
    assert torch.allclose(a.blocks[0].attn.wq.weight.grad, b.blocks[0].attn.wq.weight.grad, atol=1e-6)


def test_pretrain_resume_and_export(shards, tmp_path):
    out = tmp_path / "run"
    h1 = pretrain(_args(shards, out, steps=4, save_interval=2))
    assert h1["step"][-1] == 4 and all(np.isfinite(h1["train_loss"]))
    assert latest_checkpoint(out).name == "step_00000004"
    h2 = pretrain(_args(shards, out, steps=6, save_interval=2))
    assert h2["step"][0] == 5 and h2["step"][-1] == 6
    assert len(list(out.glob("step_*"))) == 2  # keep_checkpoints

    final = tmp_path / "final"
    export(out, final, dtype="fp32")
    model, tok, meta = load_checkpoint(final)
    assert meta["step"] == 6
    assert model.lm_head.weight is model.tok_emb.weight
    ids = model.generate(torch.tensor([[tok.bos_id]]), max_new_tokens=4, temperature=0)
    assert ids.shape == (1, 5)


def test_pretrain_bf16_autocast(shards, tmp_path):
    h = pretrain(_args(shards, tmp_path / "bf16", steps=2, dtype="bf16"))
    assert all(np.isfinite(h["train_loss"]))


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_pretrain_fsdp_two_ranks(shards, tmp_path):
    out = tmp_path / "fsdp"
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "-m", "mnx.pretrain",
        "--config", "tiny", "--data", str(shards), "--out", str(out), "--steps", "3", "--micro-batch", "2",
        "--seq-len", "64", "--eval-interval", "3", "--eval-iters", "1", "--save-interval", "3",
        "--device", "cpu", "--grad-accum", "2",
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "world 2" in res.stdout
    assert latest_checkpoint(out).name == "step_00000003"


def test_budget_scales_with_tokens():
    a = budget("max", 1e11, gpus=8)
    b = budget("max", 2e11, gpus=8)
    assert b["gpu_hours"] == pytest.approx(2 * a["gpu_hours"])
    assert a["params"] == 7_113_805_824
