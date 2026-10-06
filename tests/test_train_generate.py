import math

import torch

from mnx.generate import complete, fill_in_middle, main as generate_main
from mnx.train import TrainArgs, cosine_lr, load_checkpoint, train, main as train_main


def _args(tmp_path, sample_dir, **kw) -> TrainArgs:
    base = dict(
        config="tiny",
        data=str(sample_dir),
        out=str(tmp_path / "run"),
        vocab_size=400,
        steps=5,
        batch_size=2,
        seq_len=32,
        eval_interval=5,
        eval_iters=1,
        log_interval=1,
        save_interval=5,
        warmup=1,
        lr=1e-3,
        fim_rate=0.5,
        overrides={"n_layers": 1, "d_model": 32, "n_heads": 2, "n_kv_heads": 1, "d_ff": 64},
    )
    base.update(kw)
    return TrainArgs(**base)


def test_cosine_schedule():
    assert cosine_lr(0, 100, 1.0, 0.1, warmup=10) == 0.1
    assert math.isclose(cosine_lr(9, 100, 1.0, 0.1, warmup=10), 1.0)
    assert math.isclose(cosine_lr(10, 100, 1.0, 0.1, warmup=10), 1.0)
    assert math.isclose(cosine_lr(55, 100, 1.0, 0.1, warmup=10), 0.55, abs_tol=1e-9)
    assert math.isclose(cosine_lr(100, 100, 1.0, 0.1, warmup=10), 0.1)


def test_training_smoke_and_checkpoint_roundtrip(tmp_path, sample_dir):
    args = _args(tmp_path, sample_dir)
    history = train(args)
    assert len(history["step"]) == 5
    assert all(math.isfinite(l) for l in history["train_loss"])
    assert math.isfinite(history["val_loss"][-1])

    run = tmp_path / "run"
    assert (run / "ckpt.pt").exists()
    assert (run / "tokenizer.json").exists()
    assert (run / "config.json").exists()
    log = (run / "log.csv").read_text().strip().splitlines()
    assert log[0].startswith("step,lr,train_loss,val_loss")
    assert len(log) == 6

    model, tok, meta = load_checkpoint(run)
    assert meta["step"] == 5
    assert model.cfg.vocab_size == tok.vocab_size
    assert model.cfg.n_layers == 1

    text = complete(model, tok, "def add(a, b):", max_new_tokens=6, temperature=0.0)
    assert isinstance(text, str)
    middle = fill_in_middle(model, tok, "def f(x):\n    ", "\n    return y\n", max_new_tokens=6, seed=0)
    assert isinstance(middle, str)


def test_training_resume(tmp_path, sample_dir):
    train(_args(tmp_path, sample_dir, steps=3, save_interval=3, eval_interval=0))
    history = train(_args(tmp_path, sample_dir, steps=5, save_interval=5, eval_interval=0, resume=True))
    assert history["step"][0] == 4 and history["step"][-1] == 5
    _, _, meta = load_checkpoint(tmp_path / "run")
    assert meta["step"] == 5


def test_cli_entry_points(tmp_path, sample_dir, capsys):
    out = tmp_path / "cli_run"
    train_main(
        [
            "--config", "tiny", "--data", str(sample_dir), "--out", str(out),
            "--steps", "2", "--batch-size", "2", "--seq-len", "16", "--vocab-size", "350",
            "--eval-interval", "0", "--save-interval", "2", "--log-interval", "1", "--warmup", "1",
            "--set", "n_layers=1", "d_model=32", "n_heads=2", "n_kv_heads=1", "d_ff=64",
        ]
    )
    generate_main(["--ckpt", str(out), "--prompt", "def add(a, b):", "--max-new-tokens", "4", "--temperature", "0"])
    generate_main(["--ckpt", str(out), "--fim", "--prefix", "def f(x):\\n    ", "--suffix", "\\n", "--max-new-tokens", "4", "--seed", "0"])
    captured = capsys.readouterr().out
    assert "MNX 2.0 Max Coder" in captured
    assert "def add(a, b):" in captured
    assert "FIM" in captured


def test_loss_decreases_on_tiny_overfit(tmp_path, sample_dir):
    """A few dozen steps on a tiny model should materially reduce training loss."""
    args = _args(
        tmp_path, sample_dir, steps=100, batch_size=4, lr=5e-3, warmup=5,
        eval_interval=0, save_interval=0, fim_rate=0.0, log_interval=1,
    )
    torch.manual_seed(0)
    history = train(args)
    first = sum(history["train_loss"][:3]) / 3
    last = sum(history["train_loss"][-3:]) / 3
    assert last < first * 0.8, (first, last)
