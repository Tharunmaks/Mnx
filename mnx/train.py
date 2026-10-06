"""Training loop for MNX 2.0 Max Coder.

Example::

    python -m mnx.train --config tiny --data data/sample --out runs/tiny --steps 300

Features: AdamW, cosine learning-rate schedule with linear warmup, gradient
clipping, gradient accumulation, periodic validation loss, CSV + stdout
logging, checkpoint save/resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from mnx.configs import PRESETS, get_config
from mnx.data import CodeDataset, load_corpus, split_documents, tokenize_documents
from mnx.model import MNXCoder, MNXConfig
from mnx.tokenizer import BPETokenizer, iter_source_files

CKPT_NAME = "ckpt.pt"
TOKENIZER_NAME = "tokenizer.json"
CONFIG_NAME = "config.json"
LOG_NAME = "log.csv"


@dataclass
class TrainArgs:
    config: str = "tiny"
    data: str = "data/sample"
    out: str = "runs/tiny"
    tokenizer: Optional[str] = None
    vocab_size: Optional[int] = None
    steps: int = 300
    batch_size: int = 8
    seq_len: Optional[int] = None
    grad_accum: int = 1
    lr: float = 3e-3
    min_lr: Optional[float] = None
    warmup: int = 30
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_iters: int = 8
    log_interval: int = 10
    save_interval: int = 100
    val_fraction: float = 0.1
    fim_rate: float = 0.3
    seed: int = 0
    device: str = "cpu"
    threads: Optional[int] = None
    resume: bool = False
    overrides: Dict[str, object] = field(default_factory=dict)


def cosine_lr(step: int, total: int, lr: float, min_lr: float, warmup: int) -> float:
    """Linear warmup to ``lr`` then cosine decay to ``min_lr``."""
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: MNXCoder, lr: float, weight_decay: float, betas: Tuple[float, float]) -> torch.optim.AdamW:
    """AdamW with weight decay applied only to matrices (not norms/biases)."""
    seen = set()
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


@torch.no_grad()
def estimate_loss(model: MNXCoder, dataset: CodeDataset, batch_size: int, iters: int, gen: torch.Generator, device: str) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = dataset.random_batch(batch_size, gen)
        _, loss, _ = model(x.to(device), y.to(device))
        losses.append(loss.item())
    if was_training:
        model.train()
    return float(sum(losses) / len(losses))


# -------------------------------------------------------------- checkpoints
def save_checkpoint(out_dir: Path, model: MNXCoder, optimizer: torch.optim.Optimizer, step: int, args: TrainArgs, best_val: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / CKPT_NAME
    tmp = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": model.cfg.to_dict(),
            "train_args": asdict(args),
            "best_val": best_val,
        },
        tmp,
    )
    tmp.replace(path)
    (out_dir / CONFIG_NAME).write_text(json.dumps(model.cfg.to_dict(), indent=2))
    return path


def load_checkpoint(ckpt: str | Path, device: str = "cpu") -> Tuple[MNXCoder, BPETokenizer, dict]:
    """Load ``(model, tokenizer, metadata)`` from a run directory or ``ckpt.pt``."""
    ckpt = Path(ckpt)
    run_dir = ckpt if ckpt.is_dir() else ckpt.parent
    ckpt_path = run_dir / CKPT_NAME if ckpt.is_dir() else ckpt
    state = torch.load(ckpt_path, map_location=device)
    cfg = MNXConfig.from_dict(state["config"])
    model = MNXCoder(cfg)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    tok = BPETokenizer.load(run_dir / TOKENIZER_NAME)
    meta = {k: v for k, v in state.items() if k not in ("model", "optimizer")}
    return model, tok, meta


# -------------------------------------------------------------------- train
def prepare_tokenizer(args: TrainArgs, out_dir: Path, target_vocab: int) -> BPETokenizer:
    out_path = out_dir / TOKENIZER_NAME
    if args.tokenizer:
        tok = BPETokenizer.load(args.tokenizer)
        if Path(args.tokenizer).resolve() != out_path.resolve():
            tok.save(out_path)
        return tok
    if args.resume and out_path.exists():
        return BPETokenizer.load(out_path)
    files = iter_source_files(Path(args.data))
    print(f"training tokenizer (vocab_size={target_vocab}) on {len(files)} files ...")
    tok = BPETokenizer.train_from_files(files, vocab_size=target_vocab)
    tok.save(out_path)
    return tok


def train(args: TrainArgs) -> Dict[str, List[float]]:
    """Run training and return a history dict with ``step``, ``train_loss``, ``val_loss``."""
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = args.device
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = dict(args.overrides)
    if args.seq_len:
        overrides["max_seq_len"] = args.seq_len
    cfg = get_config(args.config, **overrides)

    tok = prepare_tokenizer(args, out_dir, args.vocab_size or cfg.vocab_size)
    if tok.vocab_size != cfg.vocab_size:
        # The tokenizer may stop early on tiny corpora; the model always
        # matches the actual tokenizer size.
        cfg = cfg.replace(vocab_size=tok.vocab_size)

    docs = load_corpus(args.data)
    train_docs, val_docs = split_documents(docs, args.val_fraction, seed=args.seed)
    seq_len = cfg.max_seq_len
    train_ds = CodeDataset(tokenize_documents(train_docs, tok), tok, seq_len, fim_rate=args.fim_rate, seed=args.seed)
    val_ds = (
        CodeDataset(tokenize_documents(val_docs, tok), tok, seq_len, fim_rate=0.0, seed=args.seed)
        if val_docs
        else train_ds
    )
    print(
        f"corpus: {len(docs)} docs ({len(train_docs)} train / {len(val_docs)} val), "
        f"{train_ds.num_tokens} train tokens, seq_len={seq_len}, vocab={tok.vocab_size}"
    )

    model = MNXCoder(cfg).to(device)
    print(f"model {cfg.name}: {MNXCoder.format_params(model.num_parameters())} parameters")
    optimizer = build_optimizer(model, args.lr, args.weight_decay, (args.beta1, args.beta2))
    min_lr = args.min_lr if args.min_lr is not None else args.lr / 10

    start_step = 0
    best_val = float("inf")
    ckpt_path = out_dir / CKPT_NAME
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        best_val = float(state.get("best_val", best_val))
        print(f"resumed from {ckpt_path} at step {start_step}")

    log_path = out_dir / LOG_NAME
    new_log = not (args.resume and log_path.exists())
    log_file = open(log_path, "a" if not new_log else "w", newline="")
    writer = csv.writer(log_file)
    if new_log:
        writer.writerow(["step", "lr", "train_loss", "val_loss", "elapsed_s"])

    history: Dict[str, List[float]] = {"step": [], "train_loss": [], "val_loss": []}
    gen = torch.Generator().manual_seed(args.seed + start_step)
    eval_gen = torch.Generator().manual_seed(args.seed + 12345)
    model.train()
    t0 = time.time()
    tokens_per_step = args.batch_size * args.grad_accum * seq_len
    # Re-sample document order / FIM cuts roughly once per pass over the data.
    steps_per_epoch = max(1, train_ds.num_tokens // tokens_per_step)

    for step in range(start_step, args.steps):
        if args.fim_rate > 0 and step > 0 and step % steps_per_epoch == 0:
            train_ds.reshuffle(args.seed + step // steps_per_epoch)

        lr = cosine_lr(step, args.steps, args.lr, min_lr, args.warmup)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            x, y = train_ds.random_batch(args.batch_size, gen)
            _, loss, _ = model(x.to(device), y.to(device))
            (loss / args.grad_accum).backward()
            loss_acc += loss.item() / args.grad_accum
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        is_last = step + 1 == args.steps
        val_loss: Optional[float] = None
        if args.eval_interval > 0 and ((step + 1) % args.eval_interval == 0 or is_last):
            val_loss = estimate_loss(model, val_ds, args.batch_size, args.eval_iters, eval_gen, device)
            best_val = min(best_val, val_loss)

        if (step + 1) % args.log_interval == 0 or is_last or step == start_step or val_loss is not None:
            elapsed = time.time() - t0
            msg = f"step {step + 1:>6}/{args.steps} | lr {lr:.2e} | loss {loss_acc:.4f}"
            if val_loss is not None:
                msg += f" | val {val_loss:.4f}"
            msg += f" | {elapsed:6.1f}s"
            print(msg, flush=True)
            writer.writerow([step + 1, f"{lr:.6e}", f"{loss_acc:.6f}", "" if val_loss is None else f"{val_loss:.6f}", f"{elapsed:.1f}"])
            log_file.flush()
            history["step"].append(step + 1)
            history["train_loss"].append(loss_acc)
            history["val_loss"].append(float("nan") if val_loss is None else val_loss)

        if args.save_interval > 0 and ((step + 1) % args.save_interval == 0 or is_last):
            save_checkpoint(out_dir, model, optimizer, step + 1, args, best_val)

    log_file.close()
    print(f"done. checkpoint: {ckpt_path}  best val loss: {best_val:.4f}")
    return history


# ---------------------------------------------------------------------- CLI
def parse_args(argv: Optional[Sequence[str]] = None) -> TrainArgs:
    d = TrainArgs()
    p = argparse.ArgumentParser(description="Train an MNX 2.0 Max Coder model.")
    p.add_argument("--config", default=d.config, help=f"preset name, one of {sorted(set(PRESETS))}")
    p.add_argument("--data", default=d.data, help="directory of source files or a text file")
    p.add_argument("--out", default=d.out, help="run directory for checkpoints and logs")
    p.add_argument("--tokenizer", default=None, help="existing tokenizer.json (default: train one on --data)")
    p.add_argument("--vocab-size", type=int, default=None, help="tokenizer vocab size (default: from preset)")
    p.add_argument("--steps", type=int, default=d.steps)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--seq-len", type=int, default=None, help="override preset max_seq_len")
    p.add_argument("--grad-accum", type=int, default=d.grad_accum)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--warmup", type=int, default=d.warmup)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--grad-clip", type=float, default=d.grad_clip)
    p.add_argument("--eval-interval", type=int, default=d.eval_interval)
    p.add_argument("--eval-iters", type=int, default=d.eval_iters)
    p.add_argument("--log-interval", type=int, default=d.log_interval)
    p.add_argument("--save-interval", type=int, default=d.save_interval)
    p.add_argument("--val-fraction", type=float, default=d.val_fraction)
    p.add_argument("--fim-rate", type=float, default=d.fim_rate, help="fraction of documents rewritten as FIM")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config fields, e.g. n_layers=2")
    ns = p.parse_args(argv)

    overrides: Dict[str, object] = {}
    for item in ns.set:
        k, v = item.split("=", 1)
        overrides[k] = json.loads(v) if v not in ("true", "false") else (v == "true")

    return TrainArgs(
        config=ns.config, data=ns.data, out=ns.out, tokenizer=ns.tokenizer, vocab_size=ns.vocab_size,
        steps=ns.steps, batch_size=ns.batch_size, seq_len=ns.seq_len, grad_accum=ns.grad_accum,
        lr=ns.lr, min_lr=ns.min_lr, warmup=ns.warmup, weight_decay=ns.weight_decay,
        grad_clip=ns.grad_clip, eval_interval=ns.eval_interval, eval_iters=ns.eval_iters,
        log_interval=ns.log_interval, save_interval=ns.save_interval, val_fraction=ns.val_fraction,
        fim_rate=ns.fim_rate, seed=ns.seed, device=ns.device, threads=ns.threads, resume=ns.resume,
        overrides=overrides,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
