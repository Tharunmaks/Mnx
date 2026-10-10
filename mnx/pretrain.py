"""Large-scale pretraining for MNX 2.0 Max Coder (up to the 7B ``max`` preset).

Single node, 8 GPUs::

    torchrun --nproc_per_node 8 -m mnx.pretrain --config max \\
        --data /data/mnx-shards --out runs/max \\
        --micro-batch 1 --grad-accum 16 --seq-len 4096 --steps 50000 \\
        --lr 3e-4 --warmup 2000 --grad-checkpointing --compile

Multi-node: launch the same command on every node with ``--nnodes``,
``--node_rank`` and ``--rdzv_endpoint`` (see ``scripts/launch_7b.sh``).

What it adds on top of :mod:`mnx.train`:

* **FSDP2** (``torch.distributed.fsdp.fully_shard``): parameters, gradients
  and optimizer state are sharded across all ranks, one unit per transformer
  block, so a 7B model fits on 80 GB GPUs.
* **Meta-device init**: the model is built without allocating memory and each
  rank materialises only its own shard.
* **bf16 mixed precision** (bf16 compute, fp32 master weights and gradient
  reduction).
* **Activation checkpointing** and optional per-block ``torch.compile``.
* **Streaming data** from the ``uint16`` shards written by :mod:`mnx.prepare`
  via ``numpy.memmap`` (the corpus is never loaded into RAM).
* **Sharded, resumable checkpoints** with ``torch.distributed.checkpoint``.
* Throughput logging: tokens/s and model FLOPs utilisation (MFU).

Run ``python -m mnx.pretrain export --run runs/max --out runs/max-final`` to
turn the latest sharded checkpoint into a ``ckpt.pt`` that
:func:`mnx.train.load_checkpoint` and ``python -m mnx.generate`` can load.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

from mnx.configs import PRESETS, get_config
from mnx.model import MNXCoder, MNXConfig
from mnx.prepare import META_NAME
from mnx.train import CKPT_NAME, CONFIG_NAME, TOKENIZER_NAME, build_optimizer, cosine_lr

LATEST_NAME = "latest"
LOG_NAME = "log.csv"


@dataclass
class PretrainArgs:
    config: str = "max"
    data: str = "data/shards"
    out: str = "runs/max"
    steps: int = 50_000
    micro_batch: int = 1
    grad_accum: int = 1
    seq_len: Optional[int] = None
    lr: float = 3e-4
    min_lr: Optional[float] = None
    warmup: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 20
    log_interval: int = 10
    save_interval: int = 1000
    keep_checkpoints: int = 2
    dtype: str = "bf16"
    grad_checkpointing: bool = False
    compile: bool = False
    peak_tflops: float = 989.0
    seed: int = 0
    device: Optional[str] = None
    overrides: Dict[str, object] = field(default_factory=dict)


# ------------------------------------------------------------- distributed
@dataclass
class Dist:
    rank: int
    local_rank: int
    world: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return dist.is_available() and dist.is_initialized()


def setup_distributed(device: Optional[str]) -> Dist:
    """Initialise the process group when launched by ``torchrun``."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
    else:
        dev = torch.device(device)
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if dev.type == "cuda" else "gloo")
    return Dist(rank, local_rank, world, dev)


def all_reduce_mean(value: float, d: Dist) -> float:
    if not d.enabled:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=d.device)
    dist.all_reduce(t)
    return t.item() / d.world


# --------------------------------------------------------------------- data
class ShardedTokens:
    """Random fixed-length windows from ``uint16`` token shards (memory-mapped)."""

    def __init__(self, data_dir: str | Path, split: str, seq_len: int) -> None:
        self.data_dir = Path(data_dir)
        meta = json.loads((self.data_dir / META_NAME).read_text())
        self.vocab_size = int(meta["vocab_size"])
        self.seq_len = seq_len
        self.shards: List[np.memmap] = []
        for s in meta.get(split, []):
            arr = np.memmap(self.data_dir / s["file"], dtype=np.uint16, mode="r")
            if len(arr) > seq_len + 1:
                self.shards.append(arr)
        if not self.shards:
            raise ValueError(f"no {split} shard in {data_dir} is longer than seq_len+1={seq_len + 1}")
        sizes = np.array([len(a) - seq_len - 1 for a in self.shards], dtype=np.float64)
        self.weights = sizes / sizes.sum()
        self.num_tokens = int(sum(len(a) for a in self.shards))

    def batch(self, batch_size: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        shard_ids = rng.choice(len(self.shards), size=batch_size, p=self.weights)
        rows = []
        for si in shard_ids:
            arr = self.shards[si]
            start = int(rng.integers(0, len(arr) - self.seq_len - 1))
            rows.append(np.asarray(arr[start : start + self.seq_len + 1], dtype=np.int64))
        buf = torch.from_numpy(np.stack(rows))
        return buf[:, :-1], buf[:, 1:]


def make_rng(seed: int, rank: int, step: int) -> np.random.Generator:
    """Per-rank generator, re-derived on resume so data never repeats a stream."""
    return np.random.default_rng([seed, rank, step])


# -------------------------------------------------------------------- model
def flops_per_token(cfg: MNXConfig, n_params: int, seq_len: int) -> float:
    """Training FLOPs per token: 6N for the matmuls plus causal attention."""
    return 6.0 * n_params + 12.0 * cfg.n_layers * cfg.d_model * seq_len


def build_model(cfg: MNXConfig, args: PretrainArgs, d: Dist) -> MNXCoder:
    """Build on the meta device, shard with FSDP2 and materialise per rank."""
    with torch.device("meta"):
        model = MNXCoder(cfg)
    model.gradient_checkpointing = args.grad_checkpointing
    if args.compile:
        for block in model.blocks:
            block.compile()

    param_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if d.enabled:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
        for block in model.blocks:
            fully_shard(block, mp_policy=mp)
        fully_shard(model, mp_policy=mp)
    model.to_empty(device=d.device)
    model.init_weights()
    return model


def autocast_ctx(args: PretrainArgs, d: Dist):
    """FSDP2 handles mixed precision itself; single-process runs use autocast."""
    if d.enabled or args.dtype == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    return torch.autocast(device_type=d.device.type, dtype=dtype)


# -------------------------------------------------------------- checkpoints
def _state(model: MNXCoder, optimizer: torch.optim.Optimizer):
    from torch.distributed.checkpoint.state_dict import get_state_dict

    msd, osd = get_state_dict(model, optimizer)
    return {"model": msd, "optim": osd}


def save_checkpoint(out_dir: Path, step: int, model, optimizer, extra: dict, args: PretrainArgs, d: Dist) -> None:
    import torch.distributed.checkpoint as dcp

    ckpt_dir = out_dir / f"step_{step:08d}"
    dcp.save(_state(model, optimizer), checkpoint_id=str(ckpt_dir))
    if d.is_main:
        (ckpt_dir / "trainer.json").write_text(json.dumps({"step": step, **extra}, indent=2))
        tmp = out_dir / (LATEST_NAME + ".tmp")
        tmp.write_text(ckpt_dir.name)
        tmp.replace(out_dir / LATEST_NAME)
        old = sorted(p for p in out_dir.glob("step_*") if p.is_dir())
        for p in old[: max(0, len(old) - args.keep_checkpoints)]:
            shutil.rmtree(p, ignore_errors=True)
    if d.enabled:
        dist.barrier()


def latest_checkpoint(out_dir: Path) -> Optional[Path]:
    pointer = out_dir / LATEST_NAME
    if not pointer.exists():
        return None
    path = out_dir / pointer.read_text().strip()
    return path if path.exists() else None


def load_checkpoint_into(ckpt_dir: Path, model, optimizer) -> dict:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import set_state_dict

    state = _state(model, optimizer)
    dcp.load(state, checkpoint_id=str(ckpt_dir))
    set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"])
    return json.loads((ckpt_dir / "trainer.json").read_text())


def export(run: str | Path, out: str | Path, dtype: str = "bf16") -> Path:
    """Consolidate the latest sharded checkpoint of ``run`` into ``out/ckpt.pt``."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    run, out = Path(run), Path(out)
    ckpt_dir = latest_checkpoint(run)
    if ckpt_dir is None:
        raise FileNotFoundError(f"no checkpoint found in {run}")
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "full_state.tmp"
    dcp_to_torch_save(str(ckpt_dir), str(tmp))
    full = torch.load(tmp, map_location="cpu", weights_only=False)
    tmp.unlink()
    cast = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    weights = {k: v.to(cast) if v.is_floating_point() else v for k, v in full["model"].items()}
    cfg = json.loads((run / CONFIG_NAME).read_text())
    info = json.loads((ckpt_dir / "trainer.json").read_text())
    torch.save({"model": weights, "step": info["step"], "config": cfg, "best_val": info.get("best_val")}, out / CKPT_NAME)
    (out / CONFIG_NAME).write_text(json.dumps(cfg, indent=2))
    shutil.copy(run / TOKENIZER_NAME, out / TOKENIZER_NAME)
    print(f"exported {ckpt_dir.name} -> {out / CKPT_NAME}")
    return out / CKPT_NAME


# -------------------------------------------------------------------- train
@torch.no_grad()
def evaluate(model, val: ShardedTokens, args: PretrainArgs, d: Dist) -> float:
    model.eval()
    rng = make_rng(args.seed + 1, d.rank, 0)  # same windows at every eval
    total = 0.0
    for _ in range(args.eval_iters):
        x, y = val.batch(args.micro_batch, rng)
        with autocast_ctx(args, d):
            _, loss, _ = model(x.to(d.device, non_blocking=True), y.to(d.device, non_blocking=True))
        total += loss.item()
    model.train()
    return all_reduce_mean(total / args.eval_iters, d)


def pretrain(args: PretrainArgs) -> Dict[str, List[float]]:
    """Run (or resume) pretraining; returns a history dict on every rank."""
    d = setup_distributed(args.device)
    torch.manual_seed(args.seed)
    if d.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out_dir = Path(args.out)
    if d.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    overrides = dict(args.overrides)
    if args.seq_len:
        overrides["max_seq_len"] = args.seq_len
    cfg = get_config(args.config, **overrides)
    train_data = ShardedTokens(args.data, "train", cfg.max_seq_len)
    if train_data.vocab_size != cfg.vocab_size:
        cfg = cfg.replace(vocab_size=train_data.vocab_size)
    try:
        val_data: Optional[ShardedTokens] = ShardedTokens(args.data, "val", cfg.max_seq_len)
    except ValueError:
        val_data = None
    seq_len = cfg.max_seq_len

    model = build_model(cfg, args, d)
    n_params = model.num_parameters()
    optimizer = build_optimizer(model, args.lr, args.weight_decay, (args.beta1, args.beta2))
    min_lr = args.min_lr if args.min_lr is not None else args.lr / 10

    start_step, best_val = 0, float("inf")
    resume_dir = latest_checkpoint(out_dir)
    if resume_dir is not None:
        info = load_checkpoint_into(resume_dir, model, optimizer)
        start_step = int(info["step"])
        best_val = float(info.get("best_val") or best_val)

    tokens_per_step = args.micro_batch * args.grad_accum * seq_len * d.world
    flops_step = flops_per_token(cfg, n_params, seq_len) * tokens_per_step
    if d.is_main:
        shutil.copy(Path(args.data) / "tokenizer.json", out_dir / TOKENIZER_NAME)
        (out_dir / CONFIG_NAME).write_text(json.dumps(cfg.to_dict(), indent=2))
        (out_dir / "pretrain_args.json").write_text(json.dumps(asdict(args), indent=2))
        print(
            f"model {cfg.name}: {MNXCoder.format_params(n_params)} params | world {d.world} | "
            f"{tokens_per_step:,} tokens/step | {args.steps * tokens_per_step / 1e9:.1f}B tokens total | "
            f"train data {train_data.num_tokens / 1e9:.2f}B tokens",
            flush=True,
        )
        if resume_dir is not None:
            print(f"resumed from {resume_dir} at step {start_step}", flush=True)

    log_file = writer = None
    if d.is_main:
        log_path = out_dir / LOG_NAME
        new_log = start_step == 0 or not log_path.exists()
        log_file = open(log_path, "w" if new_log else "a", newline="")
        writer = csv.writer(log_file)
        if new_log:
            writer.writerow(["step", "lr", "train_loss", "val_loss", "grad_norm", "tokens_per_s", "mfu", "elapsed_s"])

    history: Dict[str, List[float]] = {"step": [], "train_loss": [], "val_loss": []}
    rng = make_rng(args.seed, d.rank, start_step)
    model.train()
    t0 = t_last = time.time()
    peak = args.peak_tflops * 1e12 * d.world

    for step in range(start_step, args.steps):
        lr = cosine_lr(step, args.steps, args.lr, min_lr, args.warmup)
        for g in optimizer.param_groups:
            g["lr"] = lr

        loss_acc = 0.0
        for micro in range(args.grad_accum):
            if d.enabled and hasattr(model, "set_requires_gradient_sync"):
                model.set_requires_gradient_sync(micro == args.grad_accum - 1)
            x, y = train_data.batch(args.micro_batch, rng)
            with autocast_ctx(args, d):
                _, loss, _ = model(x.to(d.device, non_blocking=True), y.to(d.device, non_blocking=True))
            (loss / args.grad_accum).backward()
            loss_acc += loss.detach().float().item() / args.grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf"))
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if not math.isfinite(loss_acc):
            raise FloatingPointError(f"non-finite loss at step {step + 1}; lower --lr or resume from the last checkpoint")

        done = step + 1
        is_last = done == args.steps
        val_loss: Optional[float] = None
        if val_data is not None and args.eval_interval > 0 and (done % args.eval_interval == 0 or is_last):
            val_loss = evaluate(model, val_data, args, d)
            best_val = min(best_val, val_loss)

        if done % args.log_interval == 0 or is_last or step == start_step or val_loss is not None:
            loss_avg = all_reduce_mean(loss_acc, d)
            now = time.time()
            steps_since = done - (history["step"][-1] if history["step"] else start_step)
            dt = max(1e-9, now - t_last)
            tok_s = steps_since * tokens_per_step / dt
            mfu = steps_since * flops_step / dt / peak
            t_last = now
            if d.is_main:
                msg = f"step {done:>7}/{args.steps} | lr {lr:.2e} | loss {loss_avg:.4f}"
                if val_loss is not None:
                    msg += f" | val {val_loss:.4f}"
                msg += f" | gnorm {float(grad_norm):.2f} | {tok_s:,.0f} tok/s | mfu {mfu:.1%} | {now - t0:7.1f}s"
                print(msg, flush=True)
                writer.writerow([
                    done, f"{lr:.6e}", f"{loss_avg:.6f}", "" if val_loss is None else f"{val_loss:.6f}",
                    f"{float(grad_norm):.4f}", f"{tok_s:.0f}", f"{mfu:.4f}", f"{now - t0:.1f}",
                ])
                log_file.flush()
            history["step"].append(done)
            history["train_loss"].append(loss_avg)
            history["val_loss"].append(float("nan") if val_loss is None else val_loss)

        if args.save_interval > 0 and (done % args.save_interval == 0 or is_last):
            save_checkpoint(out_dir, done, model, optimizer, {"best_val": best_val}, args, d)

    if log_file is not None:
        log_file.close()
    if d.is_main:
        print(f"done. latest checkpoint: {latest_checkpoint(out_dir)}  best val loss: {best_val:.4f}", flush=True)
    if d.enabled:
        dist.barrier()
        dist.destroy_process_group()
    return history


# ---------------------------------------------------------------------- CLI
def parse_args(argv: Optional[Sequence[str]] = None) -> PretrainArgs:
    a = PretrainArgs()
    p = argparse.ArgumentParser(description="Pretrain MNX 2.0 Max Coder with FSDP2 (launch with torchrun).")
    p.add_argument("--config", default=a.config, help=f"preset name, one of {sorted(set(PRESETS))}")
    p.add_argument("--data", default=a.data, help="shard directory written by `python -m mnx.prepare`")
    p.add_argument("--out", default=a.out, help="run directory (resumes automatically if it has a checkpoint)")
    p.add_argument("--steps", type=int, default=a.steps)
    p.add_argument("--micro-batch", type=int, default=a.micro_batch, help="sequences per GPU per micro-step")
    p.add_argument("--grad-accum", type=int, default=a.grad_accum)
    p.add_argument("--seq-len", type=int, default=None, help="override preset max_seq_len")
    p.add_argument("--lr", type=float, default=a.lr)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--warmup", type=int, default=a.warmup)
    p.add_argument("--weight-decay", type=float, default=a.weight_decay)
    p.add_argument("--grad-clip", type=float, default=a.grad_clip)
    p.add_argument("--eval-interval", type=int, default=a.eval_interval)
    p.add_argument("--eval-iters", type=int, default=a.eval_iters)
    p.add_argument("--log-interval", type=int, default=a.log_interval)
    p.add_argument("--save-interval", type=int, default=a.save_interval)
    p.add_argument("--keep-checkpoints", type=int, default=a.keep_checkpoints)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=a.dtype)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--compile", action="store_true", help="torch.compile each transformer block")
    p.add_argument("--peak-tflops", type=float, default=a.peak_tflops, help="per-GPU peak for MFU (H100 bf16 dense: 989)")
    p.add_argument("--seed", type=int, default=a.seed)
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config fields, e.g. n_layers=2")
    ns = p.parse_args(argv)
    overrides: Dict[str, object] = {}
    for item in ns.set:
        k, v = item.split("=", 1)
        overrides[k] = json.loads(v) if v not in ("true", "false") else (v == "true")
    return PretrainArgs(
        config=ns.config, data=ns.data, out=ns.out, steps=ns.steps, micro_batch=ns.micro_batch,
        grad_accum=ns.grad_accum, seq_len=ns.seq_len, lr=ns.lr, min_lr=ns.min_lr, warmup=ns.warmup,
        weight_decay=ns.weight_decay, grad_clip=ns.grad_clip, eval_interval=ns.eval_interval,
        eval_iters=ns.eval_iters, log_interval=ns.log_interval, save_interval=ns.save_interval,
        keep_checkpoints=ns.keep_checkpoints, dtype=ns.dtype, grad_checkpointing=ns.grad_checkpointing,
        compile=ns.compile, peak_tflops=ns.peak_tflops, seed=ns.seed, device=ns.device, overrides=overrides,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "export":
        p = argparse.ArgumentParser(description="Export the latest sharded checkpoint to ckpt.pt.")
        p.add_argument("--run", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
        ns = p.parse_args(argv[1:])
        export(ns.run, ns.out, ns.dtype)
        return
    pretrain(parse_args(argv))


if __name__ == "__main__":
    main()
