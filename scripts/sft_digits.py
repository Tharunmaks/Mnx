"""Digit-aligned supervised fine-tuning for the MNX cpu checkpoint.

    python scripts/sft_digits.py --ckpt runs/cpu --out runs/digit --steps 9000

What it does differently from mnx/trainer_bot.py:

* the run directory gets a *digit-split* tokenizer: the same merges, the same
  vocab, the same ids for every non-numeric string, but the pre-tokeniser
  pattern cuts digit runs into single digits, so every digit of every number
  is its own token at its own position. Nothing else in the repo changes:
  ``load_checkpoint`` reads ``tokenizer.json`` out of the run directory, so
  ``scripts/eval_qa.py`` picks the new segmentation up untouched.
* no quiz, no grading, no replay: one gradient pass over packed 256-token
  windows, which is where the compute actually belongs.
* the data is a fixed mixture: a *maintenance* share drawn from
  ``mnx.agent.qa_stream`` exactly as pretraining drew it (so code, syllogism,
  primality ... keep being rehearsed), and a *targeted* share from widened
  samplers for the sub-skills that are both weak and reachable.

Every question that is trained on is appended to ``qa_history.jsonl`` so that
``eval_qa.py --exclude`` can drop it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnx.agent.qa_stream import _fmt, stream  # noqa: E402
from mnx.model import MNXConfig, MNXCoder  # noqa: E402
from mnx.tokenizer import DEFAULT_PATTERN, BPETokenizer  # noqa: E402
from mnx.train import build_optimizer, cosine_lr  # noqa: E402

DIGIT_PATTERN = DEFAULT_PATTERN.replace(r"| ?\d+", r"| ?\d")
PROMPT = "Q: {q}\nA:"


# --------------------------------------------------------------- tokenizer
def digit_tokenizer(src: Path) -> BPETokenizer:
    """The checkpoint's own tokenizer with digit runs split into single digits."""
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    d["pattern"] = DIGIT_PATTERN
    return BPETokenizer.from_dict(d)


# ------------------------------------------------------------------ samplers
# Phrasings are copied from mnx/agent/qa_stream.py so the surface form the model
# trains on is the surface form it is evaluated on. The *parameter ranges* are
# deliberately wider than the evaluation generator's, which (a) keeps the test
# range in the interior of the training range instead of at its edge and
# (b) makes a collision with an evaluation question vanishingly unlikely.
_WORD = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}


def _digits(r: random.Random, lo: int = 1, hi: int = 7) -> int:
    """A magnitude with a uniformly chosen number of digits (1..7)."""
    d = r.randint(lo, hi)
    return r.randint(0, 9) if d == 1 else r.randint(10 ** (d - 1), 10**d - 1)


def _signed(r: random.Random, v: int, p: float = 0.3) -> int:
    return -v if r.random() < p else v


def _ask(r: random.Random, a: int, op: str, b: int, ans: str) -> Tuple[str, str]:
    q = r.choice([f"what is {a} {op} {b}", f"what is {a} {_WORD[op]} {b}",
                  f"calculate {a} {op} {b}", f"{a} {op} {b} = ?"])
    return q, ans


def g_addsub(r: random.Random) -> Tuple[str, str]:
    a, b = _signed(r, _digits(r)), _signed(r, _digits(r))
    if r.random() < 0.25:                      # same length: carry/borrow chains
        b = _signed(r, r.randint(10 ** (len(str(abs(a))) - 1), 10 ** len(str(abs(a))) - 1))
    op = r.choice("+-")
    return _ask(r, a, op, b, str(a + b if op == "+" else a - b))


def g_mul_easy(r: random.Random) -> Tuple[str, str]:
    """Only the reachable half of multiplication: one factor of 1-2 digits,
    or a factor that is a 1-2 digit number followed by zeros."""
    big = _signed(r, _digits(r, 1, 7))
    k = r.random()
    if k < 0.45:
        small = _signed(r, r.randint(0, 9))
    elif k < 0.8:
        small = _signed(r, r.randint(10, 99))
    else:
        small = _signed(r, r.randint(1, 99) * 10 ** r.randint(1, 4))
    a, b = (big, small) if r.random() < 0.5 else (small, big)
    return _ask(r, a, "*", b, str(a * b))


def g_div(r: random.Random) -> Tuple[str, str]:
    a, b = _signed(r, _digits(r)), _signed(r, _digits(r)) or 7
    if r.random() < 0.15:                      # exact division shows up too
        q = _signed(r, _digits(r, 1, 4))
        a = q * b
    return _ask(r, a, "/", b, _fmt(Fraction(a, b)))


def g_percent(r: random.Random) -> Tuple[str, str]:
    p = r.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 75, r.randint(1, 99)])
    n = r.randint(1, 20000)
    q = r.choice([f"what is {p}% of {n}", f"{p} percent of {n}"])
    return q, _fmt(Fraction(p * n, 100))


def g_gcd(r: random.Random) -> Tuple[str, str]:
    a, b = r.randint(2, 2000), r.randint(2, 2000)
    if r.random() < 0.3:                       # make sure non-trivial gcds occur
        g = r.randint(2, 40)
        a, b = g * r.randint(1, 2000 // g), g * r.randint(1, 2000 // g)
    q = r.choice([f"gcd of {a} and {b}", f"greatest common divisor of {a} and {b}"])
    return q, str(math.gcd(a, b))


def g_linear(r: random.Random) -> Tuple[str, str]:
    a = r.choice([i for i in range(-15, 16) if i])
    x, b = r.randint(-40, 40), r.randint(-80, 80)
    c = a * x + b
    lhs = f"{a}x" if a not in (1, -1) else ("x" if a == 1 else "-x")
    lhs += f" + {b}" if b > 0 else (f" - {-b}" if b < 0 else "")
    return f"solve {lhs} = {c}", f"x = {x}"


def g_system2(r: random.Random) -> Tuple[str, str]:
    x, y = r.randint(-30, 30), r.randint(-30, 30)
    while True:
        a, b, c, d = (r.choice([1, 2, 3, 4, -1, -2]) for _ in range(4))
        if a * d - b * c:
            break
    def eq(p, s):
        return f"{p}x" + (f" + {s}y" if s >= 0 else f" - {-s}y")
    q = f"solve {eq(a, b)} = {a * x + b * y}, {eq(c, d)} = {c * x + d * y}"
    return q.replace("1x", "x").replace(" 1y", " y"), f"x = {x}, y = {y}"


def _date(r: random.Random):
    import datetime as dt
    return dt.date(1850, 1, 1) + dt.timedelta(days=r.randint(0, 300 * 365))


def g_dates(r: random.Random) -> Tuple[str, str]:
    import datetime as dt
    d = _date(r)
    if r.random() < 0.5:
        n = r.randint(1, 1200)
        return f"what date is {n} days after {d.isoformat()}", (d + dt.timedelta(days=n)).isoformat()
    e = d + dt.timedelta(days=r.randint(1, 2500))
    return f"how many days between {d.isoformat()} and {e.isoformat()}", f"{(e - d).days} days"


TARGETED: Dict[str, Tuple] = {
    "addsub":  (g_addsub, 0.35),
    "div":     (g_div, 0.15),
    "mul":     (g_mul_easy, 0.12),
    "dates":   (g_dates, 0.12),
    "system2": (g_system2, 0.10),
    "gcd":     (g_gcd, 0.08),
    "linear":  (g_linear, 0.04),
    "percent": (g_percent, 0.04),
}


def mixture(seed: int, maintenance: float, weights: Optional[Dict[str, float]] = None) -> Iterator[dict]:
    """Infinite stream of {'q','a','kind'}: `maintenance` of the items straight
    from qa_stream (the pretraining distribution), the rest from TARGETED."""
    r = random.Random(seed)
    maint = stream(10**12, seed=seed + 1)
    names = list(TARGETED)
    w = [(weights or {}).get(k, TARGETED[k][1]) for k in names]
    while True:
        if r.random() < maintenance:
            it = next(maint)
            yield {"q": it["q"], "a": it["a"], "kind": "maint/" + it["kind"]}
        else:
            k = r.choices(names, w)[0]
            q, a = TARGETED[k][0](r)
            yield {"q": q, "a": a, "kind": k}


# -------------------------------------------------------------------- packing
class Packer:
    """Items -> (x, y) windows. Loss is taken on answer tokens only; prompt
    tokens, padding and the item separator are -100."""

    def __init__(self, tok: BPETokenizer, seq_len: int, items: Iterator[dict]):
        self.tok, self.seq_len, self.items = tok, seq_len, items
        self.carry: List[Tuple[List[int], List[int]]] = []
        self.n_items = 0
        self.n_answer_tokens = 0
        self.n_tokens = 0
        self.by_kind: Dict[str, int] = {}
        self.trained: List[dict] = []   # questions used since the last drain

    def _encode(self, it: dict) -> Tuple[List[int], List[int]]:
        p = self.tok.encode(PROMPT.format(q=it["q"]), add_bos=True, allow_special=False)
        a = self.tok.encode(" " + it["a"], allow_special=False) + [self.tok.eos_id]
        ids = (p + a)[: self.seq_len + 1]
        tgt = ([-100] * len(p) + a)[: self.seq_len + 1]
        return ids, tgt

    def window(self) -> Tuple[List[int], List[int]]:
        ids: List[int] = []
        tgt: List[int] = []
        while len(ids) < self.seq_len + 1:
            if not self.carry:
                it = next(self.items)
                self.carry.append(self._encode(it))
                self.n_items += 1
                self.by_kind[it["kind"]] = self.by_kind.get(it["kind"], 0) + 1
                self.trained.append(it)
            i, t = self.carry[0]
            if ids and len(ids) + len(i) > self.seq_len + 1:
                break
            self.carry.pop(0)
            ids += i
            tgt += t
        pad = self.seq_len + 1 - len(ids)
        ids += [self.tok.pad_id] * pad
        tgt += [-100] * pad
        self.n_tokens += self.seq_len
        self.n_answer_tokens += sum(1 for v in tgt[1:] if v != -100)
        return ids, tgt

    def batch(self, bs: int, device: str = "cpu"):
        rows = [self.window() for _ in range(bs)]
        x = torch.tensor([r[0][:-1] for r in rows], dtype=torch.long, device=device)
        y = torch.tensor([r[1][1:] for r in rows], dtype=torch.long, device=device)
        return x, y


# ----------------------------------------------------------------------- run
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", default="runs/cpu")
    p.add_argument("--out", default="runs/digit")
    p.add_argument("--steps", type=int, default=9000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--maintenance", type=float, default=0.25)
    p.add_argument("--weights", default=None, help='JSON, e.g. \'{"dates":0.0,"addsub":0.47}\'')
    p.add_argument("--digit-tokenizer", type=int, default=1, help="0 keeps the checkpoint's tokenizer (ablation)")
    p.add_argument("--save-interval", type=int, default=250)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--history", type=int, default=1, help="0 skips writing qa_history.jsonl")
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    (out).mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"

    tok_path = out / "tokenizer.json"
    if not tok_path.exists():
        src = Path(a.ckpt)
        src = src / "tokenizer.json" if src.is_dir() else src.parent / "tokenizer.json"
        (digit_tokenizer(src) if a.digit_tokenizer else BPETokenizer.load(src)).save(tok_path)
    tok = BPETokenizer.load(tok_path)

    state = torch.load(ckpt_path if ckpt_path.exists() else
                       (Path(a.ckpt) / "ckpt.pt" if Path(a.ckpt).is_dir() else Path(a.ckpt)), map_location="cpu")
    model = MNXCoder(MNXConfig.from_dict(state["config"]))
    model.load_state_dict(state["model"])
    model.train()
    opt = build_optimizer(model, a.lr, a.weight_decay, (0.9, 0.95))
    start = 0
    if ckpt_path.exists():
        opt.load_state_dict(state["optimizer"])
        start = int(state.get("sft_step", 0))
        print(f"resumed {ckpt_path} at step {start}")

    weights = json.loads(a.weights) if a.weights else None
    items = mixture(a.seed + 7 * (start + 1), a.maintenance, weights)
    packer = Packer(tok, a.seq_len, items)
    hist = (out / "qa_history.jsonl").open("a", buffering=1 << 20) if a.history else None

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))

    def save(step: int) -> None:
        tmp = out / "ckpt.pt.tmp"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "config": model.cfg.to_dict(), "step": state.get("step", 0), "sft_step": step,
                    "sft_args": vars(a)}, tmp)
        os.replace(tmp, ckpt_path)
        (out / "config.json").write_text(json.dumps(model.cfg.to_dict(), indent=2))
        if hist:
            hist.flush()

    log = (out / "sft_log.csv").open("a", buffering=1)
    if start == 0:
        log.write("step,lr,loss,items,answer_tokens,elapsed_s\n")
    t0 = time.time()
    for step in range(start, a.steps):
        lr = cosine_lr(step, a.steps, a.lr, a.min_lr, a.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = packer.batch(a.batch_size)
        if hist:
            for it in packer.trained:
                hist.write(json.dumps({"q": it["q"], "kind": it["kind"]}) + "\n")
        packer.trained.clear()
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        if (step + 1) % a.log_interval == 0 or step == start:
            el = time.time() - t0
            print(f"step {step+1:>6}/{a.steps} | lr {lr:.2e} | loss {loss.item():.4f} | "
                  f"{packer.n_items} items | {packer.n_answer_tokens/1e3:.0f}k answer tok | {el:6.0f}s", flush=True)
            log.write(f"{step+1},{lr:.6e},{loss.item():.6f},{packer.n_items},{packer.n_answer_tokens},{el:.1f}\n")
        if (step + 1) % a.save_interval == 0 or step + 1 == a.steps or stop["now"]:
            save(step + 1)
        if stop["now"]:
            print("stopped; checkpoint saved")
            break
    print(json.dumps({"items": packer.n_items, "tokens": packer.n_tokens,
                      "answer_tokens": packer.n_answer_tokens,
                      "by_kind": dict(sorted(packer.by_kind.items()))}, indent=1))
    if hist:
        hist.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
