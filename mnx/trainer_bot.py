"""24/7 autonomous CPU Q&A trainer bot for the MNX coder.

Tharun's design (handoff doc "24/7 Autonomous CPU Q&A AI Trainer Bot"),
implemented against MNX's own transformer and its verified Q&A generators
instead of a NumPy model, so it trains the checkpoint MNX1 actually uses.

The loop, per his spec:

1. **Batch construction** mixes fresh Q&A from the generator hub with
   high-priority past items from the replay buffer (``--replay-ratio``).
2. **Pre-update quiz**: the model answers ``Q: ...\\nA:`` greedily before any
   weights change.
3. **Grading**: token F1 (65%) plus exact numeric match (35%) on
   ``[0.0, 1.0]``.
4. **Adaptive inner epochs**: a batch graded below ``--pass-mark`` gets
   ``--active-epochs`` updates, a batch above it gets one maintenance update.
5. **Post-update verification** re-quizzes the same batch, and every ten steps
   the curriculum level (1-5) moves on the 50-step rolling score.
6. **Prioritized replay** keeps what the model got wrong coming back:
   ``priority = max(0.05, loss * mistake_boost + novelty_boost)``.
7. **Atomic checkpoints** after every ``--save-interval`` steps and on SIGINT,
   so a 24/7 run survives being killed.

Run it:

    python -m mnx.trainer_bot --ckpt runs/cpu_code --workdir runs/bot
    python -m mnx.trainer_bot --workdir runs/bot --status

Everything stays on the CPU: ``--cpu-threads`` caps the thread count and
``--sleep-ms`` leaves the machine usable (and a phone cool).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from mnx.agent.qa_stream import GENERATORS, stream

# Which generators each curriculum level is allowed to draw from. Level 5 is
# everything, including writing code.
LEVELS: Dict[int, Tuple[str, ...]] = {
    1: ("arithmetic", "leap", "units"),
    2: ("arithmetic", "leap", "units", "powers", "percent", "gcd_lcm", "primality", "sequences"),
    3: ("arithmetic", "leap", "units", "powers", "percent", "gcd_lcm", "primality", "sequences",
        "linear", "factorise", "syllogism"),
    4: ("arithmetic", "leap", "units", "powers", "percent", "gcd_lcm", "primality", "sequences",
        "linear", "factorise", "syllogism", "quadratic", "system2", "dates"),
    5: tuple(GENERATORS),
}

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-z0-9_.\-]+")


# ------------------------------------------------------------------ grading
def token_f1(pred: str, gold: str) -> float:
    p, g = _WORD.findall(pred.lower()), _WORD.findall(gold.lower())
    if not p or not g:
        return float(p == g)
    common = 0
    pool = list(g)
    for t in p:
        if t in pool:
            pool.remove(t)
            common += 1
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def numeric_match(pred: str, gold: str) -> Optional[float]:
    """1.0 when every number in the answer is reproduced, ``None`` when the answer has none."""
    gold_nums = _NUM.findall(gold)
    if not gold_nums:
        return None
    pred_nums = _NUM.findall(pred)
    hit = 0
    pool = list(pred_nums)
    for n in gold_nums:
        match = next((m for m in pool if _close(m, n)), None)
        if match is not None:
            pool.remove(match)
            hit += 1
    return hit / len(gold_nums)


def _close(a: str, b: str) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    except ValueError:
        return a == b


def grade_prediction(pred: str, gold: str) -> float:
    """Score an answer on [0, 1]: token F1 (65%) plus exact numeric match (35%)."""
    pred = pred.strip().split("\n")[0].strip()
    f1 = token_f1(pred, gold)
    num = numeric_match(pred, gold)
    if num is None:
        return f1
    return 0.65 * f1 + 0.35 * num


# ------------------------------------------------------------- replay buffer
@dataclass
class Item:
    q: str
    a: str
    kind: str
    last_loss: float = 1.0
    last_score: float = 0.0
    times_seen: int = 0

    @property
    def priority(self) -> float:
        mistake_boost = 1.5 if self.last_score < 0.6 else 1.0
        novelty_boost = 1.0 / (1.0 + 0.15 * self.times_seen)
        return max(0.05, self.last_loss * mistake_boost + novelty_boost)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 20000, seed: int = 0):
        self.capacity = capacity
        self.items: Dict[str, Item] = {}
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def add(self, item: Item) -> None:
        self.items.setdefault(item.q, item)
        if len(self.items) > self.capacity:  # drop the least useful
            worst = min(self.items.values(), key=lambda i: i.priority)
            self.items.pop(worst.q, None)

    def sample(self, k: int, kinds: Sequence[str]) -> List[Item]:
        pool = [i for i in self.items.values() if i.kind in kinds]
        if not pool or k <= 0:
            return []
        weights = [i.priority for i in pool]
        picked: List[Item] = []
        for _ in range(min(k, len(pool))):
            choice = self.rng.choices(pool, weights)[0]
            idx = pool.index(choice)
            pool.pop(idx)
            weights.pop(idx)
            picked.append(choice)
        return picked

    def to_json(self) -> list:
        return [{"q": i.q, "a": i.a, "kind": i.kind, "last_loss": i.last_loss,
                 "last_score": i.last_score, "times_seen": i.times_seen} for i in self.items.values()]

    def load_json(self, rows: list) -> None:
        for r in rows:
            self.items[r["q"]] = Item(r["q"], r["a"], r["kind"], r.get("last_loss", 1.0),
                                      r.get("last_score", 0.0), r.get("times_seen", 0))


# -------------------------------------------------------------- generator hub
class QAGeneratorHub:
    """Fresh Q&A at the current curriculum level, plus anything dropped in a hot folder."""

    def __init__(self, seed: int = 0, data_path: Optional[Path] = None):
        self.seed = seed
        self.data_path = Path(data_path) if data_path else None
        self._mtimes: Dict[Path, float] = {}
        self._pending: List[dict] = []
        self._it: Iterator[dict] = stream(10**9, seed=seed)

    def _hot_reload(self) -> None:
        if not self.data_path or not self.data_path.exists():
            return
        files = [self.data_path] if self.data_path.is_file() else sorted(self.data_path.glob("*"))
        for f in files:
            if f.suffix not in (".jsonl", ".json", ".txt"):
                continue
            mtime = f.stat().st_mtime
            if self._mtimes.get(f) == mtime:
                continue
            self._mtimes[f] = mtime
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q, a = row.get("q") or row.get("question"), row.get("a") or row.get("answer")
                if q and a:
                    self._pending.append({"q": str(q), "a": str(a), "kind": row.get("kind", "custom")})

    def take(self, k: int, kinds: Sequence[str]) -> List[Item]:
        self._hot_reload()
        out: List[Item] = []
        while self._pending and len(out) < k:
            row = self._pending.pop(0)
            out.append(Item(row["q"], row["a"], row["kind"]))
        guard = 0
        while len(out) < k and guard < 2000:
            guard += 1
            row = next(self._it)
            if row["kind"] in kinds:
                out.append(Item(row["q"], row["a"], row["kind"]))
        return out


# ---------------------------------------------------------------- the bot
PROMPT = "Q: {q}\nA:"


@dataclass
class BotConfig:
    ckpt: str = "runs/cpu_code"
    workdir: Path = Path("runs/bot")
    steps: int = 0  # 0 = until stopped
    batch_size: int = 4
    replay_ratio: float = 0.5
    active_epochs: int = 3
    pass_mark: float = 0.65
    lr: float = 1e-4
    max_answer_tokens: int = 48
    cpu_threads: int = 0
    sleep_ms: int = 50
    save_interval: int = 50
    seed: int = 0
    level: int = 1
    data_path: Optional[Path] = None
    verify: bool = True
    history: List[float] = field(default_factory=list)


class ContinuousQATrainerBot:
    def __init__(self, cfg: BotConfig):
        import torch

        self.torch = torch
        if cfg.cpu_threads:
            torch.set_num_threads(cfg.cpu_threads)
        self.cfg = cfg
        self.dir = Path(cfg.workdir)
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        from mnx.train import load_checkpoint

        resume = self.dir / "checkpoints" / "latest_model.pt"
        source = str(resume) if resume.exists() else cfg.ckpt
        self.model, self.tok, _ = load_checkpoint(source)
        self.model.train()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=0.01, betas=(0.9, 0.95))
        self.buffer = PrioritizedReplayBuffer(seed=cfg.seed)
        self.hub = QAGeneratorHub(seed=cfg.seed + 1, data_path=cfg.data_path)
        self.step = 0
        self.level = cfg.level
        self.rolling: List[float] = []
        self.stop = False
        self._load_state(source == str(resume))
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    # ------------------------------------------------------------- state
    def _state_path(self) -> Path:
        return self.dir / "checkpoints" / "bot_state.json"

    def _load_state(self, resumed: bool) -> None:
        p = self._state_path()
        if not (resumed and p.exists()):
            return
        state = json.loads(p.read_text())
        self.step = state.get("step", 0)
        self.level = state.get("level", self.level)
        self.rolling = state.get("rolling", [])[-50:]
        self.buffer.load_json(state.get("replay", []))
        print(f"resumed at step {self.step}, level {self.level}, {len(self.buffer)} replay items")

    def _save(self) -> None:
        ck = self.dir / "checkpoints"
        tmp = ck / "latest_model.pt.tmp"
        self.torch.save({"model": self.model.state_dict(), "config": self.model.cfg.to_dict(),
                         "optimizer": self.opt.state_dict(), "step": self.step}, tmp)
        os.replace(tmp, ck / "latest_model.pt")
        tok_path = ck / "tokenizer.json"
        if not tok_path.exists():
            self.tok.save(tok_path)
        tmp = ck / "bot_state.json.tmp"
        tmp.write_text(json.dumps({"step": self.step, "level": self.level, "rolling": self.rolling[-50:],
                                   "replay": self.buffer.to_json(), "saved_at": time.time()}))
        os.replace(tmp, self._state_path())

    def _on_signal(self, *_args) -> None:
        print("\nstopping: saving an atomic checkpoint first")
        self.stop = True

    # -------------------------------------------------------------- quiz
    def quiz(self, items: Sequence[Item]) -> List[Tuple[str, float]]:
        from mnx.generate import complete

        out = []
        self.model.eval()
        with self.torch.no_grad():
            for it in items:
                text = complete(self.model, self.tok, PROMPT.format(q=it.q),
                                max_new_tokens=self.cfg.max_answer_tokens, temperature=0.0)
                out.append((text, grade_prediction(text, it.a)))
        self.model.train()
        return out

    # ------------------------------------------------------------- update
    def _batch(self, items: Sequence[Item]):
        torch = self.torch
        rows, max_len = [], 0
        limit = self.model.cfg.max_seq_len
        for it in items:
            prompt = self.tok.encode(PROMPT.format(q=it.q), add_bos=True, allow_special=False)
            answer = self.tok.encode(" " + it.a, allow_special=False) + [self.tok.eos_id]
            ids = (prompt + answer)[:limit]
            mask = ([-100] * len(prompt) + answer)[:limit]
            if len(ids) < 2:
                continue
            rows.append((ids, mask))
            max_len = max(max_len, len(ids))
        if not rows:
            return None, None
        pad = self.tok.pad_id
        x = torch.full((len(rows), max_len), pad, dtype=torch.long)
        y = torch.full((len(rows), max_len), -100, dtype=torch.long)
        for i, (ids, mask) in enumerate(rows):
            x[i, : len(ids)] = torch.tensor(ids)
            y[i, : len(mask)] = torch.tensor(mask)
        # next-token targets: predict position t+1 from t
        return x[:, :-1].contiguous(), y[:, 1:].contiguous()

    def update(self, items: Sequence[Item], epochs: int) -> float:
        x, y = self._batch(items)
        if x is None:
            return 0.0
        last = 0.0
        for _ in range(max(1, epochs)):
            _, loss, _ = self.model(x, y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            last = float(loss.item())
        return last

    # --------------------------------------------------------------- loop
    def run_step(self) -> dict:
        kinds = LEVELS[self.level]
        n_replay = int(round(self.cfg.batch_size * self.cfg.replay_ratio))
        replayed = self.buffer.sample(n_replay, kinds)
        fresh = self.hub.take(self.cfg.batch_size - len(replayed), kinds)
        batch = replayed + fresh
        graded = self.quiz(batch)
        before = sum(s for _, s in graded) / len(graded)
        epochs = self.cfg.active_epochs if before < self.cfg.pass_mark else 1
        loss = self.update(batch, epochs)
        after = before
        if self.cfg.verify:
            after = sum(s for _, s in self.quiz(batch)) / len(batch)
        for it, (_, score) in zip(batch, graded):
            it.times_seen += 1
            it.last_score = score
            it.last_loss = loss
            self.buffer.add(it)
        self.step += 1
        self.rolling = (self.rolling + [after])[-50:]
        rolling = sum(self.rolling) / len(self.rolling)
        if self.step % 10 == 0:
            self._scale_curriculum(rolling)
        row = {"step": self.step, "level": self.level, "loss": round(loss, 4),
               "score_before": round(before, 4), "score_after": round(after, 4),
               "rolling": round(rolling, 4), "epochs": epochs, "replay": len(self.buffer),
               "kinds": sorted({i.kind for i in batch}), "time": round(time.time(), 1)}
        with (self.dir / "training_metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        with (self.dir / "qa_history.jsonl").open("a") as f:
            for it, (pred, score) in zip(batch, graded):
                f.write(json.dumps({"step": self.step, "kind": it.kind, "q": it.q, "gold": it.a,
                                    "pred": pred.strip().split("\n")[0][:200], "score": round(score, 3)}) + "\n")
        return row

    def _scale_curriculum(self, rolling: float) -> None:
        if rolling >= 0.75 and self.level < 5:
            self.level += 1
            print(f"curriculum up: level {self.level} (rolling {rolling:.2f})")
        elif rolling < 0.45 and self.level > 1:
            self.level -= 1
            print(f"curriculum down: level {self.level} (rolling {rolling:.2f})")

    def run(self) -> int:
        target = self.cfg.steps
        print(f"trainer bot: level {self.level}, batch {self.cfg.batch_size}, "
              f"{'forever' if not target else target} step(s), workdir {self.dir}")
        start = time.time()
        while not self.stop and (not target or self.step < target):
            row = self.run_step()
            if self.step % 10 == 0 or self.step == 1:
                print(f"step {self.step:>6} | L{row['level']} | loss {row['loss']:.3f} | "
                      f"score {row['score_before']:.2f} -> {row['score_after']:.2f} | "
                      f"rolling {row['rolling']:.2f} | replay {row['replay']} | {time.time() - start:.0f}s",
                      flush=True)
            if self.step % self.cfg.save_interval == 0:
                self._save()
            if self.cfg.sleep_ms:
                time.sleep(self.cfg.sleep_ms / 1000.0)
        self._save()
        print(f"saved at step {self.step}; rolling score {sum(self.rolling) / max(1, len(self.rolling)):.3f}")
        return 0


# --------------------------------------------------------------- status/CLI
def status(workdir: Path) -> dict:
    d = Path(workdir)
    state_path = d / "checkpoints" / "bot_state.json"
    out: dict = {"workdir": str(d), "running_state": "unknown"}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        out.update(step=state.get("step"), level=state.get("level"),
                   replay=len(state.get("replay", [])),
                   rolling=round(sum(state.get("rolling", [0])) / max(1, len(state.get("rolling", [0]))), 4),
                   saved_ago_s=round(time.time() - state.get("saved_at", 0)))
    metrics = d / "training_metrics.jsonl"
    if metrics.exists():
        rows = metrics.read_text().splitlines()[-200:]
        parsed = [json.loads(r) for r in rows if r.strip()]
        if parsed:
            out["last"] = parsed[-1]
            out["mean_score_last_200"] = round(sum(p["score_after"] for p in parsed) / len(parsed), 4)
            by_kind: Dict[str, List[float]] = {}
            for p in parsed:
                for k in p.get("kinds", []):
                    by_kind.setdefault(k, []).append(p["score_after"])
            out["by_kind"] = {k: round(sum(v) / len(v), 3) for k, v in sorted(by_kind.items())}
    history = d / "qa_history.jsonl"
    if history.exists():
        rows = [json.loads(r) for r in history.read_text().splitlines()[-5:] if r.strip()]
        out["recent_answers"] = rows
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mnx-trainer-bot", description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", default="runs/cpu_code", help="checkpoint to start from (ignored once the workdir has one)")
    p.add_argument("--workdir", default="runs/bot")
    p.add_argument("--steps", type=int, default=0, help="0 runs until stopped (SIGINT saves and exits)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--replay-ratio", type=float, default=0.5)
    p.add_argument("--active-epochs", type=int, default=3)
    p.add_argument("--pass-mark", type=float, default=0.65)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cpu-threads", type=int, default=0)
    p.add_argument("--sleep-ms", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--level", type=int, default=1, choices=sorted(LEVELS))
    p.add_argument("--data-path", default=None, help="hot folder or file of extra Q&A (.jsonl/.json/.txt)")
    p.add_argument("--no-verify", action="store_true", help="skip the post-update re-quiz (about a third faster)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--status", action="store_true", help="print what the bot has done and exit")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.status:
        print(json.dumps(status(Path(a.workdir)), indent=2)[:4000])
        return 0
    cfg = BotConfig(ckpt=a.ckpt, workdir=Path(a.workdir), steps=a.steps, batch_size=a.batch_size,
                    replay_ratio=a.replay_ratio, active_epochs=a.active_epochs, pass_mark=a.pass_mark,
                    lr=a.lr, cpu_threads=a.cpu_threads, sleep_ms=a.sleep_ms, save_interval=a.save_interval,
                    level=a.level, data_path=Path(a.data_path) if a.data_path else None,
                    verify=not a.no_verify, seed=a.seed)
    return ContinuousQATrainerBot(cfg).run()


if __name__ == "__main__":
    sys.exit(main())
