"""Score the trained model on fresh Q&A pairs it never saw in training.

    python scripts/eval_qa.py --ckpt runs/cpu --db runs/qa --n 600

Pairs come from the same generators with a different seed; any question that
is also in the training database is dropped. The model answers greedily
after ``Q: ...\\nA:`` and must match the stored answer exactly (code answers
must pass the task's tests instead).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnx.agent.qa_stream import read, stream  # noqa: E402
from mnx.agent.skills.neural import run_tests  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--db", default="runs/qa")
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--seed", type=int, default=12345)
    a = p.parse_args()

    from mnx.generate import complete
    from mnx.train import load_checkpoint

    model, tok, _ = load_checkpoint(a.ckpt)
    train_q = {hash(it["q"]) for it in read(Path(a.db))}
    items = [it for it in stream(a.n * 3, seed=a.seed) if hash(it["q"]) not in train_q][: a.n]
    stats = defaultdict(lambda: [0, 0])
    for it in items:
        out = complete(model, tok, f"Q: {it['q']}\nA:", max_new_tokens=120 if it["kind"] == "code" else 40, temperature=0.0)
        if it["kind"] == "code":
            tests = [l for l in it["q"].splitlines() if l.startswith("assert ")]
            ok = bool(tests) and run_tests(out.strip() + "\n", tests)[0]
        else:
            ok = out.strip().split("\n")[0].strip() == it["a"].strip()
        s = stats[it["kind"]]
        s[0] += 1
        s[1] += ok
    tot = [sum(v[0] for v in stats.values()), sum(v[1] for v in stats.values())]
    for k, (n, c) in sorted(stats.items()):
        print(f"{k:<11} {c:>4}/{n:<4} {c / n:>5.0%}")
    print(f"{'total':<11} {tot[1]:>4}/{tot[0]:<4} {tot[1] / tot[0]:>5.0%}  (unseen questions, {len(train_q)} in training db)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
