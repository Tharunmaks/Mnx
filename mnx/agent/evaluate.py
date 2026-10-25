"""Evaluate the agent on the bundled benchmark::

    python -m mnx.agent.evaluate            # summary table
    python -m mnx.agent.evaluate --verbose  # every question

Reports, per category: accuracy, how often it answered wrongly (the number
that matters most for trust), and how often it correctly abstained on
questions it cannot answer. Also reports the router's 5-fold
cross-validated accuracy on its own training data.

The benchmark is small and written alongside the system, so these numbers
show the system works as designed; they are **not** comparable to public
LLM benchmarks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from mnx.agent.agent import MNXAgent
from mnx.agent.router import cross_validate, default_intents_path, load_intents


def default_benchmark() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "eval" / "benchmark.jsonl"


def load_benchmark(path: Optional[Path] = None) -> List[dict]:
    return [json.loads(l) for l in Path(path or default_benchmark()).read_text(encoding="utf-8").splitlines() if l.strip()]


def _contains(text: str, expected: str) -> bool:
    """Case-insensitive match of ``expected`` as a whole word/phrase."""
    return re.search(r"(?<!\w)" + re.escape(expected.lower()) + r"(?!\w)", text.lower()) is not None


def run(items: List[dict], agent: Optional[MNXAgent] = None, verbose: bool = False) -> Dict[str, dict]:
    agent = agent or MNXAgent(today=dt.date(2026, 1, 1))
    stats: Dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0, "wrong": 0, "abstained": 0, "ms": 0.0})
    for it in items:
        start = time.perf_counter()
        ans = agent.ask(it["question"])
        ms = (time.perf_counter() - start) * 1000
        s = stats[it["category"]]
        s["n"] += 1
        s["ms"] += ms
        abstained = ans.skill == "abstain" or ans.failed
        if it["expect"] is None:
            ok = abstained
        else:
            ok = not abstained and any(_contains(ans.text, e) for e in it["expect"])
        if ok:
            s["correct"] += 1
        elif abstained:
            s["abstained"] += 1
        else:
            s["wrong"] += 1
        if verbose:
            mark = "✓" if ok else ("·" if abstained else "✗")
            print(f"{mark} [{it['category']}] {it['question']}\n    → {ans.text.splitlines()[0][:110]}  ({ans.skill}, {ms:.0f} ms)")
    return dict(stats)


def report(stats: Dict[str, dict]) -> str:
    lines = [f"{'category':<11} {'n':>3} {'correct':>8} {'wrong':>6} {'abstain':>8} {'avg ms':>7}"]
    tot = {"n": 0, "correct": 0, "wrong": 0, "abstained": 0, "ms": 0.0}
    for cat, s in stats.items():
        lines.append(f"{cat:<11} {s['n']:>3} {s['correct'] / s['n']:>8.0%} {s['wrong']:>6} {s['abstained']:>8} {s['ms'] / s['n']:>7.0f}")
        for k in tot:
            tot[k] += s[k]
    lines.append(f"{'total':<11} {tot['n']:>3} {tot['correct'] / tot['n']:>8.0%} {tot['wrong']:>6} {tot['abstained']:>8} {tot['ms'] / tot['n']:>7.0f}")
    lines.append("('wrong' = answered but incorrect; for the abstain category, correct means it said it didn't know)")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--benchmark", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    stats = run(load_benchmark(args.benchmark), verbose=args.verbose)
    print(report(stats))
    print(f"\nrouter 5-fold cross-validated accuracy: {cross_validate(load_intents(default_intents_path())):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
