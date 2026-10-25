"""Measure the neural coder on its own: pass@1 and pass@k on def + assert tasks.

    python scripts/eval_neural.py --ckpt runs/cpu data/eval/code_holdout2.jsonl

A task counts for pass@k if any of k samples passes *all* tests, including the
hidden ones the model never saw. Only the shown asserts go into the prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnx.agent.expert_stream import prompt_for  # noqa: E402
from mnx.agent.skills.neural import assemble, run_tests, split_prompt_and_tests, torch_sampler  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("benchmarks", nargs="+")
    p.add_argument("--ckpt", required=True)
    p.add_argument("-k", type=int, default=16)
    p.add_argument("--no-tests-in-prompt", action="store_true")
    a = p.parse_args()
    sample = torch_sampler(a.ckpt)
    for bench in a.benchmarks:
        items = [json.loads(l) for l in Path(bench).read_text().splitlines()]
        items = [it for it in items if split_prompt_and_tests(it["question"])]
        p1 = pk = 0
        for it in items:
            prompt, shown = split_prompt_and_tests(it["question"])
            full = prompt if a.no_tests_in_prompt else prompt_for(prompt, shown)
            passes = [run_tests(assemble(prompt, c), it["tests"])[0] for c in sample(full, a.k)]
            p1 += passes[0]
            pk += any(passes)
            print(f"{'✓' if any(passes) else '·'} {sum(passes):>2}/{a.k}  {prompt.splitlines()[0]}")
        print(f"{bench}: {len(items)} tasks, pass@1 {p1}/{len(items)}, pass@{a.k} {pk}/{len(items)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
