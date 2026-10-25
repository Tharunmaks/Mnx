"""Score MNX1 on HumanEval, the public 164-problem Python benchmark.

    pip download human-eval --no-deps -d /tmp/he && unzip -o /tmp/he/human_eval-*.whl -d /tmp/he
    python scripts/humaneval.py /tmp/he/human_eval/data/HumanEval.jsonl.gz

MNX1 gets what any model gets: the prompt (signature and docstring). The
docstring's own examples (``>>> f(1)`` then ``2``, or ``f(1) ==> 2``) are
turned into asserts, because MNX1's coder works from tests. The answer counts
only if it passes HumanEval's hidden ``check`` function. Abstaining counts as
not solved, so the number is pass@1 over all 164 problems.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnx.agent import MNXAgent  # noqa: E402
from mnx.agent.skills.code import extract_code  # noqa: E402
from mnx.agent.skills.neural import run_tests  # noqa: E402


def doc_asserts(prompt: str, name: str) -> list:
    """Examples written in the docstring, as assert lines."""
    out = []
    lines = prompt.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(">>>") and name in s and i + 1 < len(lines):
            want = lines[i + 1].strip()
            if want and not want.startswith(">>>"):
                out.append(f"assert {s[3:].strip()} == {want}")
            continue
        m = re.match(rf"^.*?\b({name}\(.*\))\s*(?:==>|➞|=>|->|==|should return|returns?)\s*(.+?)\.?$", s)
        if m:
            out.append(f"assert {m.group(1)} == {m.group(2)}")
    good = []
    for a in out:
        try:
            compile(a, "<doc>", "exec")
            good.append(a)
        except SyntaxError:
            continue
    return good


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", default=None, help="write per-problem results as JSONL")
    p.add_argument("--raw-prompt", action="store_true",
                   help="hand MNX1 only the prompt, as a user would paste it, and let it find the docstring examples")
    a = p.parse_args()
    items = [json.loads(l) for l in gzip.open(a.data, "rt")]
    agent = MNXAgent(learn=False, ckpt=a.ckpt)
    solved = wrong = abstained = 0
    rows = []
    for it in items:
        name = it["entry_point"]
        asserts = doc_asserts(it["prompt"], name)
        question = it["prompt"].strip("\n") + ("" if a.raw_prompt else "\n" + "\n".join(asserts))
        ans = agent.ask(question)
        code = None if ans.skill == "abstain" or ans.failed else extract_code(ans.text)
        if not code:
            abstained += 1
            verdict = "abstained"
        else:
            header = "\n".join(l for l in it["prompt"].splitlines() if l.startswith(("import ", "from ")))
            ok, _ = run_tests(header + "\n" + code, [it["test"], f"check({name})"], timeout=10)
            solved += ok
            wrong += not ok
            verdict = "solved" if ok else "wrong"
        rows.append({"task_id": it["task_id"], "verdict": verdict, "asserts": len(asserts), "skill": ans.skill})
        if a.verbose:
            print(f"{verdict:<9} {it['task_id']:<14} {name:<28} asserts={len(asserts)} skill={ans.skill}")
    n = len(items)
    print(f"HumanEval: {n} problems | solved {solved} ({solved / n:.1%}) | wrong {wrong} | abstained {abstained}")
    if a.out:
        Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
