"""Score a local LLM on MNX's evaluation sets, the same way MNX is scored.

Runs entirely on your machine through Ollama (https://ollama.com), so no API
key is needed::

    ollama pull qwen2.5:0.5b
    python scripts/compare_models.py --model qwen2.5:0.5b
    python scripts/compare_models.py --model llama3.2:1b data/eval/code_holdout2.jsonl

Scoring matches ``python -m mnx.agent.evaluate``:

- questions with ``expect``: correct if an expected string appears (whole words)
- coding questions: the first ```python block must pass the hidden tests in a sandbox
- questions MNX should decline (``expect`` is null, no tests): correct only
  if the model says it doesn't know or can't answer
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnx.agent.evaluate import _contains, report  # noqa: E402
from mnx.agent.skills.code import extract_code  # noqa: E402
from mnx.agent.skills.neural import run_tests  # noqa: E402

DEFAULT_SETS = ["data/eval/holdout.jsonl", "data/eval/code_holdout.jsonl", "data/eval/code_holdout2.jsonl"]
SYSTEM = ("Answer briefly. If the question asks for code, reply with one complete Python function in a ```python block, "
          "using the function name the question uses. If you don't know or can't do it, say \"I don't know\".")
_DECLINE = re.compile(r"\b(i don'?t know|i do not know|i'?m not sure|cannot (be )?(predict|know|determine)|can'?t (predict|know|"
                      r"determine|generate|create)|unable to|no way to know|impossible to (know|predict))\b", re.I)


def ask(model: str, question: str, host: str) -> str:
    body = json.dumps({"model": model, "stream": False, "options": {"temperature": 0},
                       "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]})
    req = urllib.request.Request(f"{host}/api/chat", body.encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"]


def score(item: dict, text: str) -> str:
    if item.get("tests"):
        code = extract_code(text)
        return "correct" if code and run_tests(code, item["tests"], timeout=10)[0] else (
            "abstained" if _DECLINE.search(text) else "wrong")
    if item["expect"] is None:
        return "correct" if _DECLINE.search(text) else "wrong"
    if any(_contains(text, e) for e in item["expect"]):
        return "correct"
    return "abstained" if _DECLINE.search(text) else "wrong"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("benchmarks", nargs="*", default=DEFAULT_SETS)
    p.add_argument("--model", required=True, help="an Ollama model name, e.g. qwen2.5:0.5b")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    for bench in a.benchmarks:
        stats = defaultdict(lambda: {"n": 0, "correct": 0, "wrong": 0, "abstained": 0, "ms": 0.0})
        for line in Path(bench).read_text().splitlines():
            item = json.loads(line)
            text = ask(a.model, item["question"], a.host)
            verdict = score(item, text)
            s = stats[item["category"]]
            s["n"] += 1
            s[verdict] += 1
            if a.verbose:
                print(f"{verdict[0].upper()} {item['question'].splitlines()[0][:70]}  →  {text.strip().splitlines()[0][:80] if text.strip() else ''}")
        print(f"== {a.model} on {bench}\n{report(dict(stats))}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
