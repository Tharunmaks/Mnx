"""Command line interface for MNX1::

    python -m mnx.agent "solve x^2 - 5x + 6 = 0"
    python -m mnx.agent "write a python function that checks for palindromes"
    python -m mnx.agent --trace "All men are mortal. Socrates is a man. Is Socrates mortal?"
    python -m mnx.agent                      # interactive chat
    python -m mnx.agent --kb ~/notes --memory ~/.mnx/memory.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from mnx.agent.agent import MNXAgent, format_answer
from mnx.agent.retrieval import default_knowledge_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mnx-agent", description="MNX1: a small from-scratch assistant (no LLM, no API key).")
    p.add_argument("question", nargs="*", help="question to answer; omit for interactive mode")
    p.add_argument("--trace", action="store_true", help="show routing, verification and proof steps")
    p.add_argument("--kb", action="append", default=[], help="extra folder of .md/.txt files to index (repeatable)")
    p.add_argument("--no-default-kb", action="store_true", help="don't index the bundled data/knowledge folder")
    p.add_argument("--memory", default=None, help="JSONL file to persist 'remember that ...' facts (default: <state>/memory.jsonl)")
    p.add_argument("--state", default="~/.mnx", help="folder where MNX1 keeps what it learns (default ~/.mnx)")
    p.add_argument("--no-learn", action="store_true", help="don't learn or log anything this run")
    p.add_argument("--ckpt", default=None, help="MNXCoder checkpoint dir: adds neural code proposals, kept only if tests pass (needs torch)")
    p.add_argument("--model", default=None, metavar="NAME",
                   help="also ask a code model served on localhost (e.g. qwen2.5-coder:7b via `ollama serve`) for "
                        "proposals; they are kept only if they pass the tests. Off by default")
    p.add_argument("--model-url", default=None, metavar="URL",
                   help="OpenAI-compatible base URL of that local server (default http://localhost:11434/v1)")
    p.add_argument("--model-samples", type=int, default=4, help="proposals to ask the local model for (default 4)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dirs = ([] if args.no_default_kb else [default_knowledge_dir()]) + [Path(k).expanduser() for k in args.kb]
    agent = MNXAgent(knowledge_dirs=dirs, memory_path=Path(args.memory).expanduser() if args.memory else None, ckpt=args.ckpt,
                     model=args.model, model_url=args.model_url, model_samples=args.model_samples,
                     state_dir=None if args.no_learn else Path(args.state).expanduser(), learn=not args.no_learn)
    if args.model:
        from mnx.agent.skills.local_model import DEFAULT_URL, LocalModelError, check

        try:
            have = check(args.model_url or DEFAULT_URL)
        except LocalModelError as exc:
            print(f"--model: {exc}", file=sys.stderr)
            return 2
        if args.model not in have and not any(h.startswith(args.model.split(":")[0]) for h in have):
            print(f"--model: the server has {have or 'no models'}; pull {args.model!r} first", file=sys.stderr)
            return 2
    if args.question:
        print(format_answer(agent.ask(" ".join(args.question)), args.trace))
        return 0
    print("MNX1: type a question, 'help' for examples, ':trace' to toggle traces, ':stats' for what it learned, Ctrl-D to quit.")
    trace = args.trace
    while True:
        try:
            q = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q.strip() == ":trace":
            trace = not trace
            print(f"trace {'on' if trace else 'off'}")
            continue
        if q.strip().lower() in ("quit", "exit", ":q"):
            return 0
        if q.strip() == ":stats" and agent.learner:
            print(f"learned so far: {agent.learner.stats}, {len(agent.learner.intents)} router examples, "
                  f"{len(agent.learner.corrections)} corrections, {len(agent.learner.experience)} remembered solutions")
            continue
        print("mnx>", format_answer(agent.ask(q), trace).replace("\n", "\n     "))


if __name__ == "__main__":
    sys.exit(main())
