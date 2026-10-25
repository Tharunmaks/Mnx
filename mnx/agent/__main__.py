"""Command line interface for MNX1::

    python -m mnx.agent "solve x^2 - 5x + 6 = 0"
    python -m mnx.agent "write a python function that checks for palindromes"
    python -m mnx.agent --trace "All men are mortal. Socrates is a man. Is Socrates mortal?"
    python -m mnx.agent                      # interactive chat
    python -m mnx.agent --kb ~/notes --memory ~/.mnx/memory.jsonl
    python -m mnx.agent --tools web,gmail,drive     # optional: the web, Gmail and Drive (see docs/tools.md)
    python -m mnx.agent --tools web --chrome        # read pages through your own Chrome (Playwright)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from mnx.agent.agent import MNXAgent, format_answer
from mnx.agent.retrieval import default_knowledge_dir
from mnx.agent.tools import build_tools


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
    p.add_argument("--tools", default="", help="optional online tools, comma separated: web, gmail, drive, google (= gmail,drive), all")
    p.add_argument("--chrome", action="store_true", help="with --tools web: open pages in Chrome via Playwright instead of fetching them")
    p.add_argument("--cdp", default="http://localhost:9222", help="Chrome remote-debugging URL to attach to (default http://localhost:9222)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dirs = ([] if args.no_default_kb else [default_knowledge_dir()]) + [Path(k).expanduser() for k in args.kb]
    tools = build_tools(args.tools.split(","), chrome=args.chrome, cdp=args.cdp) if args.tools else []
    agent = MNXAgent(tools=tools, knowledge_dirs=dirs, memory_path=Path(args.memory).expanduser() if args.memory else None, ckpt=args.ckpt,
                     state_dir=None if args.no_learn else Path(args.state).expanduser(), learn=not args.no_learn)
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
