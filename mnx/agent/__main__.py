"""Command line interface::

    python -m mnx.agent "solve x^2 - 5x + 6 = 0"
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
    p = argparse.ArgumentParser(prog="mnx-agent", description="MNX: a small from-scratch assistant (no LLM, no API key).")
    p.add_argument("question", nargs="*", help="question to answer; omit for interactive mode")
    p.add_argument("--trace", action="store_true", help="show routing, verification and proof steps")
    p.add_argument("--kb", action="append", default=[], help="extra folder of .md/.txt files to index (repeatable)")
    p.add_argument("--no-default-kb", action="store_true", help="don't index the bundled data/knowledge folder")
    p.add_argument("--memory", default=None, help="JSONL file to persist 'remember that ...' facts")
    p.add_argument("--ckpt", default=None, help="MNXCoder checkpoint dir: enables sample-and-verify code generation (needs torch)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dirs = ([] if args.no_default_kb else [default_knowledge_dir()]) + [Path(k).expanduser() for k in args.kb]
    agent = MNXAgent(knowledge_dirs=dirs, memory_path=Path(args.memory).expanduser() if args.memory else None, ckpt=args.ckpt)
    if args.question:
        print(format_answer(agent.ask(" ".join(args.question)), args.trace))
        return 0
    print("MNX agent: type a question, 'help' for examples, ':trace' to toggle traces, Ctrl-D to quit.")
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
        print("mnx>", format_answer(agent.ask(q), trace).replace("\n", "\n     "))


if __name__ == "__main__":
    sys.exit(main())
