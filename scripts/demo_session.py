"""Run a scripted MNX1 session and write the questions and answers as Markdown.

    python scripts/demo_session.py > session.md

Uses a throwaway state directory, so it shows learning (corrections,
feedback, replay) without touching ``~/.mnx``.
"""

from __future__ import annotations

import datetime as dt
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnx.agent import MNXAgent  # noqa: E402
from mnx.agent.agent import SKILL_TO_AGENT  # noqa: E402

QUESTIONS = [
    "hi",
    "write a python function that checks if a string is a palindrome",
    "implement merge sort",
    "write a function called fib that returns the nth fibonacci number",
    "def mystery(a, b):\n    \"\"\"Find the rule.\"\"\"\nassert mystery(2, 3) == 13\nassert mystery(1, 1) == 2\nassert mystery(0, 4) == 16\nassert mystery(3, 0) == 9",
    "check this code:\n```python\ndef average(xs):\n    return total / len(xs)\n```",
    "run this:\n```python\nprint([n * n for n in range(6)])\n```",
    "write a function that sends an email via smtp",
    "what is 123456789 times 987654321",
    "solve x^3 - 6x^2 + 11x - 6 = 0",
    "solve x + y = 10, x - y = 2",
    "is 1000003 prime?",
    "convert 26.2 miles to km",
    "what day of the week was July 20, 1969",
    "All dolphins are mammals. All mammals breathe air. Flipper is a dolphin. Does Flipper breathe air?",
    "If something is a dog then it is loyal. Rex is not loyal. Is Rex a dog?",
    "1, 3, 7, 15, 31, ?",
    "'John Smith' -> 'JS', 'Ada Lovelace' -> 'AL', 'Alan Turing' -> ?",
    "who discovered penicillin",
    "what is the capital of Australia",
    "generate an image of a cat",
    "What is the capital of Wakanda?",
    "wrong, the answer is Birnin Zana",
    "what's the capital of wakanda?",
    "remember that my favourite language is Python",
    "what is my favourite language?",
    "Who will win the next World Cup?",
]


def main() -> int:
    state = Path(tempfile.mkdtemp(prefix="mnx1-demo-"))
    agent = MNXAgent(state_dir=state, today=dt.date(2026, 10, 25))
    out = ["# MNX1 sample session", "",
           "Every answer below was produced by `MNXAgent` with no LLM and no network. "
           "Each entry shows which agent answered, which skill it used, and whether the answer was verified.", ""]
    for i, q in enumerate(QUESTIONS, 1):
        a = agent.ask(q)
        tag = "verified" if a.verified else ("could not answer" if a.failed or a.skill == "abstain" else "unverified")
        agent_name = SKILL_TO_AGENT.get(a.skill, a.skill)
        quoted = "\n".join("> " + line for line in q.splitlines())
        out += [f"### {i}.", "", quoted, "", a.text, "", f"*{agent_name} / {a.skill} · {tag} · confidence {a.confidence:.2f}*", ""]
    s = agent.learner.stats
    out += ["---", "", f"Learned during this session: {s['self_trained']} router examples from verified answers, "
            f"{s['corrected']} correction, {len(agent.learner.experience)} solved problems remembered for instant replay."]
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
