"""Small talk and self-description, from templates (no generation)."""

from __future__ import annotations

import re
from typing import Optional

from mnx.agent.base import Answer, Skill

HELP = (
    "I'm MNX, a small from-scratch assistant that runs locally with no LLM or API key. I can:\n"
    "  • do exact math: 2^100, 15% of 240, solve x^2 - 5x + 6 = 0, x + y = 10, x - y = 2, factor, expand, derivatives\n"
    "  • convert units: 5 km to miles, 100 F in C\n"
    "  • work with dates: what day was 1969-07-20, days between two dates\n"
    "  • reason logically: All men are mortal. Socrates is a man. Is Socrates mortal?\n"
    "  • continue number sequences: 1, 1, 2, 3, 5, 8, ?\n"
    "  • write programs from examples: [1,2,3] -> [2,4,6], [5] -> [10]\n"
    "  • answer factual questions from my local knowledge base, and remember what you teach me\n"
    "I check my answers where I can and say \"I don't know\" instead of guessing."
)

_RULES = [
    (r"^(hi|hello|hey|hiya|yo|good (morning|afternoon|evening))\b.*$", "Hello! Ask me a math, logic, date, unit or factual question. Type 'help' to see what I can do."),
    (r"^(thanks|thank you|thx|cheers|nice|great|cool|ok|okay|nice job|good job)\b.*$", "You're welcome."),
    (r"^(bye|goodbye|see you|quit|exit)\b.*$", "Goodbye!"),
    (r"^how are you\b.*$", "I'm a program, so I'm always the same, but thanks for asking. What can I work out for you?"),
    (r"^(help|what can you do|what do you do|commands|\?)\W*$", HELP),
    (r"^(who|what) are you\W*$|^tell me about yourself\W*$|^what('s| is) your name\W*$", HELP),
    (r"^(are you|is this) (an? )?(llm|large language model|chatgpt|gpt|ai)\W*$",
     "I'm an AI, but not a large language model: I'm a router plus verified symbolic solvers, search and retrieval, all written from scratch."),
    (r"^how do you work\W*$",
     "A small naive Bayes router picks likely skills; each skill (algebra, logic, dates, units, sequences, program search, retrieval) "
     "tries to answer and checks itself; I return the best verified answer or say I don't know. Ask with --trace to see the steps."),
]


class ChatSkill(Skill):
    name = "chat"
    description = "greetings, help and self-description"

    def answer(self, question: str) -> Optional[Answer]:
        q = question.strip().lower()
        for pat, reply in _RULES:
            if re.match(pat, q):
                return Answer(reply, self.name, 0.9, True, ["matched a conversation template"])
        return None
