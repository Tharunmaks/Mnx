"""Shared types for the MNX agent: answers, skills and small text helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Answer:
    """One candidate answer produced by a skill.

    ``confidence`` is the skill's own estimate in ``[0, 1]``. ``verified`` is
    True only when the skill checked the answer independently (substituting a
    root back into the equation, re-running a synthesised program on every
    example, predicting a held-out term of a sequence, ...).
    """

    text: str
    skill: str
    confidence: float
    verified: bool = False
    trace: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    #: True when the skill is reporting that it could not answer
    #: ("no program fits", "no verifiable pattern"); counted as an abstention
    failed: bool = False

    def score(self, prior: float = 1.0) -> float:
        """Ranking score used by the agent to pick between skills."""
        bonus = 0.25 if self.verified else 0.0
        return min(1.0, self.confidence + bonus) * (0.6 + 0.4 * prior)


class Skill:
    """A capability the agent can route a question to.

    Subclasses implement :meth:`answer`, returning ``None`` when the question
    is outside what they can handle. Skills must never guess: returning
    ``None`` is always better than returning something unverified with high
    confidence.
    """

    name: str = "skill"
    description: str = ""

    def answer(self, question: str) -> Optional[Answer]:  # pragma: no cover - interface
        raise NotImplementedError


_WS = re.compile(r"\s+")


def normalize_space(text: str) -> str:
    return _WS.sub(" ", text).strip()


def strip_question(text: str) -> str:
    """Lower-case and drop common question lead-ins and trailing punctuation."""
    t = normalize_space(text).lower()
    t = re.sub(
        r"^(please\s+)?(can you\s+|could you\s+)?(tell me\s+)?"
        r"(what('s| is| are)\s+|whats\s+|calculate\s+|compute\s+|evaluate\s+|work out\s+|find\s+)?",
        "",
        t,
    )
    return t.rstrip(" ?.!")


def fmt_number(value) -> str:
    """Format a Fraction / int / float compactly for display."""
    from fractions import Fraction

    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        dec = float(value)
        return f"{value.numerator}/{value.denominator} (≈ {_fmt_float(dec)})"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _fmt_float(value)
    if isinstance(value, complex):
        im = "" if abs(value.imag) == 1 else _fmt_float(abs(value.imag))
        if value.real == 0:
            return f"{'-' if value.imag < 0 else ''}{im}i"
        sign = "-" if value.imag < 0 else "+"
        return f"{_fmt_float(value.real)} {sign} {im}i"
    return str(value)


def _fmt_float(x: float) -> str:
    if x != x or x in (float("inf"), float("-inf")):
        return str(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    s = f"{x:.10g}"
    return s
