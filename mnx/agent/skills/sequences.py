"""Next-term prediction for integer / rational sequences.

Hypotheses are tried from simplest to most complex (a crude minimum
description length prior). Each one is *fitted on the fewest terms it needs*
and then must reproduce every remaining given term; an answer is marked
verified only if at least one term was held out and predicted correctly.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Callable, List, Optional, Tuple

from mnx.agent.base import Answer, Skill, fmt_number

Hyp = Tuple[Fraction, str, int]  # (next term, description, terms needed to fit)


def _primes(n: int) -> List[int]:
    out, c = [], 2
    while len(out) < n:
        if all(c % p for p in out if p * p <= c):
            out.append(c)
        c += 1
    return out


def _known() -> List[Tuple[str, List[int]]]:
    fib = [0, 1]
    while len(fib) < 40:
        fib.append(fib[-1] + fib[-2])
    cat = [math.comb(2 * n, n) // (n + 1) for n in range(25)]
    return [
        ("prime numbers", _primes(60)),
        ("Fibonacci numbers", fib),
        ("factorials", [math.factorial(n) for n in range(20)]),
        ("Catalan numbers", cat),
        ("perfect squares", [n * n for n in range(60)]),
        ("perfect cubes", [n**3 for n in range(40)]),
        ("triangular numbers", [n * (n + 1) // 2 for n in range(60)]),
        ("powers of 2", [2**n for n in range(40)]),
        ("powers of 3", [3**n for n in range(30)]),
        ("Lucas numbers", [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843, 1364, 2207]),
    ]


_KNOWN = _known()


def _named(terms: List[Fraction]) -> Optional[Hyp]:
    if len(terms) < 4 or any(t.denominator != 1 for t in terms):
        return None
    ints = [int(t) for t in terms]
    for name, seq in _KNOWN:
        for start in range(0, len(seq) - len(ints)):
            if seq[start:start + len(ints)] == ints:
                return Fraction(seq[start + len(ints)]), f"the {name}", 2
    return None


def _constant(t: List[Fraction]) -> Optional[Hyp]:
    if all(x == t[0] for x in t):
        return t[0], f"constant sequence ({fmt_number(t[0])})", 1
    return None


def _poly(t: List[Fraction]) -> Optional[Hyp]:
    rows = [t]
    while len(rows[-1]) > 1 and not all(x == rows[-1][0] for x in rows[-1]):
        r = rows[-1]
        rows.append([b - a for a, b in zip(r, r[1:])])
        if len(rows) > len(t) - 1:
            return None
    last = rows[-1]
    if len(last) < 2:  # constant row has a single element: nothing held out
        return None
    deg = len(rows) - 1
    nxt = Fraction(last[0])
    for r in reversed(rows[:-1]):
        nxt = r[-1] + nxt
    if deg == 1:
        desc = f"arithmetic: add {fmt_number(rows[1][0])} each time"
    else:
        desc = f"polynomial of degree {deg} (difference table becomes constant after {deg} steps)"
    return nxt, desc, deg + 1


def _geometric(t: List[Fraction]) -> Optional[Hyp]:
    if len(t) < 3 or any(x == 0 for x in t[:-1]):
        return None
    r = t[1] / t[0]
    if r in (0, 1) or any(b != a * r for a, b in zip(t, t[1:])):
        return None
    return t[-1] * r, f"geometric: multiply by {fmt_number(r)} each time", 2


def _affine_rec(t: List[Fraction]) -> Optional[Hyp]:
    """a(n) = p·a(n-1) + q."""
    if len(t) < 4 or t[1] == t[0]:
        return None
    p = (t[2] - t[1]) / (t[1] - t[0])
    q = t[1] - p * t[0]
    if any(t[i] != p * t[i - 1] + q for i in range(2, len(t))):
        return None
    return p * t[-1] + q, f"a(n) = {fmt_number(p)}·a(n-1) {'+' if q >= 0 else '-'} {fmt_number(abs(q))}", 3


def _linear_rec2(t: List[Fraction]) -> Optional[Hyp]:
    """a(n) = p·a(n-1) + q·a(n-2) (Fibonacci-like)."""
    if len(t) < 5:
        return None
    det = t[1] * t[1] - t[0] * t[2]
    if det == 0:
        return None
    p = (t[2] * t[1] - t[0] * t[3]) / det
    q = (t[1] * t[3] - t[2] * t[2]) / det
    if any(t[i] != p * t[i - 1] + q * t[i - 2] for i in range(2, len(t))):
        return None
    return p * t[-1] + q * t[-2], f"a(n) = {fmt_number(p)}·a(n-1) + {fmt_number(q)}·a(n-2)", 4


def _differences(t: List[Fraction]) -> Optional[Hyp]:
    """Differences follow a simpler (geometric / recurrence) pattern."""
    if len(t) < 5:
        return None
    d = [b - a for a, b in zip(t, t[1:])]
    for f in (_geometric, _affine_rec):
        h = f(d)
        if h:
            return t[-1] + h[0], f"differences {', '.join(fmt_number(x) for x in d)} follow: {h[1]}", h[2] + 1
    return None


def _ratios(t: List[Fraction]) -> Optional[Hyp]:
    """Ratios form an arithmetic sequence (e.g. factorial-like)."""
    if len(t) < 5 or any(x == 0 for x in t):
        return None
    r = [b / a for a, b in zip(t, t[1:])]
    h = _poly(r)
    if h and h[2] <= 2:
        return t[-1] * h[0], f"ratios {', '.join(fmt_number(x) for x in r)} are arithmetic", h[2] + 1
    return None


def _alternating(t: List[Fraction]) -> Optional[Hyp]:
    """Two interleaved arithmetic sequences."""
    if len(t) < 6:
        return None
    ev, od = t[0::2], t[1::2]
    he, ho = _poly(ev), _poly(od)
    if he and ho and he[2] <= 2 and ho[2] <= 2:
        nxt = he[0] if len(t) % 2 == 0 else ho[0]
        return nxt, "two interleaved sequences (odd and even positions)", 4
    return None


HYPOTHESES: List[Callable[[List[Fraction]], Optional[Hyp]]] = [
    _constant, _named, _geometric, _poly, _affine_rec, _linear_rec2, _differences, _ratios, _alternating,
]


def predict_next(terms: List[Fraction]) -> Optional[Tuple[Fraction, str, bool, List[str]]]:
    trace = []
    for h in HYPOTHESES:
        res = h(terms)
        if res is None:
            continue
        nxt, desc, need = res
        held_out = len(terms) - need
        trace.append(f"hypothesis '{desc}' fits all {len(terms)} terms ({max(held_out, 0)} held out beyond the {need} needed to fit)")
        # second opinion: does any *other* hypothesis disagree on the next term?
        rivals = []
        for other in HYPOTHESES:
            if other is h:
                continue
            r = other(terms)
            if r and r[0] != nxt and len(terms) - r[2] >= 1:
                rivals.append(f"{r[1]} → {fmt_number(r[0])}")
        if rivals:
            trace.append("alternative patterns also fit: " + "; ".join(rivals))
        return nxt, desc, held_out >= 1, trace
    return None


_SEQ = re.compile(r"(-?\d+(?:/\d+)?(?:\.\d+)?(?:\s*,\s*-?\d+(?:/\d+)?(?:\.\d+)?){2,})")


class SequenceSkill(Skill):
    name = "sequence"
    description = "find the pattern in a number sequence and predict the next term"

    def answer(self, question: str) -> Optional[Answer]:
        low = question.lower()
        if not re.search(r"\b(next|sequence|series|pattern|comes after|continue|following)\b|,\s*\.\.\.|,\s*_+|\?\s*$", low):
            return None
        m = _SEQ.search(low)
        if not m:
            return None
        terms = [Fraction(x.strip()) for x in m.group(1).split(",")]
        if len(terms) < 3:
            return None
        res = predict_next(terms)
        if res is None:
            return Answer("I can't find a pattern I can verify in that sequence.", self.name, 0.4, False,
                          ["tried: constant, named sequences, geometric, polynomial, linear recurrences, differences, ratios, interleaved"],
                          failed=True)
        nxt, desc, verified, trace = res
        conf = 0.93 if verified else 0.55
        if any("alternative patterns" in t for t in trace):
            conf -= 0.15
        return Answer(f"{fmt_number(nxt)} — {desc}.", self.name, conf, verified, trace)
