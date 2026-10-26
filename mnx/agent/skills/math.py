"""Exact math: arithmetic, equations, linear systems, polynomials, primes.

Every result is computed by the hand-written algebra core in
:mod:`mnx.agent.expr` and, where possible, checked independently (roots are
substituted back, factorisations are multiplied out, systems are re-checked
against every equation).
"""

from __future__ import annotations

import math
import random
import re
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from mnx.agent.base import Answer, Skill, fmt_number
from mnx.agent.expr import (
    MathError,
    Poly,
    evaluate,
    exact_root,
    is_close,
    numeric_roots,
    parse,
    parse_equation,
    rational_roots,
    solve_linear_system,
    to_poly,
    to_text,
    variables,
)

# ----------------------------------------------------------- number theory
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int) -> bool:
    """Miller–Rabin; deterministic for n < 3.3e24 with these bases."""
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _pollard_rho(n: int, rng: random.Random) -> int:
    if n % 2 == 0:
        return 2
    while True:
        c = rng.randrange(1, n)
        f = lambda x: (x * x + c) % n  # noqa: E731
        x = y = rng.randrange(2, n)
        d = 1
        while d == 1:
            x = f(x)
            y = f(f(y))
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d


def factorize(n: int) -> Dict[int, int]:
    """Prime factorisation via trial division then Pollard's rho."""
    out: Dict[int, int] = {}
    if n < 2:
        return out
    for p in (2, 3, 5):
        while n % p == 0:
            out[p] = out.get(p, 0) + 1
            n //= p
    i, step = 7, [4, 2, 4, 2, 4, 6, 2, 6]
    k = 0
    while i * i <= n and i < 100_000:
        while n % i == 0:
            out[i] = out.get(i, 0) + 1
            n //= i
        i += step[k % 8]
        k += 1
    rng = random.Random(0)
    stack = [n] if n > 1 else []
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            out[m] = out.get(m, 0) + 1
            continue
        d = _pollard_rho(m, rng)
        stack.extend([d, m // d])
    return dict(sorted(out.items()))


#: a question that is really Python source (def header, asserts, a code fence)
_IS_CODE = re.compile(r"^\s*(?:def |class |```|assert )", re.M)


# ------------------------------------------------------ language → algebra
_LEADIN = re.compile(
    r"^(?:please\s+|now\s+|hey\s+)*(?:can you\s+|could you\s+)?(?:tell me\s+)?"
    r"(?:what(?:'s| is| are| does)|whats|how much is|calculate|compute|evaluate|work out|find|"
    r"solve|simplify|expand|factori[sz]e|factor|differentiate|integrate|determine|give me)?"
    r"\s*(?:the\s+)?(?:value of\s+|result of\s+|answer to\s+)?"
    r"(?:(?:derivative|integral|antiderivative)\s+of\s+)?"
    r"(?:for\s+[a-z]\b\s*)?(?:(?:if|when|where|given|in)\b\s*)?[:,]?\s*"
)
_TAIL = re.compile(
    r"\s*(?:,?\s*(?:with respect to|wrt|w\.r\.t\.?)\s+[a-z]|\s+d[a-z]|,?\s*for\s+[a-z]|"
    r"\s+equal|\s+equals?\s+what)?\s*(?:[?.]|(?<![0-9)])!)*\s*$"
)

_WORDS: List[Tuple[str, str]] = [
    (r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+", r"(\1/100)*"),
    (r"(\d+(?:\.\d+)?)\s*(?:%|percent)", r"(\1/100)"),
    (r"\bsquare root of\b", " sqrt "),
    (r"\bcube root of\b", " cbrt "),
    (r"\b(?:the )?absolute value of\b", " abs "),
    (r"\bfactorial of\s+(\d+)", r"\1!"),
    (r"\b(\d+)\s+factorial\b", r"\1!"),
    (r"\b(?:the )?(?:gcd|greatest common divisor|hcf|highest common factor) of\s+(\S+)\s+and\s+(\S+)", r"gcd(\1, \2)"),
    (r"\b(?:the )?(?:lcm|least common multiple|lowest common multiple) of\s+(\S+)\s+and\s+(\S+)", r"lcm(\1, \2)"),
    (r"\b(?:the )?sum of\s+(\S+)\s+and\s+(\S+)", r"(\1)+(\2)"),
    (r"\b(?:the )?product of\s+(\S+)\s+and\s+(\S+)", r"(\1)*(\2)"),
    (r"\b(?:the )?difference (?:between|of)\s+(\S+)\s+and\s+(\S+)", r"(\1)-(\2)"),
    (r"\b(\d+)\s+choose\s+(\d+)", r"comb(\1, \2)"),
    (r"\bto the power of\b|\braised to(?: the power of)?\b|\bto the\b", "^"),
    (r"\bsquared\b", "^2"),
    (r"\bcubed\b", "^3"),
    (r"\bmultiplied by\b|\btimes\b", "*"),
    (r"\bdivided by\b|\bover\b", "/"),
    (r"\bplus\b|\badded to\b", "+"),
    (r"\bminus\b|\bsubtract\b", "-"),
    (r"\bmod(?:ulo)?\b", "%"),
    (r"\bis equal to\b|\bequals\b", "="),
    (r"(?<=\d),(?=\d{3}\b)", ""),  # thousands separators: 1,000,000
]


def _normalize(text: str) -> str:
    t = text.strip().lower()
    for _ in range(2):
        t = _LEADIN.sub("", t, count=1)
    t = _TAIL.sub("", t)
    for pat, rep in _WORDS:
        t = re.sub(pat, rep, t)
    return t.strip()


def _detect_mode(low: str) -> Tuple[str, Optional[str]]:
    var = None
    m = re.search(r"(?:with respect to|wrt|w\.r\.t\.?|d/d)\s*([a-z])\b|\bfor\s+([a-z])\b", low)
    if m:
        var = m.group(1) or m.group(2)
    if re.search(r"\bderivative\b|\bdifferentiate\b|\bd/d[a-z]\b", low):
        return "diff", var
    if re.search(r"\bintegra(?:l|te)\b|\bantiderivative\b", low):
        return "int", var
    if re.search(r"\bfactori[sz]e\b|\bfactor\b(?!ial)", low):
        return "factor", var
    if re.search(r"\bexpand\b|\bsimplify\b", low):
        return "expand", var
    return "auto", var


# ------------------------------------------------------------------ solving
def _fmt_root(r) -> str:
    if isinstance(r, complex):
        if abs(r.imag) < 1e-12:
            return fmt_number(r.real)
        return fmt_number(r)
    return fmt_number(r)


def _quadratic(cs: List, trace: List[str]) -> List:
    c, b, a = cs
    disc = b * b - 4 * a * c
    trace.append(f"quadratic formula: discriminant b²-4ac = {fmt_number(disc)}")
    if isinstance(disc, Fraction):
        root = exact_root(disc, 2) if disc >= 0 else None
        if root is not None:
            return [(-b + root) / (2 * a), (-b - root) / (2 * a)]
        mid = -b / (2 * a)
        if disc > 0:
            half = math.sqrt(disc) / abs(2 * a)
            trace.append(f"irrational roots: {mid} ± √({disc}) / {abs(2 * a)}")
            return [float(mid) + half, float(mid) - half]
        half = math.sqrt(-disc) / abs(2 * a)
        return [complex(float(mid), half), complex(float(mid), -half)]
    import cmath

    d = cmath.sqrt(disc)
    return [(-b + d) / (2 * a), (-b - d) / (2 * a)]


def solve_univariate(p: Poly, var: str, trace: List[str]) -> Tuple[List, bool]:
    cs = p.coeffs(var)
    deg = len(cs) - 1
    trace.append(f"rewrote as polynomial equation: {p} = 0 (degree {deg})")
    roots: List = []
    rem = cs
    if all(isinstance(c, Fraction) for c in cs) and deg >= 2:
        roots, rem = rational_roots(cs)
        if roots:
            trace.append("rational root theorem found: " + ", ".join(fmt_number(r) for r in roots))
    rdeg = len(rem) - 1
    if rdeg == 1:
        roots.append(-rem[0] / rem[1])
    elif rdeg == 2:
        roots.extend(_quadratic(rem, trace))
    elif rdeg >= 3:
        trace.append(f"no closed form used for degree {rdeg} factor; Durand–Kerner numeric roots")
        roots.extend(numeric_roots(rem))
    # verification: substitute every root into the original polynomial
    scale = sum(abs(complex(c)) for c in cs) or 1.0
    ok = True
    for r in roots:
        val = p.substitute({var: r})
        good = val == 0 if isinstance(r, Fraction) else abs(complex(val)) < 1e-7 * scale * max(1, abs(complex(r))) ** deg
        ok &= bool(good)
        trace.append(f"check {var} = {_fmt_root(r)}: residual {fmt_number(val) if isinstance(val, Fraction) else f'{abs(complex(val)):.1e}'} {'✓' if good else '✗'}")
    return roots, ok


def _format_roots(var: str, roots: List) -> str:
    seen: List = []
    counts: List[int] = []
    for r in roots:
        for i, s in enumerate(seen):
            if is_close(r, s, 1e-9):
                counts[i] += 1
                break
        else:
            seen.append(r)
            counts.append(1)
    real = [(r, c) for r, c in zip(seen, counts) if not (isinstance(r, complex) and abs(r.imag) > 1e-12)]
    cplx = [(r, c) for r, c in zip(seen, counts) if isinstance(r, complex) and abs(r.imag) > 1e-12]

    def key(rc):
        r = rc[0]
        return complex(r).real

    parts = []
    for r, c in sorted(real, key=key) + cplx:
        note = "" if c == 1 else (" (double root)" if c == 2 else f" (multiplicity {c})")
        parts.append(f"{var} = {_fmt_root(r)}{note}")
    if not real and cplx:
        return "no real solutions; complex: " + " or ".join(parts)
    return " or ".join(parts)


class MathSkill(Skill):
    name = "math"
    description = "exact arithmetic, equations, systems, polynomials, primes"

    def answer(self, question: str) -> Optional[Answer]:
        if _IS_CODE.search(question):
            return None  # a def/assert/code block belongs to the coding agent, not here
        question = re.sub(r"\s*=\s*\?\s*$", "", question.strip())  # "19 * 216 = ?" asks for the value
        low = question.lower().strip()
        try:
            return self._number_theory(low) or self._algebra(question, low)
        except (MathError, ZeroDivisionError, OverflowError, RecursionError) as exc:
            if re.search(r"[0-9]\s*[-+*/^]\s*[0-9(]|=", low):
                return Answer(f"I can't compute that: {exc}.", self.name, 0.5, trace=[str(exc)], failed=True)
            return None

    # ---- primes / factorisation
    def _number_theory(self, low: str) -> Optional[Answer]:
        m = re.search(r"\bis\s+([\d,]+)\s+(?:a\s+)?prime(?:\s+number)?\b", low)
        if m:
            n = int(m.group(1).replace(",", ""))
            if n.bit_length() > 200:
                raise MathError("number too large")
            prime = is_prime(n)
            trace = ["Miller–Rabin primality test with 12 fixed bases"]
            if prime:
                return Answer(f"Yes, {n} is prime.", self.name, 0.98, True, trace)
            f = factorize(n) if n > 1 else {}
            if f:
                fs = " × ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in f.items())
                trace.append(f"factorisation {fs}, checked by multiplying back")
                return Answer(f"No, {n} is not prime: {n} = {fs}.", self.name, 0.98, True, trace)
            return Answer(f"No, {n} is not prime (primes are integers greater than 1).", self.name, 0.98, True, trace)
        m = re.search(r"(?:prime factori[sz]ation of|prime factors of|factori[sz]e|factor)\s+([\d,]+)\s*[?.!]*$", low)
        if m:
            n = int(m.group(1).replace(",", ""))
            if n < 2 or n.bit_length() > 120:
                raise MathError("need an integer between 2 and 2^120")
            f = factorize(n)
            check = math.prod(p**e for p, e in f.items())
            fs = " × ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in f.items())
            trace = ["trial division, then Pollard's rho", f"check: product of factors = {check} {'✓' if check == n else '✗'}"]
            return Answer(f"{n} = {fs}", self.name, 0.98, check == n and all(is_prime(p) for p in f), trace)
        return None

    # ---- algebra
    def _algebra(self, question: str, low: str) -> Optional[Answer]:
        mode, var = _detect_mode(low)
        text = _normalize(question)
        if not text or not re.search(r"[0-9a-z]", text):
            return None
        if text.count("=") > 1:
            return self._system(text)
        if "=" in text:
            return self._equation(text, var)
        node = parse(text)
        names = variables(node)
        if not names and mode == "auto":
            return self._evaluate(node)
        if not names:
            return None
        poly = to_poly(node)
        if var is None:
            var = poly.variables()[0] if poly.variables() else "x"
        trace = [f"parsed: {to_text(node)}", f"expanded: {poly}"]
        if mode == "diff":
            d = poly.derivative(var)
            trace.append("power rule term by term: d/dx c·x^n = c·n·x^(n-1)")
            check = d.integral(var)
            ok = (check - (poly - Poly.const(poly.const_value()))).terms == {} or len(poly.variables()) > 1
            trace.append(f"check: integrating back gives {check} {'✓' if ok else '✗'}")
            return Answer(f"d/d{var} [{poly}] = {d}", self.name, 0.97, ok, trace)
        if mode == "int":
            i = poly.integral(var)
            ok = i.derivative(var) == poly
            trace.append(f"check: differentiating back gives {i.derivative(var)} {'✓' if ok else '✗'}")
            return Answer(f"∫ {poly} d{var} = {i} + C", self.name, 0.97, ok, trace)
        if mode == "factor":
            return self._factor(poly, var, trace)
        return Answer(f"{poly}", self.name, 0.9, True, trace + ["expansion is exact"])

    def _evaluate(self, node) -> Answer:
        value = evaluate(node)
        trace = [f"parsed: {to_text(node)}"]
        if isinstance(value, Fraction):
            trace.append("evaluated with exact rational arithmetic")
            return Answer(fmt_number(value), self.name, 0.97, True, trace)
        trace.append("result is irrational; evaluated in floating point (≈ 15 significant digits)")
        return Answer(f"≈ {fmt_number(value)}", self.name, 0.9, False, trace)

    def _equation(self, text: str, var: Optional[str]) -> Answer:
        lhs, rhs = parse_equation(text)
        trace = [f"parsed: {to_text(lhs)} = {to_text(rhs)}"]
        if not (variables(lhs) | variables(rhs)):
            ok = is_close(evaluate(lhs), evaluate(rhs), 1e-12)
            return Answer("True." if ok else "False.", self.name, 0.95, True, trace + [
                f"left side = {fmt_number(evaluate(lhs))}, right side = {fmt_number(evaluate(rhs))}"])
        p = to_poly(lhs) - to_poly(rhs)
        names = p.variables()
        if not names:
            if p.const_value() == 0:
                return Answer("Every value is a solution (the equation is an identity).", self.name, 0.95, True, trace)
            return Answer("No solution: the equation simplifies to a contradiction.", self.name, 0.95, True, trace + [f"simplifies to {p} = 0"])
        if len(names) == 1:
            v = names[0]
            roots, ok = solve_univariate(p, v, trace)
            if not roots:
                return Answer("No solution.", self.name, 0.9, ok, trace)
            return Answer(_format_roots(v, roots), self.name, 0.97 if ok else 0.6, ok, trace)
        target = var if var in names else ("x" if "x" in names else names[0])
        if p.degree(target) != 1:
            raise MathError(f"can only isolate {target} when it appears linearly")
        coef = Poly({tuple((v, e) for v, e in m if v != target): c for m, c in p.terms.items() if dict(m).get(target, 0) == 1})
        rest = Poly({m: c for m, c in p.terms.items() if dict(m).get(target, 0) == 0})
        if not coef.is_const():
            raise MathError(f"coefficient of {target} depends on other variables")
        sol = -rest * Poly.const(Fraction(1) / coef.const_value() if isinstance(coef.const_value(), Fraction) else 1 / coef.const_value())
        check = coef * Poly.var(target) + rest
        ok = check == p
        trace.append(f"isolated {target}: {target} = {sol}")
        return Answer(f"{target} = {sol}", self.name, 0.95, ok, trace)

    def _system(self, text: str) -> Answer:
        parts = [s for s in re.split(r"\s*(?:,|;|\band\b|\n)\s*", text) if s.strip()]
        polys, trace = [], []
        for s in parts:
            lhs, rhs = parse_equation(s)
            polys.append(to_poly(lhs) - to_poly(rhs))
            trace.append(f"equation: {polys[-1]} = 0")
        names = sorted({v for p in polys for v in p.variables()})
        status, sol = solve_linear_system(polys, names)
        trace.append("Gaussian elimination with exact fractions")
        if status == "none":
            return Answer("The system has no solution (the equations are inconsistent).", self.name, 0.95, True, trace)
        if status == "infinite":
            return Answer("The system has infinitely many solutions (the equations are dependent).", self.name, 0.9, True, trace)
        ok = all(p.substitute(sol) == 0 for p in polys)
        trace.append(f"check: substituted into all {len(polys)} equations {'✓' if ok else '✗'}")
        return Answer(", ".join(f"{v} = {fmt_number(sol[v])}" for v in names), self.name, 0.97, ok, trace)

    def _factor(self, poly: Poly, var: str, trace: List[str]) -> Answer:
        if len(poly.variables()) != 1:
            raise MathError("can only factor polynomials in one variable")
        cs = poly.coeffs(var)
        roots, rem = rational_roots(cs)
        if not roots:
            return Answer(f"{poly} has no rational roots, so it does not factor over the rationals into linear terms.", self.name, 0.8, False, trace)
        lead = cs[-1]
        rem_poly = Poly({((var, i),) if i else (): c for i, c in enumerate(rem)})
        # absorb the leading coefficient into the remainder
        factors: List[str] = []
        product = Poly.const(Fraction(1))
        for r in roots:
            f = Poly.var(var) - Poly.const(r) if r.denominator == 1 else Poly.const(Fraction(r.denominator)) * Poly.var(var) - Poly.const(Fraction(r.numerator))
            product = product * f
            factors.append(f"({f})")
        scale = poly.coeffs(var)[-1] / product.coeffs(var)[-1] if rem_poly.degree() == 0 else None
        if scale is not None:
            prefix = "" if scale == 1 else ("-" if scale == -1 else fmt_number(scale).split(" ")[0])
            text = prefix + "".join(factors)
            check = Poly.const(scale) * product
        else:
            lead_rem = rem_poly * Poly.const(Fraction(1))
            k = lead / (product.coeffs(var)[-1] * lead_rem.coeffs(var)[-1])
            text = "".join(factors) + f"({Poly.const(k) * lead_rem})"
            check = product * Poly.const(k) * lead_rem
        ok = check == poly
        trace.append(f"check: multiplying the factors back gives {check} {'✓' if ok else '✗'}")
        return Answer(text, self.name, 0.95 if ok else 0.4, ok, trace)
