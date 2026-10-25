"""A small exact computer-algebra core, written from scratch.

* :func:`parse` turns infix text (``2x^2 - 3(x + 1) = 5``) into a tiny AST.
* :func:`evaluate` evaluates an AST exactly with :class:`fractions.Fraction`
  where possible and falls back to floats for irrational results.
* :class:`Poly` is a multivariate polynomial with exact coefficients, used to
  expand, differentiate, integrate and solve equations.

There is no use of ``eval``: everything is parsed and interpreted by hand,
with guards against huge intermediate numbers.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple, Union

Number = Union[Fraction, float, complex]
Node = tuple


class MathError(ValueError):
    """Raised for anything we refuse or fail to compute."""


# ---------------------------------------------------------------- tokenizer
FUNCTIONS = {
    "sqrt", "cbrt", "abs", "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "ln", "log", "log2", "log10", "exp", "floor",
    "ceil", "round", "factorial", "gcd", "lcm", "min", "max", "comb", "perm",
}
CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

_TOKEN = re.compile(
    r"\s*(?:(?P<num>\d+(?:\.\d+)?|\.\d+)|(?P<id>[A-Za-z_][A-Za-z_0-9]*)|(?P<op>\*\*|[-+*/^(),=!%]))"
)
_UNICODE = {"×": "*", "·": "*", "÷": "/", "−": "-", "–": "-", "²": "^2", "³": "^3", "√": "sqrt", "π": "pi"}


def tokenize(text: str) -> List[Tuple[str, str]]:
    for k, v in _UNICODE.items():
        text = text.replace(k, v)
    pos, out = 0, []
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise MathError(f"unexpected character {text[pos]!r}")
        pos = m.end()
        if m.group("num") is not None:
            out.append(("num", m.group("num")))
        elif m.group("id") is not None:
            out.append(("id", m.group("id")))
        else:
            op = m.group("op")
            out.append(("op", "^" if op == "**" else op))
        while pos < len(text) and text[pos].isspace():
            pos += 1
    return _implicit_mul(out)


def _implicit_mul(tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Insert ``*`` for ``2x``, ``3(x+1)``, ``(a)(b)`` and ``x y``."""
    out: List[Tuple[str, str]] = []
    for tok in tokens:
        if out:
            prev = out[-1]
            left = prev[0] == "num" or (prev[0] == "id" and prev[1] not in FUNCTIONS) or prev[1] in (")", "!")
            right = tok[0] in ("num", "id") or tok[1] == "("
            if left and right and not (prev[0] == "num" and tok[0] == "num"):
                out.append(("op", "*"))
        out.append(tok)
    return out


# ------------------------------------------------------------------- parser
class _Parser:
    def __init__(self, tokens: List[Tuple[str, str]]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> Optional[Tuple[str, str]]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, value: Optional[str] = None) -> Tuple[str, str]:
        tok = self.peek()
        if tok is None or (value is not None and tok[1] != value):
            raise MathError(f"expected {value or 'token'}, got {tok[1] if tok else 'end of input'}")
        self.i += 1
        return tok

    def at(self, *values: str) -> bool:
        tok = self.peek()
        return tok is not None and tok[0] == "op" and tok[1] in values

    def expr(self) -> Node:
        node = self.term()
        while self.at("+", "-"):
            op = self.take()[1]
            node = ("bin", op, node, self.term())
        return node

    def term(self) -> Node:
        node = self.unary()
        while self.at("*", "/", "%"):
            op = self.take()[1]
            node = ("bin", op, node, self.unary())
        return node

    def unary(self) -> Node:
        if self.at("-"):
            self.take()
            return ("neg", self.unary())
        if self.at("+"):
            self.take()
            return self.unary()
        return self.power()

    def power(self) -> Node:
        base = self.postfix()
        if self.at("^"):
            self.take()
            return ("bin", "^", base, self.unary())
        return base

    def postfix(self) -> Node:
        node = self.atom()
        while self.at("!"):
            self.take()
            node = ("call", "factorial", [node])
        return node

    def atom(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise MathError("unexpected end of input")
        kind, val = tok
        if kind == "num":
            self.take()
            return ("num", Fraction(val))
        if kind == "id":
            self.take()
            if val in FUNCTIONS:
                if self.at("("):
                    self.take("(")
                    args = [self.expr()]
                    while self.at(","):
                        self.take()
                        args.append(self.expr())
                    self.take(")")
                else:  # "sqrt 16", "sin x"
                    args = [self.power()]
                return ("call", val, args)
            if val in CONSTANTS:
                return ("const", val)
            if len(val) == 1:
                return ("var", val)
            raise MathError(f"unknown name {val!r}")
        if val == "(":
            self.take()
            node = self.expr()
            self.take(")")
            return node
        raise MathError(f"unexpected {val!r}")


def parse(text: str) -> Node:
    tokens = tokenize(text)
    if not tokens:
        raise MathError("empty expression")
    p = _Parser(tokens)
    node = p.expr()
    if p.peek() is not None:
        raise MathError(f"unexpected {p.peek()[1]!r}")
    return node


def parse_equation(text: str) -> Tuple[Node, Node]:
    if text.count("=") != 1:
        raise MathError("expected exactly one '='")
    lhs, rhs = text.split("=")
    return parse(lhs), parse(rhs)


def variables(node: Node) -> set:
    kind = node[0]
    if kind == "var":
        return {node[1]}
    if kind in ("num", "const"):
        return set()
    if kind == "neg":
        return variables(node[1])
    if kind == "bin":
        return variables(node[2]) | variables(node[3])
    if kind == "call":
        out = set()
        for a in node[2]:
            out |= variables(a)
        return out
    raise MathError(f"bad node {kind}")


def to_text(node: Node) -> str:
    """Render an AST back to fully parenthesised infix (used in traces)."""
    kind = node[0]
    if kind == "num":
        v = node[1]
        return str(v.numerator) if v.denominator == 1 else f"({v})"
    if kind in ("var", "const"):
        return node[1]
    if kind == "neg":
        return f"-{to_text(node[1])}"
    if kind == "bin":
        return f"({to_text(node[2])} {node[1]} {to_text(node[3])})"
    if kind == "call":
        return f"{node[1]}({', '.join(to_text(a) for a in node[2])})"
    return "?"


# --------------------------------------------------------------- evaluation
MAX_BITS = 100_000


def _check_size(v: Number) -> Number:
    if isinstance(v, Fraction):
        if v.numerator.bit_length() > MAX_BITS or v.denominator.bit_length() > MAX_BITS:
            raise MathError("number too large")
    elif isinstance(v, float) and (v != v):
        raise MathError("undefined result")
    return v


def exact_root(x: Fraction, n: int) -> Optional[Fraction]:
    """Exact n-th root of a non-negative rational, or None if irrational."""
    if x < 0:
        if n % 2 == 1:
            r = exact_root(-x, n)
            return -r if r is not None else None
        return None
    num = _iroot(x.numerator, n)
    den = _iroot(x.denominator, n)
    if num is not None and den is not None:
        return Fraction(num, den)
    return None


def _iroot(v: int, n: int) -> Optional[int]:
    if v < 2:
        return v
    r = int(round(v ** (1.0 / n))) if v.bit_length() < 1000 else _newton_root(v, n)
    for c in (r - 1, r, r + 1):
        if c >= 0 and c ** n == v:
            return c
    return None


def _newton_root(v: int, n: int) -> int:
    x = 1 << ((v.bit_length() + n - 1) // n)
    while True:
        y = ((n - 1) * x + v // x ** (n - 1)) // n
        if y >= x:
            return x
        x = y


def _as_float(v: Number) -> float:
    if isinstance(v, complex):
        raise MathError("complex value")
    return float(v)


def _as_int(v: Number, what: str) -> int:
    if isinstance(v, Fraction) and v.denominator == 1:
        return v.numerator
    if isinstance(v, float) and v.is_integer():
        return int(v)
    raise MathError(f"{what} needs an integer")


def _power(a: Number, b: Number) -> Number:
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        if b.denominator == 1:
            e = b.numerator
            if a == 0 and e < 0:
                raise MathError("division by zero")
            bits = max(a.numerator.bit_length(), a.denominator.bit_length())
            if bits * abs(e) > MAX_BITS:
                raise MathError("number too large")
            return a ** e
        root = exact_root(a, b.denominator)
        if root is not None:
            return _power(root, Fraction(b.numerator))
        if a < 0:
            raise MathError("even root of a negative number")
    fa, fb = _as_float(a), _as_float(b)
    if fa < 0 and not fb.is_integer():
        raise MathError("fractional power of a negative number")
    try:
        return fa ** fb
    except OverflowError:
        raise MathError("number too large") from None


def _call(name: str, args: List[Number]) -> Number:
    def one() -> Number:
        if len(args) != 1:
            raise MathError(f"{name} takes one argument")
        return args[0]

    if name == "sqrt":
        x = one()
        if isinstance(x, Fraction):
            r = exact_root(x, 2)
            if r is not None:
                return r
            if x < 0:
                return complex(0, math.sqrt(-x))
        return math.sqrt(_as_float(x))
    if name == "cbrt":
        x = one()
        if isinstance(x, Fraction):
            r = exact_root(x, 3)
            if r is not None:
                return r
        f = _as_float(x)
        return math.copysign(abs(f) ** (1 / 3), f)
    if name == "abs":
        return abs(one())
    if name in ("floor", "ceil"):
        x = one()
        if isinstance(x, Fraction):
            return Fraction(math.floor(x) if name == "floor" else math.ceil(x))
        return Fraction(math.floor(_as_float(x)) if name == "floor" else math.ceil(_as_float(x)))
    if name == "round":
        if len(args) == 2:
            nd = _as_int(args[1], "round digits")
            return Fraction(round(Fraction(args[0]) if isinstance(args[0], Fraction) else _as_float(args[0]), nd))
        x = one()
        return Fraction(round(x if isinstance(x, Fraction) else _as_float(x)))
    if name == "factorial":
        n = _as_int(one(), "factorial")
        if n < 0:
            raise MathError("factorial of a negative number")
        if n > 5000:
            raise MathError("number too large")
        return Fraction(math.factorial(n))
    if name in ("gcd", "lcm"):
        if len(args) < 2:
            raise MathError(f"{name} takes at least two arguments")
        ints = [_as_int(a, name) for a in args]
        return Fraction(math.gcd(*ints) if name == "gcd" else math.lcm(*ints))
    if name in ("min", "max"):
        if not args:
            raise MathError(f"{name} needs arguments")
        return (min if name == "min" else max)(args, key=_as_float)
    if name in ("comb", "perm"):
        if len(args) != 2:
            raise MathError(f"{name} takes two arguments")
        n, k = _as_int(args[0], name), _as_int(args[1], name)
        if n > 100_000 or n < 0 or k < 0:
            raise MathError("argument out of range")
        return Fraction(math.comb(n, k) if name == "comb" else math.perm(n, k))
    if name == "log" and len(args) == 2:
        x, b = _as_float(args[0]), _as_float(args[1])
        return _exactish(math.log(x) / math.log(b))
    x = _as_float(one())
    try:
        if name == "ln":
            return math.log(x)
        if name in ("log", "log10"):
            return _exactish(math.log10(x))
        if name == "log2":
            return _exactish(math.log2(x))
        if name == "exp":
            return math.exp(x)
        return getattr(math, name)(x)
    except (ValueError, OverflowError) as exc:
        raise MathError(f"{name}: {exc}") from None


def _exactish(x: float) -> Number:
    """Snap floats like 2.0000000000000004 from logs back to integers."""
    r = round(x)
    if abs(x - r) < 1e-12:
        return Fraction(r)
    return x


def evaluate(node: Node, env: Optional[Dict[str, Number]] = None) -> Number:
    env = env or {}
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "const":
        return CONSTANTS[node[1]]
    if kind == "var":
        if node[1] not in env:
            raise MathError(f"unknown variable {node[1]}")
        return env[node[1]]
    if kind == "neg":
        return -evaluate(node[1], env)
    if kind == "call":
        return _check_size(_call(node[1], [evaluate(a, env) for a in node[2]]))
    if kind == "bin":
        op = node[1]
        a, b = evaluate(node[2], env), evaluate(node[3], env)
        if op == "+":
            r = a + b
        elif op == "-":
            r = a - b
        elif op == "*":
            r = a * b
        elif op == "/":
            if b == 0:
                raise MathError("division by zero")
            r = a / b
        elif op == "%":
            if b == 0:
                raise MathError("division by zero")
            r = a % b
        elif op == "^":
            r = _power(a, b)
        else:
            raise MathError(f"unknown operator {op}")
        return _check_size(r)
    raise MathError(f"bad node {kind}")


# --------------------------------------------------------------- polynomials
Monomial = Tuple[Tuple[str, int], ...]


class NotPolynomial(MathError):
    pass


class Poly:
    """Multivariate polynomial ``{monomial: coefficient}``.

    A monomial is a sorted tuple of ``(variable, exponent)`` pairs; ``()`` is
    the constant term. Coefficients are Fractions, or floats when the input
    contained irrational constants.
    """

    __slots__ = ("terms",)
    MAX_DEGREE = 24

    def __init__(self, terms: Optional[Dict[Monomial, Number]] = None):
        self.terms: Dict[Monomial, Number] = {m: c for m, c in (terms or {}).items() if c != 0}

    @classmethod
    def const(cls, c: Number) -> "Poly":
        return cls({(): c})

    @classmethod
    def var(cls, name: str) -> "Poly":
        return cls({((name, 1),): Fraction(1)})

    def __add__(self, other: "Poly") -> "Poly":
        out = dict(self.terms)
        for m, c in other.terms.items():
            out[m] = out.get(m, 0) + c
        return Poly(out)

    def __neg__(self) -> "Poly":
        return Poly({m: -c for m, c in self.terms.items()})

    def __sub__(self, other: "Poly") -> "Poly":
        return self + (-other)

    def __mul__(self, other: "Poly") -> "Poly":
        out: Dict[Monomial, Number] = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                powers = dict(m1)
                for v, e in m2:
                    powers[v] = powers.get(v, 0) + e
                m = tuple(sorted(powers.items()))
                out[m] = out.get(m, 0) + c1 * c2
        p = Poly(out)
        if p.degree() > self.MAX_DEGREE:
            raise MathError("polynomial degree too high")
        return p

    def __eq__(self, other) -> bool:
        return isinstance(other, Poly) and self.terms == other.terms

    def is_const(self) -> bool:
        return all(m == () for m in self.terms)

    def const_value(self) -> Number:
        return self.terms.get((), Fraction(0))

    def variables(self) -> List[str]:
        return sorted({v for m in self.terms for v, _ in m})

    def degree(self, var: Optional[str] = None) -> int:
        if not self.terms:
            return 0
        if var is None:
            return max(sum(e for _, e in m) for m in self.terms)
        return max(dict(m).get(var, 0) for m in self.terms)

    def coeffs(self, var: str) -> List[Number]:
        """Univariate coefficient list ``[c0, c1, ..., cn]`` for ``var``."""
        if any(v != var for m in self.terms for v, _ in m):
            raise NotPolynomial("polynomial has other variables")
        n = self.degree(var)
        out: List[Number] = [Fraction(0)] * (n + 1)
        for m, c in self.terms.items():
            out[dict(m).get(var, 0)] = c
        return out

    def derivative(self, var: str) -> "Poly":
        out: Dict[Monomial, Number] = {}
        for m, c in self.terms.items():
            powers = dict(m)
            e = powers.get(var, 0)
            if e == 0:
                continue
            if e == 1:
                del powers[var]
            else:
                powers[var] = e - 1
            key = tuple(sorted(powers.items()))
            out[key] = out.get(key, 0) + c * e
        return Poly(out)

    def integral(self, var: str) -> "Poly":
        out: Dict[Monomial, Number] = {}
        for m, c in self.terms.items():
            powers = dict(m)
            e = powers.get(var, 0) + 1
            powers[var] = e
            key = tuple(sorted(powers.items()))
            out[key] = out.get(key, 0) + (c / e if isinstance(c, Fraction) else c / e)
        return Poly(out)

    def substitute(self, env: Dict[str, Number]) -> Number:
        total: Number = Fraction(0)
        for m, c in self.terms.items():
            term: Number = c
            for v, e in m:
                term = term * env[v] ** e
            total = total + term
        return total

    def __str__(self) -> str:
        if not self.terms:
            return "0"

        def key(item):
            m, _ = item
            return (-sum(e for _, e in m), m)

        parts: List[str] = []
        for i, (m, c) in enumerate(sorted(self.terms.items(), key=key)):
            mono = "*".join(v if e == 1 else f"{v}^{e}" for v, e in m)
            neg = (c.real if isinstance(c, complex) else c) < 0
            mag = -c if neg else c
            if mono:
                body = ("" if mag == 1 else _coef_str(mag)) + mono
            else:
                body = _coef_str(mag)
            if i == 0:
                parts.append(("-" if neg else "") + body)
            else:
                parts.append((" - " if neg else " + ") + body)
        return "".join(parts)


def _coef_str(c: Number) -> str:
    if isinstance(c, Fraction):
        return str(c.numerator) if c.denominator == 1 else f"({c})"
    from mnx.agent.base import fmt_number

    return fmt_number(c)


def to_poly(node: Node) -> Poly:
    kind = node[0]
    if kind == "num":
        return Poly.const(node[1])
    if kind == "const":
        return Poly.const(CONSTANTS[node[1]])
    if kind == "var":
        return Poly.var(node[1])
    if kind == "neg":
        return -to_poly(node[1])
    if kind == "call":
        if variables(node):
            raise NotPolynomial(f"{node[1]}() of a variable")
        return Poly.const(evaluate(node))
    if kind == "bin":
        op = node[1]
        if op in "+-*":
            a, b = to_poly(node[2]), to_poly(node[3])
            return a + b if op == "+" else a - b if op == "-" else a * b
        if op == "/":
            a, b = to_poly(node[2]), to_poly(node[3])
            if not b.is_const():
                raise NotPolynomial("division by a variable expression")
            d = b.const_value()
            if d == 0:
                raise MathError("division by zero")
            return a * Poly.const(1 / d if not isinstance(d, Fraction) else Fraction(1) / d)
        if op == "^":
            base = to_poly(node[2])
            if variables(node[3]):
                raise NotPolynomial("variable exponent")
            e = evaluate(node[3])
            if base.is_const():
                return Poly.const(_power(base.const_value(), e))
            if not (isinstance(e, Fraction) and e.denominator == 1 and 0 <= e <= Poly.MAX_DEGREE):
                raise NotPolynomial("exponent must be a small non-negative integer")
            out = Poly.const(Fraction(1))
            for _ in range(int(e)):
                out = out * base
            return out
        raise NotPolynomial(f"operator {op}")
    raise NotPolynomial(kind)


# ------------------------------------------------------------ root finding
def _divisors(n: int, limit: int = 10**6) -> List[int]:
    n = abs(n)
    if n == 0:
        return [0]
    small, large = [], []
    i = 1
    while i * i <= n:
        if i > limit:
            break
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    return small + large[::-1]


def _poly_eval(coeffs: Sequence[Number], x: Number) -> Number:
    acc: Number = Fraction(0) if isinstance(x, Fraction) else 0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def _deflate(coeffs: List[Number], root: Number) -> List[Number]:
    """Divide ``coeffs`` (low → high) by ``(x - root)``."""
    n = len(coeffs) - 1
    out: List[Number] = [0] * n
    carry: Number = coeffs[n]
    for i in range(n - 1, -1, -1):
        out[i] = carry
        carry = coeffs[i] + carry * root
    return out


def rational_roots(coeffs: List[Fraction]) -> Tuple[List[Fraction], List[Fraction]]:
    """Find all rational roots (with multiplicity); returns ``(roots, remainder)``."""
    if not all(isinstance(c, Fraction) for c in coeffs):
        return [], coeffs
    roots: List[Fraction] = []
    cs = list(coeffs)
    while len(cs) > 1 and cs[0] == 0:
        roots.append(Fraction(0))
        cs = cs[1:]
    if len(cs) <= 1:
        return roots, cs
    lcm = 1
    for c in cs:
        lcm = lcm * c.denominator // math.gcd(lcm, c.denominator)
    ints = [int(c * lcm) for c in cs]
    if abs(ints[0]).bit_length() > 60 or abs(ints[-1]).bit_length() > 60:
        return roots, cs
    candidates = sorted(
        {Fraction(s * p, q) for p in _divisors(ints[0]) for q in _divisors(ints[-1]) if q for s in (1, -1)},
        key=lambda r: (abs(r), r),
    )
    for r in candidates:
        while len(cs) > 1 and _poly_eval(cs, r) == 0:
            roots.append(r)
            cs = _deflate(cs, r)
    return roots, cs


def numeric_roots(coeffs: Sequence[Number], iters: int = 500) -> List[complex]:
    """Durand–Kerner: all complex roots of a univariate polynomial."""
    cs = [complex(c) for c in coeffs]
    n = len(cs) - 1
    lead = cs[-1]
    cs = [c / lead for c in cs]
    radius = 1 + max(abs(c) for c in cs[:-1])
    roots = [radius * complex(0.4, 0.9) ** k for k in range(n)]
    for _ in range(iters):
        new = []
        delta = 0.0
        for i, r in enumerate(roots):
            denom = 1 + 0j
            for j, s in enumerate(roots):
                if i != j:
                    denom *= r - s
            if denom == 0:
                denom = 1e-12
            nr = r - _poly_eval(cs, r) / denom
            delta = max(delta, abs(nr - r))
            new.append(nr)
        roots = new
        if delta < 1e-14:
            break
    return [complex(round(r.real, 12), round(r.imag, 12)) for r in roots]


def solve_linear_system(polys: List[Poly], names: List[str]) -> Tuple[str, Dict[str, Number]]:
    """Gaussian elimination with exact fractions.

    Returns ``("unique", solution)``, ``("none", {})`` or ``("infinite", {})``.
    """
    rows: List[List[Number]] = []
    for p in polys:
        if p.degree() > 1:
            raise NotPolynomial("system is not linear")
        row = [Fraction(0)] * (len(names) + 1)
        for m, c in p.terms.items():
            if m == ():
                row[-1] = -c
            else:
                row[names.index(m[0][0])] = c
        rows.append(row)
    n = len(names)
    r = 0
    pivots = []
    for col in range(n):
        piv = next((i for i in range(r, len(rows)) if rows[i][col] != 0), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        pv = rows[r][col]
        rows[r] = [x / pv for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][col] != 0:
                f = rows[i][col]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        pivots.append(col)
        r += 1
    for row in rows[r:]:
        if row[-1] != 0 and all(x == 0 for x in row[:-1]):
            return "none", {}
    if r < n:
        return "infinite", {}
    return "unique", {names[c]: rows[i][-1] for i, c in enumerate(pivots)}


def is_close(a: Number, b: Number, tol: float = 1e-9) -> bool:
    return abs(complex(a) - complex(b)) <= tol * max(1.0, abs(complex(b)))
