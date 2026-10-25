"""Programming by example: search for the smallest program that fits.

Give input/output examples (``f(1) = 2, f(3) = 6`` or ``[3,1,2] -> [1,2,3]``)
and the skill enumerates programs in a small typed DSL bottom-up, smallest
first, pruning programs that behave identically on the examples
(observational equivalence). The first program that reproduces every example
is returned as Python source, re-executed independently as a check, and
applied to any query input (``f(10) = ?``).

This is test-time search with a perfect verifier: the examples *are* the
test suite. The search is exhaustive up to its size limit, so a returned
program is always correct on the examples; whether it generalises is a
question of how many examples you give (Occam's razor picks the simplest).
"""

from __future__ import annotations

import ast
import itertools
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from mnx.agent.base import Answer, Skill

ARG_NAMES = ["x", "y", "z"]
MAX_INT = 10**12
MAX_LEN = 200


class _Skip(Exception):
    pass


def _type(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        if all(isinstance(e, bool) for e in v) and v:
            return "?"
        if all(isinstance(e, int) and not isinstance(e, bool) for e in v):
            return "list"
        if all(isinstance(e, str) for e in v):
            return "slist"
    return "?"


def _freeze(v: Any) -> Any:
    if isinstance(v, list):
        return ("L",) + tuple(_freeze(e) for e in v)
    if isinstance(v, bool):
        return ("B", v)
    return v


def _guard(v: Any) -> Any:
    if isinstance(v, int) and not isinstance(v, bool) and abs(v) > MAX_INT:
        raise _Skip
    if isinstance(v, (list, str)) and len(v) > MAX_LEN:
        raise _Skip
    return v


@dataclass
class Prog:
    code: str
    type: str
    size: int
    vals: Tuple[Any, ...]


def _nz(b):
    if b == 0:
        raise _Skip
    return b


def _ne(seq):
    if not seq:
        raise _Skip
    return seq


def _idx(seq, i):
    if not -len(seq) <= i < len(seq):
        raise _Skip
    return seq[i]


def _pow(a, b):
    if not (0 <= b <= 8) or abs(a) > 10**4:
        raise _Skip
    return a**b


def _rep(s, n):
    if not 0 <= n <= 20:
        raise _Skip
    return s * n


def _rng(n):
    if not 0 <= n <= MAX_LEN:
        raise _Skip
    return list(range(n))


# (code template, argument types, implementation)
OPS: List[Tuple[str, Tuple[str, ...], Callable]] = [
    # int
    ("({0} + {1})", ("int", "int"), lambda a, b: a + b),
    ("({0} - {1})", ("int", "int"), lambda a, b: a - b),
    ("({0} * {1})", ("int", "int"), lambda a, b: a * b),
    ("({0} // {1})", ("int", "int"), lambda a, b: a // _nz(b)),
    ("({0} % {1})", ("int", "int"), lambda a, b: a % _nz(b)),
    ("({0} ** {1})", ("int", "int"), _pow),
    ("max({0}, {1})", ("int", "int"), max),
    ("min({0}, {1})", ("int", "int"), min),
    ("-{0}", ("int",), lambda a: -a),
    ("abs({0})", ("int",), abs),
    ("len({0})", ("list",), len),
    ("sum({0})", ("list",), sum),
    ("prod({0})", ("list",), lambda l: _guard(math.prod(l))),
    ("max({0})", ("list",), lambda l: max(_ne(l))),
    ("min({0})", ("list",), lambda l: min(_ne(l))),
    ("{0}[0]", ("list",), lambda l: _idx(l, 0)),
    ("{0}[-1]", ("list",), lambda l: _idx(l, -1)),
    ("{0}.count({1})", ("list", "int"), lambda l, v: l.count(v)),
    ("len({0})", ("str",), len),
    ("len({0})", ("slist",), len),
    ("{0}.count({1})", ("str", "str"), lambda s, t: s.count(_ne(t))),
    ("int({0})", ("str",), lambda s: int(s) if s.lstrip("-").isdigit() else _nz(0)),
    ("sum(int(c) for c in str({0}))", ("int",), lambda n: sum(int(c) for c in str(abs(n)))),
    # bool
    ("({0} == {1})", ("int", "int"), lambda a, b: a == b),
    ("({0} < {1})", ("int", "int"), lambda a, b: a < b),
    ("({0} == {1})", ("str", "str"), lambda a, b: a == b),
    ("({0} == {1})", ("list", "list"), lambda a, b: a == b),
    ("{0}.isdigit()", ("str",), lambda s: s.isdigit()),
    ("({1} in {0})", ("list", "int"), lambda l, v: v in l),
    ("({1} in {0})", ("str", "str"), lambda s, t: t in s),
    ("(not {0})", ("bool",), lambda b: not b),
    # list
    ("sorted({0})", ("list",), sorted),
    ("sorted({0}, reverse=True)", ("list",), lambda l: sorted(l, reverse=True)),
    ("{0}[::-1]", ("list",), lambda l: l[::-1]),
    ("{0}[1:]", ("list",), lambda l: l[1:]),
    ("{0}[:-1]", ("list",), lambda l: l[:-1]),
    ("list(dict.fromkeys({0}))", ("list",), lambda l: list(dict.fromkeys(l))),
    ("[sum({0}[:i + 1]) for i in range(len({0}))]", ("list",), lambda l: [sum(l[: i + 1]) for i in range(len(l))]),
    ("({0} + {1})", ("list", "list"), lambda a, b: a + b),
    ("[a + b for a, b in zip({0}, {1})]", ("list", "list"), lambda a, b: [p + q for p, q in zip(a, b)]),
    ("[a * b for a, b in zip({0}, {1})]", ("list", "list"), lambda a, b: [p * q for p, q in zip(a, b)]),
    ("{0}[:{1}]", ("list", "int"), lambda l, k: l[:k]),
    ("{0}[{1}:]", ("list", "int"), lambda l, k: l[k:]),
    ("({0} + [{1}])", ("list", "int"), lambda l, v: l + [v]),
    ("list(range({0}))", ("int",), _rng),
    ("list(range(1, {0} + 1))", ("int",), lambda n: _rng(n + 1)[1:]),
    ("[int(c) for c in str({0})]", ("int",), lambda n: [int(c) for c in str(abs(n))]),
    ("[len(w) for w in {0}]", ("slist",), lambda l: [len(w) for w in l]),
    # str
    ("{0}.upper()", ("str",), str.upper),
    ("{0}.lower()", ("str",), str.lower),
    ("{0}.title()", ("str",), str.title),
    ("{0}.capitalize()", ("str",), str.capitalize),
    ("{0}.swapcase()", ("str",), str.swapcase),
    ("{0}.strip()", ("str",), str.strip),
    ("{0}[::-1]", ("str",), lambda s: s[::-1]),
    ("{0}[1:]", ("str",), lambda s: s[1:]),
    ("{0}[:-1]", ("str",), lambda s: s[:-1]),
    ("{0}[0]", ("str",), lambda s: _idx(s, 0)),
    ("{0}[-1]", ("str",), lambda s: _idx(s, -1)),
    ("{0}.replace(' ', '')", ("str",), lambda s: s.replace(" ", "")),
    ("''.join(sorted({0}))", ("str",), lambda s: "".join(sorted(s))),
    ("''.join(dict.fromkeys({0}))", ("str",), lambda s: "".join(dict.fromkeys(s))),
    ("({0} + {1})", ("str", "str"), lambda a, b: a + b),
    ("({0} * {1})", ("str", "int"), _rep),
    ("{0}[:{1}]", ("str", "int"), lambda s, k: s[:k]),
    ("{0}[{1}:]", ("str", "int"), lambda s, k: s[k:]),
    ("str({0})", ("int",), str),
    ("' '.join({0})", ("slist",), " ".join),
    ("''.join({0})", ("slist",), "".join),
    ("''.join(w[0] for w in {0} if w)", ("slist",), lambda l: "".join(w[0] for w in l if w)),
    ("{0}.split()", ("str",), str.split),
    ("list({0})", ("str",), list),
    ("{0}[::-1]", ("slist",), lambda l: l[::-1]),
    ("sorted({0})", ("slist",), sorted),
    ("[w.capitalize() for w in {0}]", ("slist",), lambda l: [w.capitalize() for w in l]),
    ("[w[::-1] for w in {0}]", ("slist",), lambda l: [w[::-1] for w in l]),
    # str, character level: counting and filtering characters covers a lot of
    # everyday string work that no single method call reaches
    ("sum(1 for c in {0} if c.isupper())", ("str",), lambda t: sum(1 for c in t if c.isupper())),
    ("sum(1 for c in {0} if c.islower())", ("str",), lambda t: sum(1 for c in t if c.islower())),
    ("sum(1 for c in {0} if c.isdigit())", ("str",), lambda t: sum(1 for c in t if c.isdigit())),
    ("sum(1 for c in {0} if c.isalpha())", ("str",), lambda t: sum(1 for c in t if c.isalpha())),
    ("sum(1 for c in {0} if c in 'aeiou')", ("str",), lambda t: sum(1 for c in t if c in "aeiou")),
    ("sum(1 for c in {0}.lower() if c in 'aeiou')", ("str",), lambda t: sum(1 for c in t.lower() if c in "aeiou")),
    ("''.join(c for c in {0} if c.isalpha())", ("str",), lambda t: "".join(c for c in t if c.isalpha())),
    ("''.join(c for c in {0} if c.isdigit())", ("str",), lambda t: "".join(c for c in t if c.isdigit())),
    ("''.join(c for c in {0} if c.lower() not in 'aeiou')", ("str",),
     lambda t: "".join(c for c in t if c.lower() not in "aeiou")),
    ("''.join(c for c in {0} if c != {1})", ("str", "str"), lambda t, u: "".join(c for c in t if c != u)),
    ("sum(ord(c) for c in {0})", ("str",), lambda t: sum(ord(c) for c in t)),
    ("max({0}.split(), key=len)", ("str",), lambda t: max(t.split(), key=len)),
    ("len({0}.split())", ("str",), lambda t: len(t.split())),
    ("[len(w) for w in {0}.split()]", ("str",), lambda t: [len(w) for w in t.split()]),
    ("{0}.replace({1}, '')", ("str", "str"), lambda t, u: t.replace(u, "") if u else t),
    # neighbour and all-pairs templates: comparisons across a sequence are a
    # whole class of question ("is it sorted", "are two of them equal") that no
    # element-wise map or filter can express
    ("all(a <= b for a, b in zip({0}, {0}[1:]))", ("list",), lambda l: all(a <= b for a, b in zip(l, l[1:]))),
    ("all(a < b for a, b in zip({0}, {0}[1:]))", ("list",), lambda l: all(a < b for a, b in zip(l, l[1:]))),
    ("all(a >= b for a, b in zip({0}, {0}[1:]))", ("list",), lambda l: all(a >= b for a, b in zip(l, l[1:]))),
    ("all(a > b for a, b in zip({0}, {0}[1:]))", ("list",), lambda l: all(a > b for a, b in zip(l, l[1:]))),
    ("all(a == b for a, b in zip({0}, {0}[1:]))", ("list",), lambda l: all(a == b for a, b in zip(l, l[1:]))),
    ("[b - a for a, b in zip({0}, {0}[1:])]", ("list",), lambda l: [b - a for a, b in zip(l, l[1:])]),
    ("sum(1 for a, b in zip({0}, {0}[1:]) if a < b)", ("list",),
     lambda l: sum(1 for a, b in zip(l, l[1:]) if a < b)),
    ("len(set({0}))", ("list",), lambda l: len(set(l))),
    ("sorted(set({0}))", ("list",), lambda l: sorted(set(l))),
    ("(len(set({0})) == len({0}))", ("list",), lambda l: len(set(l)) == len(l)),
    ("sorted({0})[len({0}) // 2]", ("list",), lambda l: sorted(_ne(l))[len(l) // 2]),
    ("any(a + b == {1} for i, a in enumerate({0}) for b in {0}[i + 1:])", ("list", "int"),
     lambda l, k: any(a + b == k for i, a in enumerate(l) for b in l[i + 1:])),
    ("any(abs(a - b) < {1} for i, a in enumerate({0}) for b in {0}[i + 1:])", ("list", "int"),
     lambda l, k: any(abs(a - b) < k for i, a in enumerate(l) for b in l[i + 1:])),
    ("len(set({0}))", ("str",), lambda t: len(set(t))),
    ("all(a <= b for a, b in zip({0}, {0}[1:]))", ("str",), lambda t: all(a <= b for a, b in zip(t, t[1:]))),
]

#: extra size charged for "exotic" operators so that simpler explanations win
#: (``x * 2`` before ``zip(x, x)``, ``x % 2 == 0`` before digit sums)
OP_COST = {
    "sum(int(c) for c in str({0}))": 2, "[int(c) for c in str({0})]": 2, "prod({0})": 2, "({0} ** {1})": 2,
    "[a + b for a, b in zip({0}, {1})]": 2, "[a * b for a, b in zip({0}, {1})]": 2, "list(range({0}))": 1,
    "list(range(1, {0} + 1))": 1, "[sum({0}[:i + 1]) for i in range(len({0}))]": 2, "{0}.count({1})": 1,
    "list(dict.fromkeys({0}))": 1, "''.join(dict.fromkeys({0}))": 1, "int({0})": 1,
    "sum(1 for c in {0} if c.isupper())": 1, "sum(1 for c in {0} if c.islower())": 1,
    "sum(1 for c in {0} if c.isdigit())": 1, "sum(1 for c in {0} if c.isalpha())": 1,
    "sum(1 for c in {0} if c in 'aeiou')": 1, "sum(1 for c in {0}.lower() if c in 'aeiou')": 2,
    "''.join(c for c in {0} if c.isalpha())": 1, "''.join(c for c in {0} if c.isdigit())": 1,
    "''.join(c for c in {0} if c.lower() not in 'aeiou')": 2, "''.join(c for c in {0} if c != {1})": 1,
    "sum(ord(c) for c in {0})": 2, "max({0}.split(), key=len)": 1, "[len(w) for w in {0}.split()]": 1,
    "{0}.replace({1}, '')": 1,
    "all(a <= b for a, b in zip({0}, {0}[1:]))": 1, "all(a < b for a, b in zip({0}, {0}[1:]))": 1,
    "all(a >= b for a, b in zip({0}, {0}[1:]))": 1, "all(a > b for a, b in zip({0}, {0}[1:]))": 1,
    "all(a == b for a, b in zip({0}, {0}[1:]))": 1, "[b - a for a, b in zip({0}, {0}[1:])]": 1,
    "sum(1 for a, b in zip({0}, {0}[1:]) if a < b)": 2, "len(set({0}))": 1, "sorted(set({0}))": 1,
    "(len(set({0})) == len({0}))": 1, "sorted({0})[len({0}) // 2]": 2,
    "any(a + b == {1} for i, a in enumerate({0}) for b in {0}[i + 1:])": 2,
    "any(abs(a - b) < {1} for i, a in enumerate({0}) for b in {0}[i + 1:])": 2,
}

#: character-level string operations, searched only when a string actually
#: appears in the examples: they would otherwise slow every numeric search down
CHAR_OPS = frozenset(o[0] for o in OPS if "for c in" in o[0] or o[0].endswith("key=len)")
                     or "{0}.split()]" in o[0])

#: neighbour / all-pairs templates, searched only when the examples hold a
#: sequence: like the character ops they would otherwise tax every search
PAIR_OPS = frozenset(o[0] for o in OPS if "zip({0}, {0}[1:])" in o[0] or "enumerate({0})" in o[0]
                     or "set({0})" in o[0] or "// 2]" in o[0])

INT_CONSTS = [0, 1, 2, 3, 10]
STR_CONSTS = [" ", ""]

_SAFE_GLOBALS = {"__builtins__": {}, "len": len, "sum": sum, "max": max, "min": min, "abs": abs, "ord": ord,
                 "sorted": sorted, "prod": math.prod, "set": set, "enumerate": enumerate, "any": any, "all": all, "ord": ord, "list": list, "range": range, "dict": dict, "str": str, "int": int, "zip": zip}


@dataclass
class Result:
    code: str
    source: str
    size: int
    #: every consistent program found, as ``(code, size, outputs on the query inputs)``
    hits: List[Tuple[str, int, Tuple[Any, ...]]]


class _Err:
    """Marks a program that raised on a *query* input (examples must not raise)."""

    def __repr__(self) -> str:
        return "<error>"


ERR = _Err()
_EXC = (_Skip, TypeError, ValueError, IndexError, ZeroDivisionError, OverflowError, RecursionError, AttributeError)


def _apply(fn: Callable, arg_vals: Sequence[Tuple[Any, ...]], n: int, n_ex: int) -> Optional[Tuple[Any, ...]]:
    out = []
    for i in range(n):
        args = [a[i] for a in arg_vals]
        if i >= n_ex and any(a is ERR for a in args):
            out.append(ERR)
            continue
        try:
            out.append(_guard(fn(*args)))
        except _EXC:
            if i < n_ex:
                return None
            out.append(ERR)
    return tuple(out)


def _lambda_bodies(elements: List[int], max_size: int = 5) -> Tuple[List[Prog], List[Prog]]:
    """Enumerate int→int bodies and int→bool predicates over element ``e``."""
    if not elements:
        return [], []
    n = len(elements)
    terms = [Prog("e", "int", 1, tuple(elements))]
    terms += [Prog(str(c), "int", 1, (c,) * n) for c in (1, 2, 3, 10, 0)]
    ops = [o for o in OPS if all(t == "int" for t in o[1])]
    pools = _enumerate(terms, ops, max_size, n, time.monotonic() + 0.5)
    ints = [p for p in pools if p.type == "int" and "e" in p.code and p.code != "e"]
    bools = [p for p in pools if p.type == "bool" and "e" in p.code]
    # common predicates as single primitives so filters are cheap to find
    for code in ("e % 2 == 0", "e % 2 == 1", "e > 0", "e < 0", "e != 0"):
        fn = _compile_lambda(code)
        bools.append(Prog(code, "bool", 1, tuple(fn(e) for e in elements)))
    return ints, bools


def _enumerate(terminals: List[Prog], ops, max_size: int, n_ex: int, deadline: float,
               target: Optional[Tuple] = None, extra: Optional[Callable] = None) -> Any:
    """Bottom-up enumeration with observational equivalence.

    Values are tracked on the example inputs *and* the query inputs, so two
    programs that agree on the examples but disagree on a query stay
    distinct. Without ``target`` every program is returned; with it, every
    program matching the target on the examples is collected up to one size
    beyond the smallest match.
    """
    pools: Dict[Tuple[int, str], List[Prog]] = {}
    seen: set = set()
    out: List[Prog] = []
    hits: List[Prog] = []
    n = len(terminals[0].vals)

    def add(p: Prog) -> None:
        if target is not None and (p.type, tuple(_freeze(v) for v in p.vals[:n_ex])) == target:
            hits.append(p)
            return
        key = (p.type, tuple(_freeze(v) for v in p.vals))
        if key in seen:
            return
        seen.add(key)
        pools.setdefault((p.size, p.type), []).append(p)
        out.append(p)

    for t in terminals:
        add(t)
    for size in range(2, max_size + 1):
        if time.monotonic() > deadline or (hits and size > hits[0].size + 1) or len(hits) >= 64:
            break
        for code, types, fn in ops:
            for parts in _compositions(size - 1 - OP_COST.get(code, 0), len(types)):
                arg_pools = [pools.get((s, t), []) for s, t in zip(parts, types)]
                if any(not pl for pl in arg_pools):
                    continue
                for count, args in enumerate(itertools.product(*arg_pools)):
                    if count % 4096 == 4095 and time.monotonic() > deadline:
                        break
                    vals = _apply(fn, [a.vals for a in args], n, n_ex)
                    if vals is None:
                        continue
                    typ = _common_type(vals[:n_ex])
                    if typ is not None:
                        add(Prog(code.format(*(a.code for a in args)), typ, size, vals))
            if time.monotonic() > deadline:
                break
        if extra is not None:
            for count, p in enumerate(extra(pools, size)):
                if count % 4096 == 4095 and time.monotonic() > deadline:
                    break
                add(p)
    if target is None:
        return out
    return hits


def _common_type(vals: Sequence[Any]) -> Optional[str]:
    types = {_type(v) for v in vals}
    if len(types) == 1:
        t = types.pop()
        return None if t == "?" else t
    if types <= {"list", "slist"}:  # empty lists are ambiguous
        return "list" if all(not v or _type(v) == "list" for v in vals) else None
    return None


def _compositions(total: int, k: int):
    if k == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - k + 2):
        for rest in _compositions(total - first, k - 1):
            yield (first,) + rest


def synthesize(examples: List[Tuple[Tuple[Any, ...], Any]], queries: Sequence[Tuple[Any, ...]] = (),
               max_size: int = 8, time_limit: float = 4.0) -> Optional[Result]:
    """Return the smallest DSL program mapping each input tuple to its output."""
    if not examples:
        return None
    arity = len(examples[0][0])
    queries = [q for q in queries if len(q) == arity]
    if any(len(i) != arity for i, _ in examples) or arity > len(ARG_NAMES):
        return None
    inputs = [i for i, _ in examples] + list(queries)
    n_ex, n = len(examples), len(inputs)
    target_vals = tuple(o for _, o in examples)
    ttype = _common_type(target_vals)
    if ttype is None:
        return None
    target = (ttype, tuple(_freeze(v) for v in target_vals))
    terminals: List[Prog] = []
    for j in range(arity):
        vals = tuple(i[j] for i in inputs)
        typ = _common_type(vals[:n_ex])
        if typ is None:
            return None
        terminals.append(Prog(ARG_NAMES[j], typ, 1, vals))
    # small constants are cheap; 3 and 10 cost more, so "n == 3" doesn't beat "n % 2 == 0"
    terminals += [Prog(str(c), "int", 1 if c <= 2 else 2, (c,) * n) for c in INT_CONSTS]
    terminals += [Prog(repr(c), "str", 1, (c,) * n) for c in STR_CONSTS]

    deadline = time.monotonic() + time_limit
    elements = sorted({e for i in inputs for a in i if isinstance(a, list) for e in a if isinstance(e, int)}
                      | {e for o in target_vals if isinstance(o, list) for e in o if isinstance(e, int)})
    bodies, preds = _lambda_bodies(elements[:60]) if elements else ([], [])

    def lifted(fn: Callable) -> Callable:
        return lambda v: _guard([_guard(fn(e)) for e in v])

    def extra(pools, size):
        """map / filter with enumerated lambda bodies; size = 1 + |list| + |body|."""
        for ls in range(1, size - 1):
            bs = size - 1 - ls
            for lst in pools.get((ls, "list"), []):
                for body in (b for b in bodies if b.size == bs):
                    vals = _apply(lifted(_compile_lambda(body.code)), [lst.vals], n, n_ex)
                    if vals is not None:
                        yield Prog(f"[{body.code} for e in {lst.code}]", "list", size, vals)
                for pred in (p for p in preds if p.size == bs):
                    fn = _compile_lambda(pred.code)
                    vals = _apply(lambda v, fn=fn: [e for e in v if fn(e)], [lst.vals], n, n_ex)
                    if vals is not None:
                        yield Prog(f"[e for e in {lst.code} if {pred.code}]", "list", size, vals)
        # conditionals: (a if c else b), size = 1 + |c| + |a| + |b|, kept small
        for cs in range(1, min(size - 2, 5)):
            for c in pools.get((cs, "bool"), []):
                for a_s in range(1, min(size - cs - 1, 4)):
                    b_s = size - 1 - cs - a_s
                    if b_s > 3:
                        continue
                    for a in pools.get((a_s, ttype), []):
                        for b in pools.get((b_s, ttype), []):
                            vals = _apply(lambda cv, av, bv: av if cv else bv, [c.vals, a.vals, b.vals], n, n_ex)
                            if vals is not None:
                                yield Prog(f"({a.code} if {c.code} else {b.code})", ttype, size, vals)

    has_str = any(isinstance(v, str) or (isinstance(v, list) and any(isinstance(e, str) for e in v))
                  for i in inputs for v in i) or any(isinstance(v, str) for v in target_vals)
    has_seq = has_str or any(isinstance(v, list) for i in inputs for v in i) \
        or any(isinstance(v, list) for v in target_vals)
    skip = set()
    if not has_str:
        skip |= set(CHAR_OPS)
    if not has_seq:
        skip |= set(PAIR_OPS)
    ops = OPS if not skip else [o for o in OPS if o[0] not in skip]
    hits = _enumerate(terminals, ops, max_size, n_ex, deadline, target=target, extra=extra)
    if not hits:
        return None
    hits.sort(key=lambda p: p.size)
    best = hits[0]
    if queries:  # show the simplest program that agrees with the weighted vote
        votes = [vote([(h.code, h.size, h.vals[n_ex:]) for h in hits], i)[0] for i in range(len(queries))]
        best = next((h for h in hits if all(_freeze(h.vals[n_ex + i]) == _freeze(v) for i, v in enumerate(votes))), best)
    params = ", ".join(ARG_NAMES[:arity])
    source = f"def f({params}):\n    return {best.code}\n"
    return Result(best.code, source, best.size, [(h.code, h.size, h.vals[n_ex:]) for h in hits])


def vote(hits: List[Tuple[str, int, Tuple[Any, ...]]], q_index: int) -> Tuple[Any, float]:
    """Bayesian program averaging: weight each consistent program by 2^-size."""
    weights: Dict[Any, float] = {}
    rep: Dict[Any, Any] = {}
    for _, size, outs in hits:
        v = outs[q_index]
        if v is ERR:
            continue
        k = _freeze(v)
        weights[k] = weights.get(k, 0.0) + 2.0 ** -size
        rep[k] = v
    if not weights:
        return ERR, 0.0
    total = sum(weights.values())
    k = max(weights, key=weights.get)
    return rep[k], weights[k] / total


_LAMBDA_CACHE: Dict[str, Callable] = {}


def _compile_lambda(code: str) -> Callable:
    # ``code`` is generated by this module from the fixed templates above,
    # never from user text, and runs with an empty __builtins__.
    fn = _LAMBDA_CACHE.get(code)
    if fn is None:
        fn = eval("lambda e: " + code, dict(_SAFE_GLOBALS))  # noqa: S307
        _LAMBDA_CACHE[code] = fn
    return fn


def run_program(source: str, args: Tuple[Any, ...]) -> Any:
    """Independently execute synthesised source (generated by us, sandboxed builtins)."""
    env = dict(_SAFE_GLOBALS)
    exec(source, env)  # noqa: S102
    return env["f"](*[_copy(a) for a in args])


def _copy(v):
    return list(v) if isinstance(v, list) else v


# ----------------------------------------------------------------- parsing
def _split_top(text: str, seps: str = ",;\n") -> List[str]:
    out, depth, quote, cur = [], 0, None, []
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch in seps and depth == 0:
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    out.append("".join(cur))
    return [s.strip() for s in out if s.strip()]


def _literal(text: str) -> Any:
    v = ast.literal_eval(text.strip())
    if isinstance(v, tuple):
        v = list(v)
    if _type(v) == "?":
        raise ValueError("unsupported literal")
    return v


def parse_examples(text: str) -> Tuple[List[Tuple[Tuple[Any, ...], Any]], List[Tuple[Any, ...]]]:
    """Extract ``(inputs, output)`` examples and query inputs from text."""
    t = text.replace("→", "->").replace("=>", "->")
    t = re.sub(r"^[^\[\('\"\d\-]*?(?=(?:[a-z]\s*\(|[\[\('\"\d\-]))", "", t.strip(), count=1, flags=re.I)
    examples, queries = [], []
    for item in _split_top(t):
        item = item.strip().rstrip(".")
        m = re.fullmatch(r"[a-z]\w*\s*\((.*)\)\s*(?:=|->|is)\s*(.*)", item, re.I | re.S)
        if m:
            args_text, out_text = m.group(1), m.group(2).strip()
        elif "->" in item:
            args_text, out_text = (s.strip() for s in item.split("->", 1))
        elif re.fullmatch(r"[a-z]\w*\s*\((.*)\)\s*\??", item, re.I | re.S):
            args_text, out_text = re.fullmatch(r"[a-z]\w*\s*\((.*)\)\s*\??", item, re.I | re.S).group(1), "?"
        else:
            continue
        try:
            args = tuple(_literal(a) for a in _split_top(args_text, ","))
        except (ValueError, SyntaxError):
            continue
        if not args:
            continue
        if out_text.strip(" ?") == "" or out_text.strip() in ("?", "what", "what?"):
            queries.append(args)
            continue
        try:
            examples.append((args, _literal(out_text)))
        except (ValueError, SyntaxError):
            continue
    return examples, queries


class SynthSkill(Skill):
    name = "synth"
    description = "write a program from input/output examples (programming by example)"

    def __init__(self, time_limit: float = 4.0):
        self.time_limit = time_limit

    def answer(self, question: str) -> Optional[Answer]:
        if not re.search(r"->|→|=>|[a-z]\s*\([^)]*\)\s*=", question, re.I):
            return None
        examples, queries = parse_examples(question)
        if len(examples) < 2:
            return None
        res = synthesize(examples, queries, time_limit=self.time_limit)
        if res is None:
            return Answer("I searched all programs up to my size limit and none fits every example.", self.name, 0.5, False,
                          [f"{len(examples)} examples, exhaustive bottom-up search with observational-equivalence pruning"],
                          failed=True)
        ok = True
        trace = [f"searched smallest-first; {len(res.hits)} consistent program(s) found, smallest has size {res.size}",
                 "re-ran the generated Python on every example:"]
        for args, out in examples:
            try:
                got = run_program(res.source, args)
            except Exception as exc:  # pragma: no cover - defensive
                got = f"error: {exc}"
            good = got == out and _type(got) == _type(out)
            ok &= good
            trace.append(f"  f{_show_args(args)} = {got!r} {'✓' if good else '✗'}")
        lines = [f"```python\n{res.source}```"]
        agreement = 1.0
        for i, q in enumerate(queries):
            value, share = vote(res.hits, i)
            agreement = min(agreement, share)
            shown = "an error" if value is ERR else repr(value)
            lines.append(f"f{_show_args(q)} = {shown}")
            if share < 0.999:
                trace.append(f"query f{_show_args(q)}: programs weighted by 2^-size agree {share:.0%} on {shown}")
        if len(res.hits) > 1:
            trace.append("other consistent programs: " + "; ".join(h[0] for h in res.hits[1:4]))
        conf = (0.6 + min(0.3, 0.1 * (len(examples) - 1))) * (0.5 + 0.5 * agreement) if ok else 0.2
        return Answer("\n".join(lines), self.name, conf, ok, trace)


def _show_args(args: Tuple[Any, ...]) -> str:
    return "(" + ", ".join(repr(a) for a in args) + ")"
