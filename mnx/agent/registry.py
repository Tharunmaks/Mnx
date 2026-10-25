"""The agent registry: thousands of small, named, self-checking capabilities.

MNX1's specialists (coder, mathematician, reasoner, researcher) are the agents
that reason. This registry is the other half of the picture Tharun drew: a very
large number of tiny agents, each of which does exactly one thing and knows how
to prove it did it.

Every registry agent is a Python function plus the asserts that show it works.
Building the registry *executes* each one: an agent whose asserts fail is never
written. So "the registry has N agents" means N functions that ran and passed
their own tests on this machine.

Two tiers, kept apart on purpose:

``verified``
    Recipes from ``data/code`` and generated parametrised families (convert to
    base 7, rotate a list by 5, 99th percentile, set bit 12, kilometres to
    miles, ...). Each one carries asserts that passed, so MNX1 may answer with
    its result.
``reference``
    Real standard-library functions with their docstrings, found by walking the
    installed stdlib. These are pointers, not answers: the registry knows
    ``statistics.median`` exists and what its docstring says, and says so,
    without claiming a tested result.

Use it::

    python -m mnx.agent.registry build --out data/registry
    python -m mnx.agent.registry verify --all
    python -m mnx.agent.registry search "convert celsius to fahrenheit"
    python -m mnx.agent.registry run unit.celsius_to_fahrenheit 100
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import inspect
import json
import pkgutil
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from mnx.agent.retrieval import BM25Index, Chunk
from mnx.agent.skills.code import Recipe, default_recipe_dir

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "data" / "registry"
MANIFEST = "agents.jsonl.gz"


@dataclass
class Agent:
    """One capability: a function, what it is for, and the asserts that passed."""

    id: str
    family: str
    name: str
    description: str
    tier: str
    code: str = ""
    tests: List[str] = field(default_factory=list)
    module: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.name} {self.description} {' '.join(self.tags)}"


# --------------------------------------------------------------- verification
def check(agent: Agent) -> Tuple[bool, str]:
    """Run an agent's code and its asserts, and say whether they passed.

    Recipe agents go through the coder's ``python -I`` sandbox, because a
    recipe's own test block is arbitrary code: some of them start a server or
    read input, and that does not belong in this process. The generated
    families are code this module wrote, so they run here.
    """
    if agent.tier != "verified":
        return True, "reference agent: not executed"
    script = agent.code + "\n" + "\n".join(agent.tests) + "\n"
    if agent.family == "recipe":
        from mnx.agent.skills.neural import run_tests

        ok, err = run_tests(agent.code, agent.tests, timeout=10)
        return ok, "ok" if ok else (err or "failed")
    try:
        exec(compile(script, f"<{agent.id}>", "exec"), {})
    except Exception as exc:  # noqa: BLE001 - any failure disqualifies the agent
        return False, f"{type(exc).__name__}: {exc}"
    return True, "ok"


def call(agent: Agent, *args: object) -> object:
    """Run a verified agent on arguments."""
    if agent.tier != "verified":
        raise ValueError(f"{agent.id} is a reference agent: import {agent.module} and call it yourself")
    ns: Dict[str, object] = {}
    exec(compile(agent.code, f"<{agent.id}>", "exec"), ns)
    fn = ns.get(agent.name)
    if not callable(fn):
        raise ValueError(f"{agent.id} defines no function called {agent.name}")
    return fn(*args)


# ------------------------------------------------------------------ families
def _mk(family: str, name: str, desc: str, code: str, tests: Sequence[str], tags: Sequence[str]) -> Agent:
    return Agent(id=f"{family}.{name}", family=family, name=name, description=desc, tier="verified",
                 code=code, tests=list(tests), tags=list(tags))


def recipe_agents() -> Iterator[Agent]:
    """Every tested recipe in data/code becomes an agent."""
    for path in sorted(default_recipe_dir().glob("*.py")):
        rec = Recipe.parse(path.read_text(encoding="utf-8"), str(path))
        # the whole test block, not only the assert lines: some recipes set a
        # fixture up first, and the asserts need it
        tests = [t for t in rec.tests if t.strip()]
        if not any(t.startswith("assert") for t in tests):
            continue
        yield _mk("recipe", rec.name, rec.description, rec.code, tests, rec.tags.split())


def base_agents(lo: int = 2, hi: int = 36) -> Iterator[Agent]:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    for b in range(lo, hi + 1):
        code = (f"def to_base_{b}(n):\n"
                f'    """Write a non-negative integer in base {b}."""\n'
                f"    if n == 0:\n        return '0'\n"
                f"    out = ''\n"
                f"    while n:\n"
                f"        out = '{digits}'[n % {b}] + out\n"
                f"        n //= {b}\n"
                f"    return out\n")
        want = []
        for n in (0, 1, b, b * b + 1, 255):
            s, m = "", n
            if m == 0:
                s = "0"
            while m:
                s, m = digits[m % b] + s, m // b
            want.append(f"assert to_base_{b}({n}) == {s!r}")
        yield _mk("base", f"to_base_{b}", f"Convert a whole number to base {b} as a string.", code, want,
                  ["base", "radix", "convert", "number", str(b)])
        code = (f"def from_base_{b}(s):\n"
                f'    """Read a base-{b} string as an integer."""\n'
                f"    return int(s, {b})\n")
        tests = [f"assert from_base_{b}({t.split('== ')[1]}) == {n}"
                 for n, t in zip((0, 1, b, b * b + 1, 255), want)]
        yield _mk("base", f"from_base_{b}", f"Parse a base-{b} string into a whole number.", code, tests,
                  ["base", "radix", "parse", "number", str(b)])


def multiple_agents(hi: int = 200) -> Iterator[Agent]:
    for k in range(2, hi + 1):
        yield _mk("number", f"is_multiple_of_{k}",
                  f"True when a whole number is a multiple of {k}.",
                  f"def is_multiple_of_{k}(n):\n"
                  f'    """True when n is divisible by {k}."""\n'
                  f"    return n % {k} == 0\n",
                  [f"assert is_multiple_of_{k}({k * 3}) is True",
                   f"assert is_multiple_of_{k}({k * 3 + 1}) is False",
                   f"assert is_multiple_of_{k}(0) is True"],
                  ["divisible", "multiple", "modulo", str(k)])
        yield _mk("number", f"remainder_mod_{k}",
                  f"The remainder of a whole number modulo {k}.",
                  f"def remainder_mod_{k}(n):\n"
                  f'    """n modulo {k}."""\n'
                  f"    return n % {k}\n",
                  [f"assert remainder_mod_{k}({k * 2 + 1}) == 1",
                   f"assert remainder_mod_{k}({k}) == 0"],
                  ["remainder", "modulo", "mod", str(k)])


def power_agents(hi: int = 50) -> Iterator[Agent]:
    for k in range(2, hi + 1):
        yield _mk("number", f"is_power_of_{k}",
                  f"True when a positive number is an exact power of {k}.",
                  f"def is_power_of_{k}(n):\n"
                  f'    """True when n is {k} raised to a whole power."""\n'
                  f"    if n < 1:\n        return False\n"
                  f"    while n % {k} == 0:\n        n //= {k}\n"
                  f"    return n == 1\n",
                  [f"assert is_power_of_{k}({k ** 3}) is True",
                   f"assert is_power_of_{k}({k ** 3 + 1}) is False",
                   f"assert is_power_of_{k}(1) is True"],
                  ["power", "exponent", str(k)])


def digit_agents(hi: int = 19) -> Iterator[Agent]:
    for i in range(hi + 1):
        yield _mk("digits", f"digit_{i}_from_right",
                  f"Digit number {i} counting from the right of a whole number (0 is the last digit).",
                  f"def digit_{i}_from_right(n):\n"
                  f'    """The digit {i} places from the right of abs(n), or 0 when the number is shorter."""\n'
                  f"    return abs(n) // 10 ** {i} % 10\n",
                  [f"assert digit_{i}_from_right({10 ** i * 7}) == 7",
                   f"assert digit_{i}_from_right(0) == 0"],
                  ["digit", "place", "position", str(i)])


def sum_power_agents(hi: int = 10) -> Iterator[Agent]:
    for p in range(2, hi + 1):
        yield _mk("math", f"sum_of_{p}th_powers",
                  f"The sum of the {p}th powers of the numbers in a list.",
                  f"def sum_of_{p}th_powers(xs):\n"
                  f'    """Add up each number raised to the power {p}."""\n'
                  f"    return sum(x ** {p} for x in xs)\n",
                  [f"assert sum_of_{p}th_powers([1, 2]) == {1 + 2 ** p}",
                   f"assert sum_of_{p}th_powers([]) == 0"],
                  ["sum", "power", "exponent", str(p)])


def caesar_agents(hi: int = 25) -> Iterator[Agent]:
    for k in range(1, hi + 1):
        code = (f"def caesar_shift_{k}(s):\n"
                f'    """Shift every letter {k} places along the alphabet, keeping case."""\n'
                f"    out = []\n"
                f"    for c in s:\n"
                f"        if c.isalpha():\n"
                f"            base = 65 if c.isupper() else 97\n"
                f"            out.append(chr((ord(c) - base + {k}) % 26 + base))\n"
                f"        else:\n"
                f"            out.append(c)\n"
                f"    return ''.join(out)\n")
        shifted = "".join(chr((ord(c) - 97 + k) % 26 + 97) for c in "abcxyz")
        yield _mk("cipher", f"caesar_shift_{k}", f"Caesar cipher with a shift of {k}.", code,
                  [f"assert caesar_shift_{k}('abcxyz') == {shifted!r}",
                   f"assert caesar_shift_{k}('a b!') == {chr((ord('a') - 97 + k) % 26 + 97) + ' ' + chr((ord('b') - 97 + k) % 26 + 97) + '!'!r}"],
                  ["caesar", "cipher", "shift", "encrypt", str(k)])


def list_agents(hi: int = 50) -> Iterator[Agent]:
    for k in range(1, hi + 1):
        yield _mk("list", f"top_{k}",
                  f"The {k} largest numbers in a list, largest first.",
                  f"def top_{k}(xs):\n"
                  f'    """The {k} largest values, largest first."""\n'
                  f"    return sorted(xs, reverse=True)[:{k}]\n",
                  [f"assert top_{k}(list(range(100))) == {sorted(range(100), reverse=True)[:k]!r}",
                   f"assert top_{k}([]) == []"],
                  ["top", "largest", "biggest", "sort", str(k)])
        yield _mk("list", f"bottom_{k}",
                  f"The {k} smallest numbers in a list, smallest first.",
                  f"def bottom_{k}(xs):\n"
                  f'    """The {k} smallest values, smallest first."""\n'
                  f"    return sorted(xs)[:{k}]\n",
                  [f"assert bottom_{k}(list(range(100))) == {list(range(100))[:k]!r}",
                   f"assert bottom_{k}([]) == []"],
                  ["bottom", "smallest", "least", "sort", str(k)])
    for k in range(2, hi + 1):
        yield _mk("list", f"chunk_{k}",
                  f"Split a list into consecutive pieces of {k} (the last piece may be shorter).",
                  f"def chunk_{k}(xs):\n"
                  f'    """Pieces of {k} items, in order."""\n'
                  f"    return [xs[i:i + {k}] for i in range(0, len(xs), {k})]\n",
                  [f"assert chunk_{k}(list(range({k * 2}))) == {[list(range(k * 2))[i:i + k] for i in range(0, k * 2, k)]!r}",
                   f"assert chunk_{k}([]) == []"],
                  ["chunk", "split", "batch", "group", str(k)])
        yield _mk("list", f"every_{k}th",
                  f"Every {k}th item of a list, starting with the first.",
                  f"def every_{k}th(xs):\n"
                  f'    """Items at positions 0, {k}, {2 * k}, ..."""\n'
                  f"    return xs[::{k}]\n",
                  [f"assert every_{k}th(list(range({k * 3}))) == {list(range(k * 3))[::k]!r}"],
                  ["every", "stride", "step", "skip", str(k)])
        yield _mk("list", f"rotate_by_{k}",
                  f"Rotate a list {k} places to the left.",
                  f"def rotate_by_{k}(xs):\n"
                  f'    """Move the first {k} items to the end."""\n'
                  f"    if not xs:\n        return []\n"
                  f"    j = {k} % len(xs)\n"
                  f"    return xs[j:] + xs[:j]\n",
                  [f"assert rotate_by_{k}(list(range(10))) == {list(range(10))[k % 10:] + list(range(10))[:k % 10]!r}",
                   f"assert rotate_by_{k}([]) == []"],
                  ["rotate", "shift", "cycle", str(k)])


def percentile_agents() -> Iterator[Agent]:
    for p in range(1, 100):
        code = (f"def percentile_{p}(xs):\n"
                f'    """The {p}th percentile by nearest rank."""\n'
                f"    if not xs:\n        raise ValueError('no data')\n"
                f"    s = sorted(xs)\n"
                f"    i = min(len(s) - 1, max(0, round({p} / 100 * len(s) + 0.5) - 1))\n"
                f"    return s[i]\n")
        data = list(range(1, 101))
        i = min(99, max(0, round(p / 100 * 100 + 0.5) - 1))
        yield _mk("stats", f"percentile_{p}", f"The {p}th percentile of a list of numbers.", code,
                  [f"assert percentile_{p}(list(range(1, 101))) == {data[i]}"],
                  ["percentile", "quantile", "statistics", str(p)])


def bit_agents(hi: int = 63) -> Iterator[Agent]:
    for i in range(hi + 1):
        for op, doc, body, expect in (
            ("set", "Turn bit {i} on.", "return n | (1 << {i})", lambda n, i: n | (1 << i)),
            ("clear", "Turn bit {i} off.", "return n & ~(1 << {i})", lambda n, i: n & ~(1 << i)),
            ("toggle", "Flip bit {i}.", "return n ^ (1 << {i})", lambda n, i: n ^ (1 << i)),
            ("test", "True when bit {i} is on.", "return bool(n >> {i} & 1)", lambda n, i: bool(n >> i & 1)),
        ):
            name = f"{op}_bit_{i}"
            got0, got1 = expect(0, i), expect((1 << i) | 1, i)
            yield _mk("bits", name, doc.format(i=i) + f" Bit {i} counts from the least significant end.",
                      f"def {name}(n):\n"
                      f'    """{doc.format(i=i)}"""\n'
                      f"    {body.format(i=i)}\n",
                      [f"assert {name}(0) == {got0!r}", f"assert {name}({(1 << i) | 1}) == {got1!r}"],
                      ["bit", "binary", "bitwise", op, str(i)])


def string_agents(hi: int = 80) -> Iterator[Agent]:
    for w in range(1, hi + 1):
        yield _mk("text", f"pad_to_{w}",
                  f"Pad a string with spaces on the right up to {w} characters.",
                  f"def pad_to_{w}(s):\n"
                  f'    """Right-pad with spaces to a width of {w}."""\n'
                  f"    return s.ljust({w})\n",
                  [f"assert pad_to_{w}('x') == {'x'.ljust(w)!r}",
                   f"assert len(pad_to_{w}('')) == {w}"],
                  ["pad", "width", "align", "format", str(w)])
        yield _mk("text", f"truncate_to_{w}",
                  f"Cut a string down to at most {w} characters.",
                  f"def truncate_to_{w}(s):\n"
                  f'    """The first {w} characters."""\n'
                  f"    return s[:{w}]\n",
                  [f"assert truncate_to_{w}('a' * {w + 10}) == {'a' * w!r}",
                   f"assert truncate_to_{w}('') == ''"],
                  ["truncate", "cut", "shorten", "limit", str(w)])


#: (name, aliases, how many of the base unit) per dimension. Conversions are
#: generated for every ordered pair inside a dimension.
UNITS: Dict[str, List[Tuple[str, float]]] = {
    "length": [("metre", 1.0), ("kilometre", 1000.0), ("centimetre", 0.01), ("millimetre", 0.001),
               ("micrometre", 1e-6), ("mile", 1609.344), ("yard", 0.9144), ("foot", 0.3048),
               ("inch", 0.0254), ("nautical_mile", 1852.0), ("furlong", 201.168), ("light_year", 9.4607304725808e15)],
    "mass": [("kilogram", 1.0), ("gram", 0.001), ("milligram", 1e-6), ("tonne", 1000.0),
             ("pound", 0.45359237), ("ounce", 0.028349523125), ("stone", 6.35029318), ("carat", 0.0002)],
    "time": [("second", 1.0), ("minute", 60.0), ("hour", 3600.0), ("day", 86400.0), ("week", 604800.0),
             ("millisecond", 0.001), ("microsecond", 1e-6), ("year", 31557600.0)],
    "volume": [("litre", 1.0), ("millilitre", 0.001), ("cubic_metre", 1000.0), ("gallon_us", 3.785411784),
               ("gallon_uk", 4.54609), ("pint_us", 0.473176473), ("cup_us", 0.2365882365),
               ("tablespoon_us", 0.0147867648), ("teaspoon_us", 0.00492892159)],
    "area": [("square_metre", 1.0), ("square_kilometre", 1e6), ("hectare", 10000.0), ("acre", 4046.8564224),
             ("square_foot", 0.09290304), ("square_mile", 2589988.110336), ("square_inch", 0.00064516)],
    "speed": [("metre_per_second", 1.0), ("kilometre_per_hour", 0.2777777777777778),
              ("mile_per_hour", 0.44704), ("knot", 0.5144444444444445), ("foot_per_second", 0.3048)],
    "data": [("byte", 1.0), ("kilobyte", 1000.0), ("megabyte", 1e6), ("gigabyte", 1e9), ("terabyte", 1e12),
             ("kibibyte", 1024.0), ("mebibyte", 1048576.0), ("gibibyte", 1073741824.0), ("bit", 0.125)],
    "energy": [("joule", 1.0), ("kilojoule", 1000.0), ("calorie", 4.184), ("kilocalorie", 4184.0),
               ("watt_hour", 3600.0), ("kilowatt_hour", 3.6e6), ("electronvolt", 1.602176634e-19)],
    "pressure": [("pascal", 1.0), ("kilopascal", 1000.0), ("bar", 100000.0), ("atmosphere", 101325.0),
                 ("psi", 6894.757293168361), ("torr", 133.32236842105263)],
    "angle": [("radian", 1.0), ("degree", 0.017453292519943295), ("gradian", 0.015707963267948967),
              ("turn", 6.283185307179586), ("arcminute", 0.0002908882086657216)],
}


def unit_agents() -> Iterator[Agent]:
    for dim, units in UNITS.items():
        for a, fa in units:
            for b, fb in units:
                if a == b:
                    continue
                factor = fa / fb
                name = f"{a}_to_{b}"
                code = (f"def {name}(x):\n"
                        f'    """Convert {a.replace("_", " ")} to {b.replace("_", " ")}."""\n'
                        f"    return x * {factor!r}\n")
                yield _mk("unit", name, f"Convert a {dim} from {a.replace('_', ' ')} to {b.replace('_', ' ')}.",
                          code, [f"assert abs({name}(1) - {factor!r}) < 1e-9 * max(1, abs({factor!r}))",
                                 f"assert {name}(0) == 0"],
                          ["convert", "conversion", "unit", dim, a, b])
    # temperature needs an offset, so it is written out rather than scaled
    temps = [
        ("celsius_to_fahrenheit", "Convert degrees Celsius to degrees Fahrenheit.", "x * 9 / 5 + 32", [(0, 32.0), (100, 212.0)]),
        ("fahrenheit_to_celsius", "Convert degrees Fahrenheit to degrees Celsius.", "(x - 32) * 5 / 9", [(32, 0.0), (212, 100.0)]),
        ("celsius_to_kelvin", "Convert degrees Celsius to kelvin.", "x + 273.15", [(0, 273.15), (-273.15, 0.0)]),
        ("kelvin_to_celsius", "Convert kelvin to degrees Celsius.", "x - 273.15", [(273.15, 0.0)]),
        ("fahrenheit_to_kelvin", "Convert degrees Fahrenheit to kelvin.", "(x - 32) * 5 / 9 + 273.15", [(32, 273.15)]),
        ("kelvin_to_fahrenheit", "Convert kelvin to degrees Fahrenheit.", "(x - 273.15) * 9 / 5 + 32", [(273.15, 32.0)]),
    ]
    for name, desc, expr, cases in temps:
        yield _mk("unit", name, desc,
                  f"def {name}(x):\n    \"\"\"{desc}\"\"\"\n    return {expr}\n",
                  [f"assert abs({name}({i}) - {o!r}) < 1e-9" for i, o in cases],
                  ["convert", "temperature", "unit"] + name.split("_to_"))


MATH_AGENTS: List[Tuple[str, str, str, List[str], List[str]]] = [
    ("factorial_of", "The factorial of a whole number.",
     "    out = 1\n    for i in range(2, n + 1):\n        out *= i\n    return out",
     ["assert factorial_of(0) == 1", "assert factorial_of(5) == 120", "assert factorial_of(10) == 3628800"],
     ["factorial", "product", "math"]),
    ("nth_fibonacci", "The nth Fibonacci number, counting from 0.",
     "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
     ["assert nth_fibonacci(0) == 0", "assert nth_fibonacci(10) == 55", "assert nth_fibonacci(30) == 832040"],
     ["fibonacci", "sequence", "math"]),
    ("divisors_of", "Every positive divisor of a whole number, in order.",
     "    return [d for d in range(1, abs(n) + 1) if n % d == 0]",
     ["assert divisors_of(12) == [1, 2, 3, 4, 6, 12]", "assert divisors_of(7) == [1, 7]"],
     ["divisors", "factors", "math"]),
    ("prime_factors_of", "The prime factors of a whole number, with repeats, smallest first.",
     "    out, d = [], 2\n    while d * d <= n:\n        while n % d == 0:\n            out.append(d)\n            n //= d\n        d += 1\n    if n > 1:\n        out.append(n)\n    return out",
     ["assert prime_factors_of(360) == [2, 2, 2, 3, 3, 5]", "assert prime_factors_of(97) == [97]"],
     ["prime", "factorise", "factors", "math"]),
    ("is_prime_number", "True when a whole number is prime.",
     "    if n < 2:\n        return False\n    if n % 2 == 0:\n        return n == 2\n    d = 3\n    while d * d <= n:\n        if n % d == 0:\n            return False\n        d += 2\n    return True",
     ["assert is_prime_number(1) is False", "assert is_prime_number(97) is True", "assert is_prime_number(91) is False"],
     ["prime", "primality", "math"]),
    ("euler_totient", "How many numbers below n share no factor with n.",
     "    out, m, d = n, n, 2\n    while d * d <= m:\n        if m % d == 0:\n            while m % d == 0:\n                m //= d\n            out -= out // d\n        d += 1\n    if m > 1:\n        out -= out // m\n    return out",
     ["assert euler_totient(1) == 1", "assert euler_totient(9) == 6", "assert euler_totient(10) == 4"],
     ["totient", "phi", "coprime", "math"]),
    ("digit_sum_of", "The sum of the digits of a whole number.",
     "    return sum(int(c) for c in str(abs(n)))",
     ["assert digit_sum_of(0) == 0", "assert digit_sum_of(987) == 24"],
     ["digit", "sum", "math"]),
    ("digit_count_of", "How many digits a whole number has.",
     "    return len(str(abs(n)))",
     ["assert digit_count_of(0) == 1", "assert digit_count_of(-1234) == 4"],
     ["digit", "length", "count", "math"]),
    ("reverse_digits", "A whole number with its digits reversed.",
     "    s = str(abs(n))[::-1]\n    return -int(s) if n < 0 else int(s)",
     ["assert reverse_digits(1230) == 321", "assert reverse_digits(-12) == -21"],
     ["reverse", "digit", "math"]),
    ("integer_square_root", "The largest whole number whose square is at most n.",
     "    if n < 0:\n        raise ValueError('negative')\n    r = 0\n    while (r + 1) * (r + 1) <= n:\n        r += 1\n    return r",
     ["assert integer_square_root(0) == 0", "assert integer_square_root(26) == 5", "assert integer_square_root(25) == 5"],
     ["sqrt", "root", "square", "math"]),
]

STAT_AGENTS: List[Tuple[str, str, str, List[str], List[str]]] = [
    ("mean_of", "The arithmetic mean of a list of numbers.",
     "    if not xs:\n        raise ValueError('no data')\n    return sum(xs) / len(xs)",
     ["assert mean_of([1, 2, 3]) == 2", "assert mean_of([5]) == 5"], ["mean", "average", "statistics"]),
    ("median_of", "The middle value of a list of numbers.",
     "    if not xs:\n        raise ValueError('no data')\n    s = sorted(xs)\n    m = len(s) // 2\n    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2",
     ["assert median_of([3, 1, 2]) == 2", "assert median_of([1, 2, 3, 4]) == 2.5"], ["median", "middle", "statistics"]),
    ("mode_of", "The most common value in a list; the smallest one when several tie.",
     "    if not xs:\n        raise ValueError('no data')\n    return min(set(xs), key=lambda v: (-xs.count(v), v))",
     ["assert mode_of([1, 2, 2, 3]) == 2", "assert mode_of([4, 4, 5, 5]) == 4"], ["mode", "common", "statistics"]),
    ("variance_of", "The population variance of a list of numbers.",
     "    if not xs:\n        raise ValueError('no data')\n    m = sum(xs) / len(xs)\n    return sum((x - m) ** 2 for x in xs) / len(xs)",
     ["assert variance_of([2, 2, 2]) == 0", "assert variance_of([1, 3]) == 1"], ["variance", "spread", "statistics"]),
    ("stdev_of", "The population standard deviation of a list of numbers.",
     "    if not xs:\n        raise ValueError('no data')\n    m = sum(xs) / len(xs)\n    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5",
     ["assert stdev_of([1, 3]) == 1", "assert stdev_of([5, 5]) == 0"], ["standard", "deviation", "statistics"]),
    ("range_of", "The difference between the largest and smallest value.",
     "    if not xs:\n        raise ValueError('no data')\n    return max(xs) - min(xs)",
     ["assert range_of([1, 9, 4]) == 8", "assert range_of([3]) == 0"], ["range", "spread", "statistics"]),
]


def written_agents() -> Iterator[Agent]:
    for family, table in (("math", MATH_AGENTS), ("stats", STAT_AGENTS)):
        for name, desc, body, tests, tags in table:
            arg = "n" if family == "math" else "xs"
            yield _mk(family, name, desc,
                      f"def {name}({arg}):\n    \"\"\"{desc}\"\"\"\n{body}\n", tests, tags)


def date_agents(hi: int = 60) -> Iterator[Agent]:
    import datetime as _dt

    base = _dt.date(2026, 1, 15)
    for k in range(1, hi + 1):
        for op, sign in (("after", 1), ("before", -1)):
            name = f"date_{k}_days_{op}"
            got = base + _dt.timedelta(days=sign * k)
            yield _mk("date", name, f"The date {k} days {op} a given ISO date.",
                      f"def {name}(iso):\n"
                      f'    """The ISO date {k} days {op} the one given."""\n'
                      f"    import datetime\n"
                      f"    d = datetime.date.fromisoformat(iso) + datetime.timedelta(days={sign * k})\n"
                      f"    return d.isoformat()\n",
                      [f"assert {name}('2026-01-15') == {got.isoformat()!r}"],
                      ["date", "days", op, "calendar", str(k)])
    for name, desc, body, tests in (
        ("weekday_of", "The day of the week of an ISO date.",
         "    import datetime\n    return datetime.date.fromisoformat(iso).strftime('%A')",
         ["assert weekday_of('2026-01-15') == 'Thursday'"]),
        ("days_in_month_of", "How many days the month of an ISO date has.",
         "    import calendar, datetime\n    d = datetime.date.fromisoformat(iso)\n    return calendar.monthrange(d.year, d.month)[1]",
         ["assert days_in_month_of('2026-02-01') == 28", "assert days_in_month_of('2024-02-01') == 29"]),
        ("is_leap_year_of", "True when the year of an ISO date is a leap year.",
         "    import calendar, datetime\n    return calendar.isleap(datetime.date.fromisoformat(iso).year)",
         ["assert is_leap_year_of('2024-03-01') is True", "assert is_leap_year_of('2026-03-01') is False"]),
        ("quarter_of", "Which quarter of the year an ISO date falls in.",
         "    import datetime\n    return (datetime.date.fromisoformat(iso).month - 1) // 3 + 1",
         ["assert quarter_of('2026-01-15') == 1", "assert quarter_of('2026-12-31') == 4"]),
        ("week_number_of", "The ISO week number of an ISO date.",
         "    import datetime\n    return datetime.date.fromisoformat(iso).isocalendar()[1]",
         ["assert week_number_of('2026-01-15') == 3"]),
        ("day_of_year_of", "Which day of the year an ISO date is.",
         "    import datetime\n    return datetime.date.fromisoformat(iso).timetuple().tm_yday",
         ["assert day_of_year_of('2026-01-15') == 15", "assert day_of_year_of('2026-12-31') == 365"]),
    ):
        yield _mk("date", name, desc,
                  f"def {name}(iso):\n    \"\"\"{desc}\"\"\"\n{body}\n", tests, ["date", "calendar"] + name.split("_"))


SKIP_MODULES = {"antigravity", "this", "idlelib", "turtledemo", "tkinter", "lib2to3", "test", "ensurepip",
                "venv", "distutils", "curses", "asyncio", "multiprocessing", "concurrent", "ctypes",
                "pydoc_data", "wsgiref", "http", "xmlrpc", "socketserver", "webbrowser", "site", "pty"}


def stdlib_agents(limit: int = 4000) -> Iterator[Agent]:
    """Real standard-library functions, as pointers with their own docstrings.

    These are the ``reference`` tier: the registry knows they exist and what
    they are for, so MNX1 can point at the right one. It does not claim a
    tested result from them, because nothing here ran them.
    """
    seen = set()
    count = 0
    for info in pkgutil.iter_modules():
        name = info.name
        if name.startswith("_") or name in SKIP_MODULES:
            continue
        try:
            mod = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - a module that will not import is not an agent
            continue
        if not (getattr(mod, "__file__", "") or "").startswith(str(Path(inspect.__file__).parent)):
            continue  # standard library only, not site-packages
        for fname, fn in sorted(vars(mod).items()):
            if fname.startswith("_") or not inspect.isfunction(fn) or fn.__module__ != name:
                continue
            doc = (inspect.getdoc(fn) or "").strip().split("\n\n")[0].replace("\n", " ").strip()
            if not doc or f"{name}.{fname}" in seen:
                continue
            seen.add(f"{name}.{fname}")
            try:
                sig = str(inspect.signature(fn))
            except (TypeError, ValueError):
                sig = "(...)"
            count += 1
            yield Agent(id=f"stdlib.{name}.{fname}", family="stdlib", name=fname,
                        description=f"{doc[:300]} Call it as {name}.{fname}{sig}.", tier="reference",
                        module=name, tags=["stdlib", "library", name, fname])
            if count >= limit:
                return


def families(scale: int = 1) -> Iterator[Agent]:
    """Every agent, at a parameter scale. ``scale`` widens the numeric families."""
    yield from recipe_agents()
    yield from written_agents()
    yield from unit_agents()
    yield from date_agents(60 * scale)
    yield from base_agents()
    yield from caesar_agents()
    yield from percentile_agents()
    yield from multiple_agents(200 * scale)
    yield from power_agents(50 * scale)
    yield from digit_agents(19 + scale)
    yield from sum_power_agents(10 + scale)
    yield from list_agents(50 * scale)
    yield from bit_agents(63 * scale)
    yield from string_agents(80 * scale)
    yield from stdlib_agents()


# -------------------------------------------------------------------- store
class Registry:
    """Loaded agents, searchable by what they are for."""

    def __init__(self, agents: Sequence[Agent]):
        self.agents = list(agents)
        self.by_id = {a.id: a for a in self.agents}
        self._index: Optional[BM25Index] = None

    def __len__(self) -> int:
        return len(self.agents)

    @property
    def index(self) -> BM25Index:
        if self._index is None:
            idx = BM25Index()
            for a in self.agents:
                idx.add(Chunk(a.text, a.id, a.family))
            self._index = idx
        return self._index

    def search(self, query: str, k: int = 5, tier: Optional[str] = None) -> List[Tuple[float, Agent]]:
        hits = self.index.search(query, k=k * 4 if tier else k)
        out = []
        for score, chunk in hits:
            agent = self.by_id.get(chunk.source)
            if agent is None or (tier and agent.tier != tier):
                continue
            out.append((score, agent))
            if len(out) >= k:
                break
        return out

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for a in self.agents:
            out[a.family] = out.get(a.family, 0) + 1
        return out

    # ------------------------------------------------------------ on disk
    def write(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / MANIFEST
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for a in self.agents:
                f.write(json.dumps(asdict(a)) + "\n")
        (out_dir / "counts.json").write_text(json.dumps(
            {"total": len(self.agents), "verified": sum(a.tier == "verified" for a in self.agents),
             "reference": sum(a.tier == "reference" for a in self.agents), "families": self.counts()}, indent=2))
        return path

    @classmethod
    def load(cls, out_dir: Path = DEFAULT_DIR) -> "Registry":
        path = Path(out_dir) / MANIFEST
        if not path.exists():
            raise FileNotFoundError(f"no registry at {path}: run `python -m mnx.agent.registry build`")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return cls([Agent(**json.loads(line)) for line in f if line.strip()])

    @classmethod
    def build(cls, scale: int = 1, verify: bool = True, on_progress: Optional[Callable[[int, int], None]] = None) -> Tuple["Registry", List[Tuple[str, str]]]:
        """Generate every agent, run each one's tests, and keep the ones that pass."""
        kept: List[Agent] = []
        rejected: List[Tuple[str, str]] = []
        for i, agent in enumerate(families(scale)):
            if verify:
                ok, why = check(agent)
                if not ok:
                    rejected.append((agent.id, why))
                    continue
            kept.append(agent)
            if on_progress and len(kept) % 500 == 0:
                on_progress(len(kept), len(rejected))
        return cls(kept), rejected


# ---------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="mnx-registry", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="generate, verify and write the registry")
    b.add_parser = None  # type: ignore[assignment]
    b.add_argument("--out", default=str(DEFAULT_DIR))
    b.add_argument("--scale", type=int, default=1, help="widen the numeric families by this factor")
    b.add_argument("--no-verify", action="store_true", help="skip running the tests (not recommended)")

    v = sub.add_parser("verify", help="re-run the tests of agents already on disk")
    v.add_argument("--dir", default=str(DEFAULT_DIR))
    v.add_argument("--sample", type=int, default=300, help="how many to check (0 = all)")

    s = sub.add_parser("search", help="find agents for a request")
    s.add_argument("query", nargs="+")
    s.add_argument("--dir", default=str(DEFAULT_DIR))
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--tier", default=None, choices=["verified", "reference"])

    r = sub.add_parser("run", help="run one verified agent")
    r.add_argument("agent_id")
    r.add_argument("args", nargs="*", help="arguments, read as Python literals")
    r.add_argument("--dir", default=str(DEFAULT_DIR))

    sub.add_parser("stats", help="what is in the registry").add_argument("--dir", default=str(DEFAULT_DIR))

    a = p.parse_args(argv)
    if a.cmd == "build":
        def progress(kept: int, bad: int) -> None:
            print(f"  {kept} agents verified ({bad} rejected)", flush=True)

        reg, rejected = Registry.build(a.scale, verify=not a.no_verify, on_progress=progress)
        path = reg.write(Path(a.out))
        print(json.dumps({"written": str(path), "agents": len(reg), "rejected": len(rejected),
                          "families": reg.counts()}, indent=2))
        for bad_id, why in rejected[:10]:
            print(f"  rejected {bad_id}: {why}", file=sys.stderr)
        return 0
    if a.cmd == "verify":
        reg = Registry.load(Path(a.dir))
        pool = [x for x in reg.agents if x.tier == "verified"]
        if a.sample:
            pool = random.Random(0).sample(pool, min(a.sample, len(pool)))
        bad = [(x.id, check(x)[1]) for x in pool if not check(x)[0]]
        print(f"checked {len(pool)} of {len(reg)} agents: {len(pool) - len(bad)} pass, {len(bad)} fail")
        for bad_id, why in bad[:20]:
            print(f"  {bad_id}: {why}")
        return 1 if bad else 0
    if a.cmd == "search":
        reg = Registry.load(Path(a.dir))
        for score, agent in reg.search(" ".join(a.query), k=a.k, tier=a.tier):
            print(f"{score:5.1f}  {agent.id:<42} [{agent.tier}] {agent.description[:90]}")
        return 0
    if a.cmd == "run":
        reg = Registry.load(Path(a.dir))
        agent = reg.by_id.get(a.agent_id)
        if agent is None:
            print(f"no agent called {a.agent_id}", file=sys.stderr)
            return 2
        import ast as _ast

        args = [_ast.literal_eval(x) for x in a.args]
        print(repr(call(agent, *args)))
        return 0
    reg = Registry.load(Path(a.dir))
    print(json.dumps({"agents": len(reg), "verified": sum(x.tier == "verified" for x in reg.agents),
                      "reference": sum(x.tier == "reference" for x in reg.agents),
                      "families": reg.counts()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
