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
    yield from roman_agents()
    yield from geometry_agents()
    yield from matrix_agents()
    yield from validation_agents()
    yield from set_agents()
    yield from extra_stat_agents()
    yield from text_agents(30 * scale)
    yield from sequence_number_agents()
    yield from coprime_agents(80 * scale)
    yield from rounding_agents()
    yield from percent_agents()
    yield from nth_value_agents(40 * scale)
    yield from window_agents(40 * scale)
    yield from stdlib_agents()


# ------------------------------------------------- more families (2026-10-26)
def coprime_agents(hi: int = 80) -> Iterator[Agent]:
    from math import gcd as _gcd

    for k in range(2, hi + 1):
        yield _mk("number", f"gcd_with_{k}",
                  f"The greatest common divisor of a whole number and {k}.",
                  f"def gcd_with_{k}(n):\n"
                  f'    """The largest number that divides both n and {k}."""\n'
                  f"    from math import gcd\n"
                  f"    return gcd(abs(n), {k})\n",
                  [f"assert gcd_with_{k}({k * 3}) == {_gcd(k * 3, k)}",
                   f"assert gcd_with_{k}(0) == {k}",
                   f"assert gcd_with_{k}(1) == 1"],
                  ["gcd", "common", "divisor", "factor", str(k)])
        yield _mk("number", f"lcm_with_{k}",
                  f"The lowest common multiple of a whole number and {k}.",
                  f"def lcm_with_{k}(n):\n"
                  f'    """The smallest number that both n and {k} divide."""\n'
                  f"    from math import gcd\n"
                  f"    if n == 0:\n        return 0\n"
                  f"    return abs(n) * {k} // gcd(abs(n), {k})\n",
                  [f"assert lcm_with_{k}(6) == {6 * k // _gcd(6, k)}",
                   f"assert lcm_with_{k}(0) == 0"],
                  ["lcm", "lowest", "common", "multiple", str(k)])
        yield _mk("number", f"is_coprime_with_{k}",
                  f"True when a whole number shares no factor above 1 with {k}.",
                  f"def is_coprime_with_{k}(n):\n"
                  f'    """True when n and {k} are coprime."""\n'
                  f"    from math import gcd\n"
                  f"    return gcd(abs(n), {k}) == 1\n",
                  [f"assert is_coprime_with_{k}(1) is True",
                   f"assert is_coprime_with_{k}({k}) is {_gcd(k, k) == 1}"],
                  ["coprime", "relatively", "prime", "gcd", str(k)])


#: the round numbers people actually round to
_NEAREST = (2, 3, 4, 5, 10, 20, 25, 50, 100, 250, 500, 1000, 5000, 10000)


def rounding_agents() -> Iterator[Agent]:
    for d in range(13):
        yield _mk("round", f"round_to_{d}_places",
                  f"Round a number to {d} decimal places.",
                  f"def round_to_{d}_places(x):\n"
                  f'    """Round to {d} decimal places, half to even as Python does."""\n'
                  f"    return round(x, {d})\n",
                  [f"assert round_to_{d}_places(1.0) == 1.0",
                   f"assert round_to_{d}_places({1 / 3}) == {round(1 / 3, d)}"],
                  ["round", "decimal", "places", "precision", str(d)])
    for k in _NEAREST:
        for op, doc, body, fn in (
            ("round", "Round to the nearest multiple of {k}.", "return round(x / {k}) * {k}",
             lambda x, k: round(x / k) * k),
            ("floor", "Round down to a multiple of {k}.", "return (x // {k}) * {k}",
             lambda x, k: (x // k) * k),
            ("ceil", "Round up to a multiple of {k}.", "return -((-x) // {k}) * {k}",
             lambda x, k: -((-x) // k) * k),
        ):
            name = f"{op}_to_nearest_{k}" if op == "round" else f"{op}_to_multiple_of_{k}"
            probe = k * 3 + 1
            yield _mk("round", name, doc.format(k=k) + f" Works on whole numbers.",
                      f"def {name}(x):\n"
                      f'    """{doc.format(k=k)}"""\n'
                      f"    {body.format(k=k)}\n",
                      [f"assert {name}({probe}) == {fn(probe, k)}",
                       f"assert {name}({k * 2}) == {fn(k * 2, k)}"],
                      [op, "round", "nearest", "multiple", str(k)])


def percent_agents() -> Iterator[Agent]:
    for p in range(1, 100):
        yield _mk("percent", f"percent_{p}_of",
                  f"{p} percent of a number.",
                  f"def percent_{p}_of(x):\n"
                  f'    """{p}% of x."""\n'
                  f"    return x * {p} / 100\n",
                  [f"assert percent_{p}_of(200) == {200 * p / 100}",
                   f"assert percent_{p}_of(0) == 0"],
                  ["percent", "percentage", "of", str(p)])
        yield _mk("percent", f"discount_{p}_percent",
                  f"A price after a {p} percent discount.",
                  f"def discount_{p}_percent(price):\n"
                  f'    """Take {p}% off a price."""\n'
                  f"    return price * (100 - {p}) / 100\n",
                  [f"assert discount_{p}_percent(100) == {100 * (100 - p) / 100}",
                   f"assert discount_{p}_percent(0) == 0"],
                  ["discount", "sale", "off", "price", "percent", str(p)])
        yield _mk("percent", f"markup_{p}_percent",
                  f"A price after a {p} percent increase.",
                  f"def markup_{p}_percent(price):\n"
                  f'    """Add {p}% to a price."""\n'
                  f"    return price * (100 + {p}) / 100\n",
                  [f"assert markup_{p}_percent(100) == {100 * (100 + p) / 100}",
                   f"assert markup_{p}_percent(0) == 0"],
                  ["markup", "increase", "tax", "vat", "tip", "percent", str(p)])


GEOMETRY_AGENTS: List[Tuple[str, str, str, str, List[str], List[str]]] = [
    ("circle_area", "The area of a circle from its radius.", "r",
     "    import math\n    return math.pi * r * r",
     ["assert abs(circle_area(1) - 3.141592653589793) < 1e-12", "assert circle_area(0) == 0"],
     ["circle", "area", "radius", "geometry"]),
    ("circle_circumference", "The circumference of a circle from its radius.", "r",
     "    import math\n    return 2 * math.pi * r",
     ["assert abs(circle_circumference(1) - 6.283185307179586) < 1e-12"],
     ["circle", "circumference", "perimeter", "radius", "geometry"]),
    ("square_area", "The area of a square from its side.", "side",
     "    return side * side", ["assert square_area(3) == 9", "assert square_area(0) == 0"],
     ["square", "area", "side", "geometry"]),
    ("square_perimeter", "The perimeter of a square from its side.", "side",
     "    return 4 * side", ["assert square_perimeter(3) == 12"],
     ["square", "perimeter", "side", "geometry"]),
    ("cube_volume", "The volume of a cube from its side.", "side",
     "    return side ** 3", ["assert cube_volume(3) == 27"],
     ["cube", "volume", "side", "geometry"]),
    ("cube_surface_area", "The surface area of a cube from its side.", "side",
     "    return 6 * side * side", ["assert cube_surface_area(2) == 24"],
     ["cube", "surface", "area", "geometry"]),
    ("sphere_volume", "The volume of a sphere from its radius.", "r",
     "    import math\n    return 4 / 3 * math.pi * r ** 3",
     ["assert abs(sphere_volume(1) - 4.188790204786391) < 1e-12"],
     ["sphere", "ball", "volume", "radius", "geometry"]),
    ("sphere_surface_area", "The surface area of a sphere from its radius.", "r",
     "    import math\n    return 4 * math.pi * r * r",
     ["assert abs(sphere_surface_area(1) - 12.566370614359172) < 1e-12"],
     ["sphere", "surface", "area", "radius", "geometry"]),
    ("hypotenuse_of", "The hypotenuse of a right triangle from its two legs.", "a, b",
     "    return (a * a + b * b) ** 0.5",
     ["assert hypotenuse_of(3, 4) == 5.0"],
     ["hypotenuse", "pythagoras", "triangle", "geometry"]),
    ("rectangle_area", "The area of a rectangle from its width and height.", "w, h",
     "    return w * h", ["assert rectangle_area(3, 4) == 12"],
     ["rectangle", "area", "width", "height", "geometry"]),
    ("rectangle_perimeter", "The perimeter of a rectangle from its width and height.", "w, h",
     "    return 2 * (w + h)", ["assert rectangle_perimeter(3, 4) == 14"],
     ["rectangle", "perimeter", "width", "height", "geometry"]),
    ("triangle_area_heron", "The area of a triangle from its three sides.", "a, b, c",
     "    s = (a + b + c) / 2\n    return (s * (s - a) * (s - b) * (s - c)) ** 0.5",
     ["assert triangle_area_heron(3, 4, 5) == 6.0"],
     ["triangle", "area", "heron", "sides", "geometry"]),
    ("trapezoid_area", "The area of a trapezoid from its two bases and its height.", "a, b, h",
     "    return (a + b) / 2 * h", ["assert trapezoid_area(3, 5, 4) == 16.0"],
     ["trapezoid", "trapezium", "area", "geometry"]),
    ("cylinder_volume", "The volume of a cylinder from its radius and height.", "r, h",
     "    import math\n    return math.pi * r * r * h",
     ["assert abs(cylinder_volume(1, 1) - 3.141592653589793) < 1e-12"],
     ["cylinder", "volume", "radius", "height", "geometry"]),
    ("cone_volume", "The volume of a cone from its radius and height.", "r, h",
     "    import math\n    return math.pi * r * r * h / 3",
     ["assert abs(cone_volume(1, 3) - 3.141592653589793) < 1e-12"],
     ["cone", "volume", "radius", "height", "geometry"]),
    ("distance_between_points", "The straight-line distance between two points on a plane.", "x1, y1, x2, y2",
     "    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5",
     ["assert distance_between_points(0, 0, 3, 4) == 5.0"],
     ["distance", "points", "euclidean", "geometry"]),
]


def geometry_agents() -> Iterator[Agent]:
    import math as _math

    for name, desc, arg, body, tests, tags in GEOMETRY_AGENTS:
        yield _mk("geometry", name, desc, f"def {name}({arg}):\n    \"\"\"{desc}\"\"\"\n{body}\n", tests, tags)
    for n in range(3, 31):
        area = n * 1.0 ** 2 / (4 * _math.tan(_math.pi / n))
        yield _mk("geometry", f"regular_{n}gon_area",
                  f"The area of a regular polygon with {n} equal sides, from the side length.",
                  f"def regular_{n}gon_area(side):\n"
                  f'    """The area of a regular {n}-sided polygon with this side length."""\n'
                  f"    import math\n"
                  f"    return {n} * side ** 2 / (4 * math.tan(math.pi / {n}))\n",
                  [f"assert abs(regular_{n}gon_area(1) - {area!r}) < 1e-9",
                   f"assert regular_{n}gon_area(0) == 0"],
                  ["polygon", "regular", "area", f"{n}gon", "geometry", str(n)])


def nth_value_agents(hi: int = 40) -> Iterator[Agent]:
    for k in range(1, hi + 1):
        yield _mk("list", f"nth_largest_{k}",
                  f"The {k}th largest number in a list.",
                  f"def nth_largest_{k}(xs):\n"
                  f'    """The {k}th largest value, counting repeats."""\n'
                  f"    s = sorted(xs, reverse=True)\n"
                  f"    if len(s) < {k}:\n        raise ValueError('fewer than {k} values')\n"
                  f"    return s[{k - 1}]\n",
                  [f"assert nth_largest_{k}(list(range(1, {hi + 5}))) == {hi + 5 - k}"],
                  ["largest", "biggest", "max", "nth", "rank", str(k)])
        yield _mk("list", f"nth_smallest_{k}",
                  f"The {k}th smallest number in a list.",
                  f"def nth_smallest_{k}(xs):\n"
                  f'    """The {k}th smallest value, counting repeats."""\n'
                  f"    s = sorted(xs)\n"
                  f"    if len(s) < {k}:\n        raise ValueError('fewer than {k} values')\n"
                  f"    return s[{k - 1}]\n",
                  [f"assert nth_smallest_{k}(list(range(1, {hi + 5}))) == {k}"],
                  ["smallest", "least", "min", "nth", "rank", str(k)])


def window_agents(hi: int = 40) -> Iterator[Agent]:
    for k in range(2, hi + 1):
        data = list(range(1, hi + 6))
        for op, doc, expr, fn in (
            ("moving_average", "The mean of every run of {k} values in a row.",
             "sum(xs[i:i + {k}]) / {k}", lambda w: sum(w) / len(w)),
            ("rolling_sum", "The total of every run of {k} values in a row.",
             "sum(xs[i:i + {k}])", sum),
            ("rolling_max", "The largest of every run of {k} values in a row.",
             "max(xs[i:i + {k}])", max),
            ("rolling_min", "The smallest of every run of {k} values in a row.",
             "min(xs[i:i + {k}])", min),
        ):
            name = f"{op}_{k}"
            want = [fn(data[i:i + k]) for i in range(len(data) - k + 1)]
            yield _mk("window", name, doc.format(k=k) + " Returns one value per window.",
                      f"def {name}(xs):\n"
                      f'    """{doc.format(k=k)}"""\n'
                      f"    return [{expr.format(k=k)} for i in range(len(xs) - {k} + 1)]\n",
                      [f"assert {name}({data!r}) == {want!r}",
                       f"assert {name}([1]) == []"],
                      [op.replace("_", " "), "window", "rolling", "moving", str(k)])


TEXT_AGENTS: List[Tuple[str, str, str, List[str], List[str]]] = [
    ("to_snake_case", "Rewrite a phrase as snake_case.",
     "    import re\n    s = re.sub(r'([a-z0-9])([A-Z])', r'\\1 \\2', s)\n"
     "    return '_'.join(w.lower() for w in re.split(r'[^A-Za-z0-9]+', s) if w)",
     ["assert to_snake_case('Hello World') == 'hello_world'",
      "assert to_snake_case('camelCaseName') == 'camel_case_name'"],
     ["snake", "case", "underscore", "text"]),
    ("to_kebab_case", "Rewrite a phrase as kebab-case.",
     "    import re\n    s = re.sub(r'([a-z0-9])([A-Z])', r'\\1 \\2', s)\n"
     "    return '-'.join(w.lower() for w in re.split(r'[^A-Za-z0-9]+', s) if w)",
     ["assert to_kebab_case('Hello World') == 'hello-world'"],
     ["kebab", "case", "dash", "slug", "text"]),
    ("to_camel_case", "Rewrite a phrase as camelCase.",
     "    import re\n    parts = [w for w in re.split(r'[^A-Za-z0-9]+', s) if w]\n"
     "    if not parts:\n        return ''\n"
     "    return parts[0].lower() + ''.join(p[:1].upper() + p[1:].lower() for p in parts[1:])",
     ["assert to_camel_case('hello world') == 'helloWorld'", "assert to_camel_case('') == ''"],
     ["camel", "case", "text"]),
    ("to_pascal_case", "Rewrite a phrase as PascalCase.",
     "    import re\n    parts = [w for w in re.split(r'[^A-Za-z0-9]+', s) if w]\n"
     "    return ''.join(p[:1].upper() + p[1:].lower() for p in parts)",
     ["assert to_pascal_case('hello world') == 'HelloWorld'"],
     ["pascal", "case", "text"]),
    ("to_title_case", "Capitalise the first letter of every word.",
     "    return ' '.join(w[:1].upper() + w[1:] for w in s.split(' '))",
     ["assert to_title_case('hello world') == 'Hello World'"],
     ["title", "case", "capitalise", "text"]),
    ("slugify_text", "Turn a title into a URL slug.",
     "    import re\n    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')",
     ["assert slugify_text('Hello, World!') == 'hello-world'"],
     ["slug", "url", "link", "text"]),
    ("reverse_words", "Reverse the order of the words in a sentence.",
     "    return ' '.join(reversed(s.split()))",
     ["assert reverse_words('a b c') == 'c b a'", "assert reverse_words('') == ''"],
     ["reverse", "words", "order", "text"]),
    ("count_words", "How many words a piece of text has.",
     "    return len(s.split())",
     ["assert count_words('a b  c') == 3", "assert count_words('') == 0"],
     ["count", "words", "length", "text"]),
    ("count_vowels", "How many vowels a piece of text has.",
     "    return sum(1 for c in s.lower() if c in 'aeiou')",
     ["assert count_vowels('Hello') == 2", "assert count_vowels('xyz') == 0"],
     ["count", "vowels", "letters", "text"]),
    ("initials_of", "The initials of a name.",
     "    return ''.join(w[0].upper() for w in s.split() if w)",
     ["assert initials_of('ada lovelace') == 'AL'", "assert initials_of('') == ''"],
     ["initials", "name", "abbreviate", "text"]),
    ("is_palindrome_text", "True when a phrase reads the same backwards, ignoring case and punctuation.",
     "    t = ''.join(c for c in s.lower() if c.isalnum())\n    return t == t[::-1]",
     ["assert is_palindrome_text('A man, a plan, a canal: Panama') is True",
      "assert is_palindrome_text('hello') is False"],
     ["palindrome", "reverse", "same", "text"]),
    ("longest_word", "The longest word in a piece of text; the first one when several tie.",
     "    words = s.split()\n    if not words:\n        raise ValueError('no words')\n"
     "    return max(words, key=len)",
     ["assert longest_word('a bb ccc') == 'ccc'"],
     ["longest", "word", "max", "text"]),
    ("squeeze_spaces", "Collapse runs of whitespace into single spaces and trim the ends.",
     "    return ' '.join(s.split())",
     ["assert squeeze_spaces('  a   b ') == 'a b'"],
     ["whitespace", "spaces", "trim", "clean", "text"]),
    ("character_counts", "How many times each character appears, as a dict.",
     "    out = {}\n    for c in s:\n        out[c] = out.get(c, 0) + 1\n    return out",
     ["assert character_counts('aab') == {'a': 2, 'b': 1}", "assert character_counts('') == {}"],
     ["count", "characters", "frequency", "histogram", "text"]),
]


def text_agents(hi: int = 30) -> Iterator[Agent]:
    for name, desc, body, tests, tags in TEXT_AGENTS:
        yield _mk("text", name, desc, f"def {name}(s):\n    \"\"\"{desc}\"\"\"\n{body}\n", tests, tags)
    for k in range(1, hi + 1):
        words = " ".join(f"w{i}" for i in range(1, hi + 3))
        yield _mk("text", f"first_{k}_words",
                  f"The first {k} words of a piece of text.",
                  f"def first_{k}_words(s):\n"
                  f'    """The first {k} words, joined with single spaces."""\n'
                  f"    return ' '.join(s.split()[:{k}])\n",
                  [f"assert first_{k}_words({words!r}) == {' '.join(words.split()[:k])!r}",
                   f"assert first_{k}_words('') == ''"],
                  ["first", "words", "head", "excerpt", "text", str(k)])


MATRIX_AGENTS: List[Tuple[str, str, str, List[str], List[str]]] = [
    ("transpose_matrix", "Swap the rows and columns of a matrix given as a list of rows.",
     "    return [list(c) for c in zip(*m)]",
     ["assert transpose_matrix([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]",
      "assert transpose_matrix([]) == []"],
     ["transpose", "matrix", "rows", "columns"]),
    ("matrix_row_sums", "The total of each row of a matrix.",
     "    return [sum(r) for r in m]", ["assert matrix_row_sums([[1, 2], [3, 4]]) == [3, 7]"],
     ["row", "sum", "matrix", "total"]),
    ("matrix_column_sums", "The total of each column of a matrix.",
     "    return [sum(c) for c in zip(*m)]", ["assert matrix_column_sums([[1, 2], [3, 4]]) == [4, 6]"],
     ["column", "sum", "matrix", "total"]),
    ("matrix_trace", "The sum of the diagonal of a square matrix.",
     "    return sum(m[i][i] for i in range(len(m)))",
     ["assert matrix_trace([[1, 2], [3, 4]]) == 5"],
     ["trace", "diagonal", "matrix"]),
    ("flatten_matrix", "One flat list from a list of rows.",
     "    return [v for r in m for v in r]",
     ["assert flatten_matrix([[1, 2], [3]]) == [1, 2, 3]"],
     ["flatten", "matrix", "list", "concatenate"]),
    ("determinant_2x2", "The determinant of a 2 by 2 matrix.",
     "    (a, b), (c, d) = m\n    return a * d - b * c",
     ["assert determinant_2x2([[1, 2], [3, 4]]) == -2"],
     ["determinant", "matrix", "2x2"]),
    ("determinant_3x3", "The determinant of a 3 by 3 matrix.",
     "    (a, b, c), (d, e, f), (g, h, i) = m\n"
     "    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)",
     ["assert determinant_3x3([[6, 1, 1], [4, -2, 5], [2, 8, 7]]) == -306"],
     ["determinant", "matrix", "3x3"]),
    ("matrix_shape", "The (rows, columns) of a matrix.",
     "    return (len(m), len(m[0]) if m else 0)",
     ["assert matrix_shape([[1, 2], [3, 4]]) == (2, 2)", "assert matrix_shape([]) == (0, 0)"],
     ["shape", "size", "dimensions", "matrix"]),
    ("matrix_diagonal", "The diagonal of a square matrix as a list.",
     "    return [m[i][i] for i in range(len(m))]",
     ["assert matrix_diagonal([[1, 2], [3, 4]]) == [1, 4]"],
     ["diagonal", "matrix"]),
    ("rotate_matrix_clockwise", "Turn a matrix a quarter turn clockwise.",
     "    return [list(r) for r in zip(*m[::-1])]",
     ["assert rotate_matrix_clockwise([[1, 2], [3, 4]]) == [[3, 1], [4, 2]]"],
     ["rotate", "turn", "clockwise", "matrix"]),
]


def matrix_agents() -> Iterator[Agent]:
    for name, desc, body, tests, tags in MATRIX_AGENTS:
        yield _mk("matrix", name, desc, f"def {name}(m):\n    \"\"\"{desc}\"\"\"\n{body}\n", tests, tags)


VALIDATION_AGENTS: List[Tuple[str, str, str, List[str], List[str]]] = [
    ("is_valid_luhn", "True when a card-style number passes the Luhn check.",
     "    ds = [int(c) for c in str(s) if c.isdigit()]\n    if not ds:\n        return False\n"
     "    total = 0\n"
     "    for i, d in enumerate(reversed(ds)):\n"
     "        if i % 2:\n            d *= 2\n            if d > 9:\n                d -= 9\n"
     "        total += d\n    return total % 10 == 0",
     ["assert is_valid_luhn('4539578763621486') is True",
      "assert is_valid_luhn('4539578763621487') is False"],
     ["luhn", "card", "checksum", "validate"]),
    ("is_valid_isbn10", "True when a 10-character ISBN has a correct check digit.",
     "    t = str(s).replace('-', '').replace(' ', '').upper()\n"
     "    if len(t) != 10:\n        return False\n    total = 0\n"
     "    for i, c in enumerate(t):\n"
     "        v = 10 if c == 'X' and i == 9 else (int(c) if c.isdigit() else None)\n"
     "        if v is None:\n            return False\n        total += (10 - i) * v\n"
     "    return total % 11 == 0",
     ["assert is_valid_isbn10('0306406152') is True", "assert is_valid_isbn10('0306406153') is False"],
     ["isbn", "book", "checksum", "validate"]),
    ("is_valid_ipv4", "True when a string is a dotted-quad IPv4 address.",
     "    parts = str(s).split('.')\n    if len(parts) != 4:\n        return False\n"
     "    for p in parts:\n"
     "        if not p.isdigit() or (len(p) > 1 and p[0] == '0') or int(p) > 255:\n            return False\n"
     "    return True",
     ["assert is_valid_ipv4('192.168.0.1') is True", "assert is_valid_ipv4('256.0.0.1') is False"],
     ["ip", "ipv4", "address", "network", "validate"]),
    ("is_valid_hex_colour", "True when a string is a #rgb or #rrggbb colour.",
     "    t = str(s)\n    if not t.startswith('#') or len(t) not in (4, 7):\n        return False\n"
     "    return all(c in '0123456789abcdefABCDEF' for c in t[1:])",
     ["assert is_valid_hex_colour('#1a2B3c') is True", "assert is_valid_hex_colour('1a2b3c') is False"],
     ["hex", "colour", "color", "css", "validate"]),
    ("looks_like_email", "True when a string has the shape of an email address.",
     "    import re\n    return bool(re.fullmatch(r'[^@\\s]+@[^@\\s.]+(\\.[^@\\s.]+)+', str(s)))",
     ["assert looks_like_email('a@b.com') is True", "assert looks_like_email('a@b') is False"],
     ["email", "address", "validate"]),
    ("digits_only", "Just the digits of a string, in order.",
     "    return ''.join(c for c in str(s) if c.isdigit())",
     ["assert digits_only('a1-b2') == '12'", "assert digits_only('') == ''"],
     ["digits", "strip", "clean", "validate"]),
]


def validation_agents() -> Iterator[Agent]:
    for name, desc, body, tests, tags in VALIDATION_AGENTS:
        yield _mk("validate", name, desc, f"def {name}(s):\n    \"\"\"{desc}\"\"\"\n{body}\n", tests, tags)


SEQUENCE_AGENTS: List[Tuple[str, str, str, List[str], List[str]]] = [
    ("nth_lucas", "The nth Lucas number, counting from 0.",
     "    a, b = 2, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
     ["assert nth_lucas(0) == 2", "assert nth_lucas(10) == 123"],
     ["lucas", "sequence", "series"]),
    ("nth_triangular", "The nth triangular number, counting from 1.",
     "    return n * (n + 1) // 2", ["assert nth_triangular(1) == 1", "assert nth_triangular(10) == 55"],
     ["triangular", "sequence", "series"]),
    ("nth_pentagonal", "The nth pentagonal number, counting from 1.",
     "    return n * (3 * n - 1) // 2", ["assert nth_pentagonal(1) == 1", "assert nth_pentagonal(5) == 35"],
     ["pentagonal", "sequence", "series"]),
    ("nth_hexagonal", "The nth hexagonal number, counting from 1.",
     "    return n * (2 * n - 1)", ["assert nth_hexagonal(1) == 1", "assert nth_hexagonal(5) == 45"],
     ["hexagonal", "sequence", "series"]),
    ("nth_catalan", "The nth Catalan number, counting from 0.",
     "    num = den = 1\n    for i in range(n):\n        num *= 2 * n - i\n        den *= i + 1\n"
     "    return num // den // (n + 1)",
     ["assert nth_catalan(0) == 1", "assert nth_catalan(5) == 42"],
     ["catalan", "sequence", "combinatorics"]),
    ("nth_tribonacci", "The nth tribonacci number, counting from 0 with 0, 0, 1.",
     "    a, b, c = 0, 0, 1\n    for _ in range(n):\n        a, b, c = b, c, a + b + c\n    return a",
     ["assert nth_tribonacci(0) == 0", "assert nth_tribonacci(10) == 81"],
     ["tribonacci", "sequence", "series"]),
    ("nth_prime", "The nth prime number, counting from 1.",
     "    if n < 1:\n        raise ValueError('n starts at 1')\n"
     "    found, c = 0, 1\n"
     "    while found < n:\n        c += 1\n        d = 2\n        ok = True\n"
     "        while d * d <= c:\n"
     "            if c % d == 0:\n                ok = False\n                break\n            d += 1\n"
     "        if ok:\n            found += 1\n    return c",
     ["assert nth_prime(1) == 2", "assert nth_prime(10) == 29"],
     ["prime", "nth", "sequence"]),
    ("next_prime_after", "The smallest prime larger than a number.",
     "    c = n + 1\n"
     "    while True:\n        if c > 1:\n            d, ok = 2, True\n"
     "            while d * d <= c:\n"
     "                if c % d == 0:\n                    ok = False\n                    break\n"
     "                d += 1\n            if ok:\n                return c\n        c += 1",
     ["assert next_prime_after(10) == 11", "assert next_prime_after(13) == 17"],
     ["prime", "next", "after", "sequence"]),
    ("primorial_of", "The product of the first n primes.",
     "    out, found, c = 1, 0, 1\n"
     "    while found < n:\n        c += 1\n        d, ok = 2, True\n"
     "        while d * d <= c:\n"
     "            if c % d == 0:\n                ok = False\n                break\n            d += 1\n"
     "        if ok:\n            out *= c\n            found += 1\n    return out",
     ["assert primorial_of(0) == 1", "assert primorial_of(5) == 2310"],
     ["primorial", "prime", "product"]),
    ("double_factorial_of", "The product of every other number down from n.",
     "    out = 1\n    while n > 1:\n        out *= n\n        n -= 2\n    return out",
     ["assert double_factorial_of(0) == 1", "assert double_factorial_of(7) == 105"],
     ["double", "factorial", "product"]),
    ("collatz_steps", "How many steps the Collatz sequence takes from n down to 1.",
     "    if n < 1:\n        raise ValueError('n starts at 1')\n    steps = 0\n"
     "    while n != 1:\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        steps += 1\n"
     "    return steps",
     ["assert collatz_steps(1) == 0", "assert collatz_steps(27) == 111"],
     ["collatz", "hailstone", "steps", "sequence"]),
    ("first_n_primes", "The first n prime numbers, in order.",
     "    out, c = [], 1\n"
     "    while len(out) < n:\n        c += 1\n"
     "        if all(c % d for d in range(2, int(c ** 0.5) + 1)):\n            out.append(c)\n"
     "    return out",
     ["assert first_n_primes(0) == []", "assert first_n_primes(5) == [2, 3, 5, 7, 11]"],
     ["prime", "primes", "first", "list"]),
    ("is_perfect_number", "True when a number equals the sum of its proper divisors.",
     "    if n < 2:\n        return False\n"
     "    return sum(d for d in range(1, n) if n % d == 0) == n",
     ["assert is_perfect_number(28) is True", "assert is_perfect_number(27) is False"],
     ["perfect", "divisors", "number"]),
]


def sequence_number_agents(hi: int = 60) -> Iterator[Agent]:
    """The named integer sequences. Deliberately not parameterised by n.

    A ``first_360_primes()`` family looked like free agents and was a trap: it
    takes no argument, so any question with 360 in it made one of them look
    runnable, and "prime factors of 360" started answering with a list of the
    first 360 primes. One agent that takes n is worth more than 400 that bake
    it in.
    """
    for name, desc, body, tests, tags in SEQUENCE_AGENTS:
        yield _mk("sequence", name, desc, f"def {name}(n):\n    \"\"\"{desc}\"\"\"\n{body}\n", tests, tags)


SET_AGENTS: List[Tuple[str, str, str, List[str], List[str]]] = [
    ("common_items", "The values in both lists, sorted.",
     "    return sorted(set(a) & set(b))",
     ["assert common_items([1, 2, 3], [2, 3, 4]) == [2, 3]"],
     ["common", "intersection", "both", "shared", "set"]),
    ("combined_items", "Every value from both lists, without repeats, sorted.",
     "    return sorted(set(a) | set(b))",
     ["assert combined_items([1, 2], [2, 3]) == [1, 2, 3]"],
     ["union", "combine", "merge", "set"]),
    ("only_in_first", "The values in the first list and not the second, sorted.",
     "    return sorted(set(a) - set(b))",
     ["assert only_in_first([1, 2, 3], [2]) == [1, 3]"],
     ["difference", "only", "missing", "set"]),
    ("in_one_but_not_both", "The values in exactly one of two lists, sorted.",
     "    return sorted(set(a) ^ set(b))",
     ["assert in_one_but_not_both([1, 2], [2, 3]) == [1, 3]"],
     ["symmetric", "difference", "exclusive", "set"]),
    ("jaccard_similarity", "How much two lists overlap, from 0 to 1.",
     "    sa, sb = set(a), set(b)\n    u = sa | sb\n"
     "    return len(sa & sb) / len(u) if u else 1.0",
     ["assert jaccard_similarity([1, 2], [2, 3]) == 1 / 3",
      "assert jaccard_similarity([], []) == 1.0"],
     ["jaccard", "similarity", "overlap", "set"]),
    ("is_subset_of", "True when every value of the first list is in the second.",
     "    return set(a) <= set(b)",
     ["assert is_subset_of([1], [1, 2]) is True", "assert is_subset_of([3], [1, 2]) is False"],
     ["subset", "contains", "inside", "set"]),
]


def set_agents() -> Iterator[Agent]:
    for name, desc, body, tests, tags in SET_AGENTS:
        yield _mk("set", name, desc, f"def {name}(a, b):\n    \"\"\"{desc}\"\"\"\n{body}\n", tests, tags)


EXTRA_STAT_AGENTS: List[Tuple[str, str, str, List[str], List[str]]] = [
    ("geometric_mean_of", "The geometric mean of a list of positive numbers.",
     "    if not xs:\n        raise ValueError('no data')\n    p = 1.0\n"
     "    for x in xs:\n        p *= x\n    return p ** (1 / len(xs))",
     ["assert abs(geometric_mean_of([1, 4]) - 2.0) < 1e-12"],
     ["geometric", "mean", "average", "statistics"]),
    ("harmonic_mean_of", "The harmonic mean of a list of positive numbers.",
     "    if not xs:\n        raise ValueError('no data')\n    return len(xs) / sum(1 / x for x in xs)",
     ["assert harmonic_mean_of([1, 1]) == 1.0", "assert abs(harmonic_mean_of([1, 2]) - 4 / 3) < 1e-12"],
     ["harmonic", "mean", "average", "statistics"]),
    ("mean_absolute_deviation", "The mean distance of the values from their mean.",
     "    if not xs:\n        raise ValueError('no data')\n    m = sum(xs) / len(xs)\n"
     "    return sum(abs(x - m) for x in xs) / len(xs)",
     ["assert mean_absolute_deviation([1, 3]) == 1.0", "assert mean_absolute_deviation([2, 2]) == 0.0"],
     ["deviation", "absolute", "spread", "statistics"]),
    ("interquartile_range", "The spread between the first and third quartile.",
     "    if not xs:\n        raise ValueError('no data')\n    s = sorted(xs)\n    n = len(s)\n"
     "    def q(p):\n        i = p * (n - 1)\n        lo = int(i)\n        hi = min(n - 1, lo + 1)\n"
     "        return s[lo] + (s[hi] - s[lo]) * (i - lo)\n    return q(0.75) - q(0.25)",
     ["assert interquartile_range([1, 2, 3, 4, 5]) == 2.0"],
     ["iqr", "quartile", "spread", "statistics"]),
    ("sample_stdev_of", "The sample standard deviation of a list of numbers.",
     "    if len(xs) < 2:\n        raise ValueError('need two values')\n    m = sum(xs) / len(xs)\n"
     "    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5",
     ["assert sample_stdev_of([1, 3]) == 2 ** 0.5"],
     ["sample", "standard", "deviation", "statistics"]),
    ("coefficient_of_variation", "The standard deviation as a fraction of the mean.",
     "    if not xs:\n        raise ValueError('no data')\n    m = sum(xs) / len(xs)\n"
     "    if m == 0:\n        raise ValueError('mean is zero')\n"
     "    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 / m",
     ["assert coefficient_of_variation([2, 2]) == 0.0"],
     ["variation", "relative", "spread", "statistics"]),
    ("cumulative_sums", "A running total of a list of numbers.",
     "    out, t = [], 0\n    for x in xs:\n        t += x\n        out.append(t)\n    return out",
     ["assert cumulative_sums([1, 2, 3]) == [1, 3, 6]", "assert cumulative_sums([]) == []"],
     ["cumulative", "running", "total", "sum", "statistics"]),
    ("normalise_to_unit", "Scale a list so the smallest value is 0 and the largest is 1.",
     "    if not xs:\n        return []\n    lo, hi = min(xs), max(xs)\n"
     "    if hi == lo:\n        return [0.0 for _ in xs]\n    return [(x - lo) / (hi - lo) for x in xs]",
     ["assert normalise_to_unit([1, 2, 3]) == [0.0, 0.5, 1.0]",
      "assert normalise_to_unit([2, 2]) == [0.0, 0.0]"],
     ["normalise", "normalize", "scale", "rescale", "statistics"]),
    ("sum_of_squares_of", "The sum of the squares of a list of numbers.",
     "    return sum(x * x for x in xs)",
     ["assert sum_of_squares_of([1, 2, 3]) == 14", "assert sum_of_squares_of([]) == 0"],
     ["sum", "squares", "statistics"]),
    ("count_distinct", "How many different values a list has.",
     "    return len(set(xs))",
     ["assert count_distinct([1, 1, 2]) == 2", "assert count_distinct([]) == 0"],
     ["distinct", "unique", "count", "statistics"]),
]


def extra_stat_agents() -> Iterator[Agent]:
    for name, desc, body, tests, tags in EXTRA_STAT_AGENTS:
        yield _mk("stats", name, desc, f"def {name}(xs):\n    \"\"\"{desc}\"\"\"\n{body}\n", tests, tags)


ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
         (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def roman_agents() -> Iterator[Agent]:
    yield _mk("roman", "to_roman_numeral", "Write a number from 1 to 3999 as a Roman numeral.",
              "def to_roman_numeral(n):\n"
              '    """A Roman numeral for 1..3999."""\n'
              "    if not 1 <= n <= 3999:\n        raise ValueError('out of range')\n"
              f"    table = {ROMAN!r}\n"
              "    out = []\n"
              "    for value, sign in table:\n"
              "        while n >= value:\n            out.append(sign)\n            n -= value\n"
              "    return ''.join(out)\n",
              ["assert to_roman_numeral(1994) == 'MCMXCIV'", "assert to_roman_numeral(4) == 'IV'"],
              ["roman", "numeral", "number", "convert"])
    yield _mk("roman", "from_roman_numeral", "Read a Roman numeral as a number.",
              "def from_roman_numeral(s):\n"
              '    """The number a Roman numeral stands for."""\n'
              "    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
              "    total, prev = 0, 0\n"
              "    for c in reversed(str(s).upper()):\n"
              "        v = values[c]\n"
              "        total += -v if v < prev else v\n"
              "        prev = max(prev, v)\n"
              "    return total\n",
              ["assert from_roman_numeral('MCMXCIV') == 1994", "assert from_roman_numeral('iv') == 4"],
              ["roman", "numeral", "parse", "convert"])


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
