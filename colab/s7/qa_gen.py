"""Standalone fresh Q&A generator for MNX session 7 (flattened from mnx/agent/qa_stream.py + expert_stream.py,
branch claude/project-thread-nupzv9). Same generators, same outputs for the same seed. Every answer is computed, not written.
    python qa_gen.py --n 600000 --seed 7007 --out fresh.jsonl
"""
from __future__ import annotations
import argparse, ast, datetime as dt, json, math, random, re, sys, textwrap
from fractions import Fraction
from typing import Callable, Dict, Iterator, List, Sequence, Set, Tuple

DOC_SEP = "\n<|doc|>\n"


def format_doc(header: str, doc: str, body: str, tests: Sequence[str]) -> str:
    """The training / prompting format: tests as comments, then the function."""
    comments = "".join(f"# {t}\n" for t in tests)
    return f'{comments}{header}\n    """{doc}"""\n{body}'


def prompt_for(prompt: str, tests: Sequence[str]) -> str:
    """Turn a neural-proposer prompt (def + docstring) into the training format."""
    return "".join(f"# {t}\n" for t in tests) + prompt


# ------------------------------------------------------------------ inputs
def _int(r: random.Random) -> int:
    return r.choice([r.randint(-9, 9), r.randint(0, 20), r.randint(-50, 50)])


def _pos(r: random.Random) -> int:
    return r.randint(1, 12)


def _ints(r: random.Random) -> list:
    return [r.randint(-9, 12) for _ in range(r.randint(0, 6))]


def _ints1(r: random.Random) -> list:
    return [r.randint(-9, 12) for _ in range(r.randint(1, 6))]


_WORDS = "apple banana cherry dog eagle fig grape house ice jam kite lemon moon nut owl pear queen rain sun tree".split()


def _word(r: random.Random) -> str:
    w = r.choice(_WORDS)
    return w.capitalize() if r.random() < 0.3 else w


def _sentence(r: random.Random) -> str:
    return " ".join(r.choice(_WORDS) for _ in range(r.randint(1, 4)))


def _char(r: random.Random) -> str:
    return r.choice("aeiouxyzs")


# --------------------------------------------------------------- templates
# (names, docstrings, params, input makers, bodies, reference). Bodies use the
# params' names; ``K`` is replaced by a constant drawn from ``ks``.
T = Tuple[Sequence[str], Sequence[str], str, Sequence[Callable], Sequence[str], Callable, Sequence[int]]

TEMPLATES: List[T] = [
    # numbers
    (["plus_k", "add_k", "increase"], ["Return n plus K.", "Add K to n."], "n", [_int],
     ["    return n + K\n"], lambda n, K: n + K, [1, 2, 5, 10, 100]),
    (["minus_k", "subtract_k", "decrease"], ["Return n minus K.", "Subtract K from n."], "n", [_int],
     ["    return n - K\n"], lambda n, K: n - K, [1, 2, 3, 10]),
    (["times_k", "multiply_by", "scale"], ["Return n multiplied by K.", "Multiply n by K."], "n", [_int],
     ["    return n * K\n", "    return K * n\n"], lambda n, K: n * K, [2, 4, 5, 10]),
    (["square", "squared", "sq"], ["Return n squared.", "Return the square of n."], "n", [_int],
     ["    return n * n\n", "    return n ** 2\n"], lambda n, K: n * n, [0]),
    (["negate", "opposite"], ["Return the negative of n.", "Flip the sign of n."], "n", [_int],
     ["    return -n\n"], lambda n, K: -n, [0]),
    (["absolute", "magnitude"], ["Return the absolute value of n.", "Return n without its sign."], "n", [_int],
     ["    return abs(n)\n", "    return n if n >= 0 else -n\n"], lambda n, K: abs(n), [0]),
    (["is_multiple", "divisible_by_k"], ["Return True if n is divisible by K.", "Check whether K divides n."], "n", [_int],
     ["    return n % K == 0\n"], lambda n, K: n % K == 0, [3, 5, 7]),
    (["is_odd", "odd"], ["Return True if n is odd.", "Check whether n is odd."], "n", [_int],
     ["    return n % 2 == 1\n", "    return n % 2 != 0\n"], lambda n, K: n % 2 == 1, [0]),
    (["is_positive", "positive"], ["Return True if n is greater than zero.", "Check whether n is positive."], "n", [_int],
     ["    return n > 0\n"], lambda n, K: n > 0, [0]),
    (["is_big", "above_k"], ["Return True if n is greater than K.", "Check whether n exceeds K."], "n", [_int],
     ["    return n > K\n"], lambda n, K: n > K, [5, 10]),
    (["half", "halve"], ["Return n divided by 2, rounded down.", "Integer half of n."], "n", [_int],
     ["    return n // 2\n"], lambda n, K: n // 2, [0]),
    (["last_digit", "ones_digit"], ["Return the last digit of a non-negative n.", "Return n mod 10."], "n", [_pos],
     ["    return n % 10\n"], lambda n, K: n % 10, [0]),
    (["sum_to", "triangle"], ["Return 1 + 2 + ... + n.", "Sum the numbers from 1 to n."], "n", [_pos],
     ["    return n * (n + 1) // 2\n", "    return sum(range(1, n + 1))\n",
      "    total = 0\n    for i in range(1, n + 1):\n        total += i\n    return total\n"], lambda n, K: n * (n + 1) // 2, [0]),
    (["add", "plus", "total_of_two"], ["Return the sum of a and b.", "Add two numbers."], "a, b", [_int, _int],
     ["    return a + b\n"], lambda a, b, K: a + b, [0]),
    (["subtract", "difference"], ["Return a minus b.", "Subtract b from a."], "a, b", [_int, _int],
     ["    return a - b\n"], lambda a, b, K: a - b, [0]),
    (["multiply", "times"], ["Return a times b.", "Multiply two numbers."], "a, b", [_int, _int],
     ["    return a * b\n"], lambda a, b, K: a * b, [0]),
    (["bigger", "larger"], ["Return the larger of a and b.", "Return whichever of a and b is bigger."], "a, b", [_int, _int],
     ["    return max(a, b)\n", "    return a if a > b else b\n"], lambda a, b, K: max(a, b), [0]),
    (["smaller", "lesser"], ["Return the smaller of a and b.", "Return whichever of a and b is smaller."], "a, b", [_int, _int],
     ["    return min(a, b)\n", "    return a if a < b else b\n"], lambda a, b, K: min(a, b), [0]),
    (["midpoint", "mean_of_two"], ["Return the average of a and b, rounded down.", "Integer midpoint of a and b."], "a, b", [_int, _int],
     ["    return (a + b) // 2\n"], lambda a, b, K: (a + b) // 2, [0]),
    (["same_sign"], ["Return True if a and b are both positive or both negative."], "a, b", [_int, _int],
     ["    return (a > 0 and b > 0) or (a < 0 and b < 0)\n"], lambda a, b, K: (a > 0 and b > 0) or (a < 0 and b < 0), [0]),
    (["sum_three", "add3"], ["Return a + b + c.", "Add three numbers."], "a, b, c", [_int, _int, _int],
     ["    return a + b + c\n"], lambda a, b, c, K: a + b + c, [0]),
    # lists
    (["add_k_each", "shift_all"], ["Add K to every number in the list.", "Return a new list with K added to each item."], "xs", [_ints],
     ["    return [x + K for x in xs]\n", "    out = []\n    for x in xs:\n        out.append(x + K)\n    return out\n"],
     lambda xs, K: [x + K for x in xs], [2, 3, 10]),
    (["times_k_each", "scale_all", "double_all"], ["Multiply every number in the list by K.", "Return each item times K."], "xs", [_ints],
     ["    return [x * K for x in xs]\n", "    return [K * x for x in xs]\n"], lambda xs, K: [x * K for x in xs], [2, 3, 10]),
    (["negate_all", "flip_signs"], ["Negate every number in the list.", "Return the list with every sign flipped."], "xs", [_ints],
     ["    return [-x for x in xs]\n"], lambda xs, K: [-x for x in xs], [0]),
    (["abs_all", "absolute_values"], ["Return the absolute value of every number.", "Make every number non-negative."], "xs", [_ints],
     ["    return [abs(x) for x in xs]\n"], lambda xs, K: [abs(x) for x in xs], [0]),
    (["cubes", "cube_each"], ["Return each number cubed.", "Cube every item in the list."], "xs", [_ints],
     ["    return [x ** 3 for x in xs]\n", "    return [x * x * x for x in xs]\n"], lambda xs, K: [x ** 3 for x in xs], [0]),
    (["keep_evens", "only_even", "evens"], ["Keep only the even numbers.", "Return the even items of the list."], "xs", [_ints],
     ["    return [x for x in xs if x % 2 == 0]\n"], lambda xs, K: [x for x in xs if x % 2 == 0], [0]),
    (["keep_odds", "only_odd", "odds"], ["Keep only the odd numbers.", "Return the odd items of the list."], "xs", [_ints],
     ["    return [x for x in xs if x % 2 == 1]\n"], lambda xs, K: [x for x in xs if x % 2 == 1], [0]),
    (["keep_positive", "positives"], ["Keep only the numbers greater than zero.", "Return the positive items."], "xs", [_ints],
     ["    return [x for x in xs if x > 0]\n"], lambda xs, K: [x for x in xs if x > 0], [0]),
    (["above", "greater_than_k"], ["Keep the numbers greater than K.", "Return the items bigger than K."], "xs", [_ints],
     ["    return [x for x in xs if x > K]\n"], lambda xs, K: [x for x in xs if x > K], [3, 5]),
    (["remove_zeros", "drop_zeros"], ["Remove every zero from the list.", "Return the list without zeros."], "xs", [_ints],
     ["    return [x for x in xs if x != 0]\n"], lambda xs, K: [x for x in xs if x != 0], [0]),
    (["total", "list_sum"], ["Return the sum of the list.", "Add up all the numbers."], "xs", [_ints],
     ["    return sum(xs)\n", "    total = 0\n    for x in xs:\n        total += x\n    return total\n"], lambda xs, K: sum(xs), [0]),
    (["count_items", "size"], ["Return how many items the list has.", "Return the length of the list."], "xs", [_ints],
     ["    return len(xs)\n"], lambda xs, K: len(xs), [0]),
    (["largest", "biggest"], ["Return the largest number in a non-empty list.", "Maximum of the list."], "xs", [_ints1],
     ["    return max(xs)\n", "    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n"],
     lambda xs, K: max(xs), [0]),
    (["smallest", "minimum"], ["Return the smallest number in a non-empty list.", "Minimum of the list."], "xs", [_ints1],
     ["    return min(xs)\n"], lambda xs, K: min(xs), [0]),
    (["head", "first_item"], ["Return the first item of a non-empty list.", "Return xs[0]."], "xs", [_ints1],
     ["    return xs[0]\n"], lambda xs, K: xs[0], [0]),
    (["tail", "rest"], ["Return the list without its first item.", "Drop the first element."], "xs", [_ints1],
     ["    return xs[1:]\n"], lambda xs, K: xs[1:], [0]),
    (["reverse_list", "backwards"], ["Return the list in reverse order.", "Reverse the list."], "xs", [_ints],
     ["    return xs[::-1]\n", "    return list(reversed(xs))\n"], lambda xs, K: xs[::-1], [0]),
    (["sort_list", "ascending"], ["Return the list sorted from smallest to largest.", "Sort the numbers."], "xs", [_ints],
     ["    return sorted(xs)\n"], lambda xs, K: sorted(xs), [0]),
    (["sort_desc", "descending"], ["Return the list sorted from largest to smallest.", "Sort in descending order."], "xs", [_ints],
     ["    return sorted(xs, reverse=True)\n"], lambda xs, K: sorted(xs, reverse=True), [0]),
    (["count_evens", "num_even"], ["Count the even numbers in the list.", "How many items are even."], "xs", [_ints],
     ["    return sum(1 for x in xs if x % 2 == 0)\n", "    return len([x for x in xs if x % 2 == 0])\n"],
     lambda xs, K: sum(1 for x in xs if x % 2 == 0), [0]),
    (["count_negative", "num_negative"], ["Count the numbers below zero.", "How many items are negative."], "xs", [_ints],
     ["    return sum(1 for x in xs if x < 0)\n"], lambda xs, K: sum(1 for x in xs if x < 0), [0]),
    (["count_k", "occurrences"], ["Count how many times K appears in the list.", "Number of items equal to K."], "xs", [_ints],
     ["    return xs.count(K)\n", "    return sum(1 for x in xs if x == K)\n"], lambda xs, K: xs.count(K), [1, 2, 3]),
    (["sum_of_evens", "even_total"], ["Return the sum of the even numbers.", "Add up the even items."], "xs", [_ints],
     ["    return sum(x for x in xs if x % 2 == 0)\n"], lambda xs, K: sum(x for x in xs if x % 2 == 0), [0]),
    (["sum_of_abs", "abs_total"], ["Return the sum of the absolute values.", "Add up the sizes of the numbers."], "xs", [_ints],
     ["    return sum(abs(x) for x in xs)\n"], lambda xs, K: sum(abs(x) for x in xs), [0]),
    (["has_negative", "any_negative"], ["Return True if any number is negative.", "Check whether the list has a negative item."], "xs", [_ints],
     ["    return any(x < 0 for x in xs)\n"], lambda xs, K: any(x < 0 for x in xs), [0]),
    (["all_positive", "every_positive"], ["Return True if every number is positive.", "Check that all items are above zero."], "xs", [_ints],
     ["    return all(x > 0 for x in xs)\n"], lambda xs, K: all(x > 0 for x in xs), [0]),
    (["contains", "has_k"], ["Return True if K is in the list.", "Check whether the list contains K."], "xs", [_ints],
     ["    return K in xs\n"], lambda xs, K: K in xs, [0, 1, 5]),
    (["running_total", "prefix_sums"], ["Return the running totals of the list.", "Cumulative sums."], "xs", [_ints],
     ["    out = []\n    total = 0\n    for x in xs:\n        total += x\n        out.append(total)\n    return out\n"],
     lambda xs, K: [sum(xs[:i + 1]) for i in range(len(xs))], [0]),
    (["unique", "dedupe"], ["Return the distinct items, keeping their first order.", "Drop repeated items, keep order."], "xs", [_ints],
     ["    seen = []\n    for x in xs:\n        if x not in seen:\n            seen.append(x)\n    return seen\n",
      "    return list(dict.fromkeys(xs))\n"], lambda xs, K: list(dict.fromkeys(xs)), [0]),
    (["range_span", "spread"], ["Return the largest minus the smallest number.", "Difference between max and min."], "xs", [_ints1],
     ["    return max(xs) - min(xs)\n"], lambda xs, K: max(xs) - min(xs), [0]),
    (["take", "first_k"], ["Return the first K items.", "Keep the first K elements."], "xs", [_ints],
     ["    return xs[:K]\n"], lambda xs, K: xs[:K], [2, 3]),
    (["pairwise_sum", "add_lists"], ["Add two lists of the same length item by item.", "Element-wise sum."], "xs, ys", [_ints, None],
     ["    return [x + y for x, y in zip(xs, ys)]\n"], lambda xs, ys, K: [x + y for x, y in zip(xs, ys)], [0]),
    (["dot", "dot_product"], ["Return the dot product of two equal-length lists.", "Sum of the products of the pairs."], "xs, ys", [_ints, None],
     ["    return sum(x * y for x, y in zip(xs, ys))\n"], lambda xs, ys, K: sum(x * y for x, y in zip(xs, ys)), [0]),
    (["index_of", "position"], ["Return the index of K in the list, or -1.", "Where K first appears, -1 if missing."], "xs", [_ints],
     ["    return xs.index(K) if K in xs else -1\n",
      "    for i, x in enumerate(xs):\n        if x == K:\n            return i\n    return -1\n"],
     lambda xs, K: xs.index(K) if K in xs else -1, [0, 1, 2]),
    # strings
    (["to_upper", "upper", "capitalise_all"], ["Return the string in upper case.", "Convert the text to capitals."], "s", [_sentence],
     ["    return s.upper()\n"], lambda s, K: s.upper(), [0]),
    (["to_lower", "lower"], ["Return the string in lower case.", "Convert the text to small letters."], "s", [_word],
     ["    return s.lower()\n"], lambda s, K: s.lower(), [0]),
    (["reverse_text", "backwards_string"], ["Return the string reversed.", "Reverse the characters."], "s", [_word],
     ["    return s[::-1]\n", "    return ''.join(reversed(s))\n"], lambda s, K: s[::-1], [0]),
    (["last_char", "final_letter"], ["Return the last character of the string.", "Return s[-1]."], "s", [_word],
     ["    return s[-1]\n"], lambda s, K: s[-1], [0]),
    (["length", "str_len"], ["Return the number of characters.", "Length of the string."], "s", [_sentence],
     ["    return len(s)\n"], lambda s, K: len(s), [0]),
    (["repeat", "repeat_k"], ["Repeat the string K times.", "Return s joined to itself K times."], "s", [_word],
     ["    return s * K\n"], lambda s, K: s * K, [3, 4]),
    (["word_count", "count_words"], ["Return how many words the sentence has.", "Count the words."], "s", [_sentence],
     ["    return len(s.split())\n"], lambda s, K: len(s.split()), [0]),
    (["words", "split_words"], ["Split the sentence into a list of words.", "Return the words of s."], "s", [_sentence],
     ["    return s.split()\n"], lambda s, K: s.split(), [0]),
    (["first_word", "leading_word"], ["Return the first word of the sentence.", "Return the word s starts with."], "s", [_sentence],
     ["    return s.split()[0]\n"], lambda s, K: s.split()[0], [0]),
    (["title_case", "capitalize_each"], ["Capitalise the first letter of every word.", "Return the sentence in title case."], "s", [_sentence],
     ["    return s.title()\n", "    return ' '.join(w.capitalize() for w in s.split())\n"],
     lambda s, K: " ".join(w.capitalize() for w in s.split()), [0]),
    (["remove_spaces", "no_spaces"], ["Remove every space from the string.", "Return s without spaces."], "s", [_sentence],
     ["    return s.replace(' ', '')\n"], lambda s, K: s.replace(" ", ""), [0]),
    (["is_capitalised", "starts_upper"], ["Return True if the first letter is upper case."], "s", [_word],
     ["    return s[:1].isupper()\n"], lambda s, K: s[:1].isupper(), [0]),
    (["longest_word", "biggest_word"], ["Return the longest word in the sentence.", "Find the longest word."], "s", [_sentence],
     ["    return max(s.split(), key=len)\n"], lambda s, K: max(s.split(), key=len), [0]),
    (["join_words", "glue"], ["Join a list of words with spaces.", "Turn a list of words into a sentence."], "ws", [lambda r: _sentence(r).split()],
     ["    return ' '.join(ws)\n"], lambda ws, K: " ".join(ws), [0]),
    (["surround", "bracket"], ["Wrap the string in square brackets.", "Return '[' + s + ']'."], "s", [_word],
     ["    return '[' + s + ']'\n", "    return f'[{s}]'\n"], lambda s, K: "[" + s + "]", [0]),
    (["is_palindrome_word"], ["Return True if the word reads the same backwards.", "Check whether s is a palindrome."], "s", [_word],
     ["    return s == s[::-1]\n"], lambda s, K: s == s[::-1], [0]),
    (["count_upper", "capitals"], ["Count the upper-case letters.", "How many capitals the text has."], "s", [_word],
     ["    return sum(1 for ch in s if ch.isupper())\n"], lambda s, K: sum(1 for ch in s if ch.isupper()), [0]),
]


def _rename_k(text: str, k: int) -> str:
    return text.replace("K", str(k))


def _held_out_names() -> Set[str]:
    return set(['abs_diff', 'add_one', 'average', 'binary_search', 'clamp', 'count_char', 'count_pos', 'count_vowels', 'count_zeros', 'cube', 'decimal_to_binary', 'double_string', 'drop_negatives', 'ends_with_vowel', 'factorial', 'first_char', 'gcd', 'initials', 'is_anagram', 'is_even', 'is_prime', 'is_sorted', 'last_item', 'max_of_three', 'median', 'merge_dicts', 'middle', 'product', 'remove_duplicates', 'reverse_words', 'shout', 'square_all', 'sum_squares', 'triple', 'word_lengths'])


def generated(n: int, seed: int = 0, banned: Set[str] = frozenset()) -> Iterator[str]:
    r = random.Random(seed)
    made = 0
    while made < n:
        names, docs, params, makers, bodies, ref, ks = r.choice(TEMPLATES)
        name = r.choice(names)
        if name in banned:
            continue
        k = r.choice(ks)
        body = _rename_k(r.choice(bodies), k)
        doc = _rename_k(r.choice(docs), k)
        args_list = []
        for _ in range(r.randint(2, 4)):
            args = []
            for m in makers:
                args.append([r.randint(-9, 12) for _ in range(len(args[0]))] if m is None else m(r))
            args_list.append(args)
        header = f"def {name}({params}):"
        ns: dict = {}
        try:
            exec(header + "\n" + body, ns)  # our own template code, checked against its reference
            fn = ns[name]
            tests = []
            for args in args_list:
                got = fn(*args)
                if got != ref(*args, k):
                    raise ValueError("body disagrees with reference")
                tests.append(f"assert {name}({', '.join(repr(a) for a in args)}) == {got!r}")
        except Exception:
            continue
        made += 1
        yield format_doc(header, doc, body, tests if r.random() < 0.8 else [])



Pair = Tuple[str, str]
NAMES = ("Alice Bob Carol Dave Erin Frank Grace Heidi Ivan Judy Kim Leo Mia Nina Omar Pia Quinn Rex Sam Tara Uma Vic "
         "Wes Xena Yuri Zoe Flipper Tweety Nemo Felix Rover Bella Max Luna Oscar Ruby").split()
KINDS = ("bird cat dog fish mammal reptile animal plant robot insect whale shark eagle penguin snake frog tree "
         "vehicle car machine student teacher doctor artist musician wizard elf dragon").split()
PROPS = ("fly swim breathe-air lay-eggs bark purr glow sing think climb run hunt sleep-at-night eat-meat").split()
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def _num(r: random.Random, hi: int = 1000) -> int:
    return r.choice([r.randint(0, 20), r.randint(0, hi), r.randint(-hi, hi), r.randint(0, 10**6)])


def _fmt(x) -> str:
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return str(x.numerator)
        dec = x.numerator / x.denominator
        return f"{x.numerator}/{x.denominator}" if abs(dec * 1000 - round(dec * 1000)) > 1e-9 else _trim(dec)
    if isinstance(x, float):
        return _trim(x)
    return str(x)


def _trim(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


# ------------------------------------------------------------ mathematician
def arithmetic(r: random.Random) -> Pair:
    a, b = _num(r), _num(r)
    op = r.choice(["+", "-", "*", "*", "/"])
    if op == "/":
        b = b or 7
        val = Fraction(a, b)
    else:
        val = {"+": a + b, "-": a - b, "*": a * b}[op]
    word = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}[op]
    q = r.choice([f"what is {a} {op} {b}", f"what is {a} {word} {b}", f"calculate {a} {op} {b}", f"{a} {op} {b} = ?"])
    return q, _fmt(val)


def powers(r: random.Random) -> Pair:
    a, b = r.randint(2, 15), r.randint(2, 20)
    return r.choice([f"what is {a}^{b}", f"{a} to the power of {b}", f"what is {a}**{b}"]), str(a ** b)


def percent(r: random.Random) -> Pair:
    p, n = r.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 75, r.randint(1, 99)]), r.randint(1, 2000)
    return r.choice([f"what is {p}% of {n}", f"{p} percent of {n}"]), _fmt(Fraction(p * n, 100))


def gcd_lcm(r: random.Random) -> Pair:
    a, b = r.randint(2, 500), r.randint(2, 500)
    if r.random() < 0.5:
        return r.choice([f"gcd of {a} and {b}", f"greatest common divisor of {a} and {b}"]), str(math.gcd(a, b))
    return r.choice([f"lcm of {a} and {b}", f"least common multiple of {a} and {b}"]), str(a * b // math.gcd(a, b))


def _is_prime(n: int) -> bool:
    return n > 1 and all(n % p for p in range(2, math.isqrt(n) + 1))


def primality(r: random.Random) -> Pair:
    n = r.randint(2, 100000)
    return r.choice([f"is {n} prime?", f"is {n} a prime number"]), "yes" if _is_prime(n) else "no"


def factorise(r: random.Random) -> Pair:
    n = r.randint(2, 50000)
    out, m, p = [], n, 2
    while p * p <= m:
        while m % p == 0:
            out.append(p)
            m //= p
        p += 1
    if m > 1:
        out.append(m)
    return r.choice([f"prime factors of {n}", f"factorize {n}"]), " × ".join(map(str, out))


def linear(r: random.Random) -> Pair:
    a = r.choice([i for i in range(-12, 13) if i])
    x, b = r.randint(-30, 30), r.randint(-50, 50)
    c = a * x + b
    lhs = f"{a}x" if a not in (1, -1) else ("x" if a == 1 else "-x")
    lhs += f" + {b}" if b > 0 else (f" - {-b}" if b < 0 else "")
    return f"solve {lhs} = {c}", f"x = {x}"


def quadratic(r: random.Random) -> Pair:
    p, q = r.randint(-12, 12), r.randint(-12, 12)
    b, c = -(p + q), p * q
    poly = "x^2" + (f" + {b}x" if b > 0 else f" - {-b}x" if b < 0 else "") + (f" + {c}" if c > 0 else f" - {-c}" if c < 0 else "")
    roots = sorted({p, q})
    return f"solve {poly} = 0", " or ".join(f"x = {v}" for v in roots)


def system2(r: random.Random) -> Pair:
    x, y = r.randint(-20, 20), r.randint(-20, 20)
    a, b, c, d = (r.choice([1, 2, 3, -1]) for _ in range(4))
    if a * d - b * c == 0:
        c, d = 1, -1
        if a * d - b * c == 0:
            a, b = 1, 1
    def eq(p, s):
        return f"{p}x" + (f" + {s}y" if s >= 0 else f" - {-s}y")
    return f"solve {eq(a, b)} = {a * x + b * y}, {eq(c, d)} = {c * x + d * y}".replace("1x", "x").replace(" 1y", " y"), \
        f"x = {x}, y = {y}"


_UNITS = [("km", "miles", Fraction(1000, 1609344) * 1000), ("miles", "km", Fraction(1609344, 1000000)),
          ("kg", "pounds", Fraction(100000000, 45359237)), ("pounds", "kg", Fraction(45359237, 100000000)),
          ("meters", "feet", Fraction(10000, 3048)), ("feet", "meters", Fraction(3048, 10000)),
          ("inches", "cm", Fraction(254, 100)), ("cm", "inches", Fraction(100, 254)),
          ("hours", "minutes", Fraction(60)), ("days", "hours", Fraction(24)), ("liters", "milliliters", Fraction(1000)),
          ("GB", "MB", Fraction(1000))]


def units(r: random.Random) -> Pair:
    if r.random() < 0.3:
        v = r.randint(-40, 212)
        if r.random() < 0.5:
            return f"convert {v} F to C", _trim(float((Fraction(v) - 32) * Fraction(5, 9))) + " °C"
        return f"convert {v} C to F", _trim(float(Fraction(v) * Fraction(9, 5) + 32)) + " °F"
    src, dst, f = r.choice(_UNITS)
    v = r.choice([r.randint(1, 100), r.randint(1, 1000)])
    return r.choice([f"convert {v} {src} to {dst}", f"how many {dst} is {v} {src}", f"{v} {src} in {dst}"]), \
        f"{_trim(float(v * f))} {dst}"


def _date(r: random.Random) -> dt.date:
    return dt.date(1900, 1, 1) + dt.timedelta(days=r.randint(0, 200 * 365))


def dates(r: random.Random) -> Pair:
    d = _date(r)
    k = r.random()
    if k < 0.4:
        return f"what day of the week was {MONTHS[d.month - 1]} {d.day}, {d.year}", WEEKDAYS[d.weekday()]
    if k < 0.7:
        e = d + dt.timedelta(days=r.randint(1, 2000))
        return f"how many days between {d.isoformat()} and {e.isoformat()}", f"{(e - d).days} days"
    n = r.randint(1, 1000)
    return f"what date is {n} days after {d.isoformat()}", (d + dt.timedelta(days=n)).isoformat()


def leap(r: random.Random) -> Pair:
    y = r.randint(1600, 2400)
    ok = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    return f"is {y} a leap year?", "yes" if ok else "no"


def sequences(r: random.Random) -> Pair:
    k = r.random()
    n = r.randint(4, 6)
    if k < 0.35:
        a, d = r.randint(-20, 30), r.choice([i for i in range(-9, 13) if i])
        seq = [a + d * i for i in range(n + 1)]
    elif k < 0.6:
        a, q = r.randint(1, 5), r.choice([2, 3, -2, 4])
        seq = [a * q ** i for i in range(n + 1)]
    elif k < 0.8:
        p, c = r.choice([2, 3]), r.randint(0, 5)
        seq = [i ** p + c for i in range(1, n + 2)]
    else:
        a, b = r.randint(0, 5), r.randint(1, 5)
        seq = [a, b]
        while len(seq) < n + 1:
            seq.append(seq[-1] + seq[-2])
    return ", ".join(map(str, seq[:-1])) + ", ?", str(seq[-1])


# ---------------------------------------------------------------- reasoner
def _an(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


def syllogism(r: random.Random) -> Pair:
    a, b, c = r.sample(KINDS, 3)
    who = r.choice(NAMES)
    k = r.random()
    if k < 0.4:
        return (f"All {a}s are {b}s. All {b}s are {c}s. {who} is {_an(a)}. Is {who} {_an(c)}?",
                f"Yes, {who} is {_an(c)}.")
    if k < 0.7:
        prop = r.choice(PROPS).replace("-", " ")
        return (f"If something is {_an(a)} then it can {prop}. {who} cannot {prop}. Is {who} {_an(a)}?",
                f"No, {who} is not {_an(a)}.")
    return (f"All {a}s are {b}s. {who} is {_an(b)}. Is {who} {_an(a)}?", "Cannot be determined from the premises.")


# ------------------------------------------------------------------- coder
def code_task(r: random.Random, banned=frozenset()) -> Pair:
    doc = next(generated(1, seed=r.randrange(2**31), banned=banned))
    lines = [l for l in doc.splitlines() if not l.startswith("# assert")]
    tests = "\n".join(l[2:] for l in doc.splitlines() if l.startswith("# assert"))
    head, docline = lines[0], lines[1].strip().strip('"')
    name = head[4:head.index("(")]
    q = r.choice([f"write a python function {name} that does this: {docline}",
                  f"{docline} Write it as a function called {name}.", f"{head}\n    \"\"\"{docline}\"\"\""])
    if tests:
        q += "\n" + tests  # the tests make each task distinct and teach test-driven answers
    return q, "\n".join(lines) + "\n"


GENERATORS: Dict[str, Tuple[str, Callable, float]] = {
    "arithmetic": ("mathematician", arithmetic, 0.14),
    "powers": ("mathematician", powers, 0.03),
    "percent": ("mathematician", percent, 0.04),
    "gcd_lcm": ("mathematician", gcd_lcm, 0.04),
    "primality": ("mathematician", primality, 0.04),
    "factorise": ("mathematician", factorise, 0.03),
    "linear": ("mathematician", linear, 0.06),
    "quadratic": ("mathematician", quadratic, 0.05),
    "system2": ("mathematician", system2, 0.03),
    "units": ("mathematician", units, 0.07),
    "dates": ("mathematician", dates, 0.07),
    "leap": ("mathematician", leap, 0.02),
    "sequences": ("mathematician", sequences, 0.06),
    "syllogism": ("reasoner", syllogism, 0.08),
    "code": ("coder", code_task, 0.24),
}


def stream(n: int, seed: int = 0) -> Iterator[dict]:
    r = random.Random(seed)
    kinds = list(GENERATORS)
    weights = [GENERATORS[k][2] for k in kinds]
    banned = _held_out_names()
    seen = set()  # only distinct questions; small spaces (leap years, ...) simply run out
    made = 0
    while made < n:
        kind = r.choices(kinds, weights)[0]
        agent, gen, _ = GENERATORS[kind]
        q, a = gen(r, banned) if kind == "code" else gen(r)
        h = hash(q)
        if h in seen:
            continue
        seen.add(h)
        made += 1
        yield {"q": q, "a": a, "agent": agent, "kind": kind}


def write(n: int, out: Path, seed: int = 0, shard_size: int = 250000) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    f = None
    for i, item in enumerate(stream(n, seed)):
        if i % shard_size == 0:
            if f:
                f.close()
            f = gzip.open(out / f"qa-{i // shard_size:03d}.jsonl.gz", "wt", encoding="utf-8")
        f.write(json.dumps(item) + "\n")
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1
    if f:
        f.close()
    return {"pairs": n, "distinct_questions": n, "by_kind": counts}


def read(folder: Path) -> Iterator[dict]:
    for shard in sorted(Path(folder).glob("qa-*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def as_document(item: dict) -> str:
    return f"Q: {item['q']}\nA: {item['a']}\n"




if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    with open(a.out, "w") as f:
        for it in stream(a.n, a.seed): f.write(json.dumps(it) + "\n")
