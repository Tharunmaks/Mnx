"""Build the "expert stream": a training corpus made only of verified code.

The neural coder is trained on what the expert agents can vouch for:

1. **Recipes** from the coding agent (``data/code``), each with its tests.
2. **Generated tasks.** Templates of small functions (numbers, lists,
   strings) with varied names, docstrings and implementations. Every one is
   executed on random inputs; the outputs become its asserts, and a body that
   disagrees with its reference implementation is dropped.
3. **Standard library functions.** Short functions with docstrings parsed
   out of the Python installation's own source, as real-world code.

Each document has the shape the neural proposer is prompted with::

    # assert add_one([1, 2]) == [2, 3]
    def add_one(xs):
        \"\"\"Add 1 to every number in the list.\"\"\"
        return [x + 1 for x in xs]

Function names that appear in the evaluation sets are never generated, so
the holdout tasks are not copied into the training data::

    python -m mnx.agent.expert_stream --out runs/expert_stream.txt
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
import sysconfig
import textwrap
from pathlib import Path
from typing import Callable, Iterator, List, Sequence, Set, Tuple

from mnx.agent.skills.code import Recipe, default_recipe_dir

DOC_SEP = "\n<|doc|>\n"
ROOT = Path(__file__).resolve().parents[2]


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
    names: Set[str] = set()
    for path in (ROOT / "data" / "eval").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            item = json.loads(line)
            for t in item.get("tests") or []:
                head = t.replace("assert ", "").replace("not ", "").strip()
                names.add(head.split("(")[0].strip())
    return names


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


def recipes(repeat: int = 20) -> Iterator[str]:
    """Recipe files, several times each (they are the highest-quality data)."""
    for path in sorted(default_recipe_dir().glob("*.py")):
        rec = Recipe.parse(path.read_text(), str(path))
        tests = [t for t in rec.tests if t.startswith("assert")][:3]
        for i in range(repeat):
            yield ("".join(f"# {t}\n" for t in tests) if i % 2 == 0 else "") + rec.code


def stdlib_functions(limit: int = 20000, max_lines: int = 20) -> Iterator[str]:
    """Short documented functions from the running Python's standard library."""
    root = Path(sysconfig.get_paths()["stdlib"])
    count = 0
    for path in sorted(root.rglob("*.py")):
        if any(part in ("test", "tests", "idlelib", "lib2to3", "site-packages", "turtledemo") for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (UnicodeDecodeError, SyntaxError, ValueError):
            continue
        lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not ast.get_docstring(node) or node.decorator_list:
                continue
            if node.end_lineno - node.lineno + 1 > max_lines:
                continue
            text = textwrap.dedent("\n".join(lines[node.lineno - 1:node.end_lineno])) + "\n"
            if text.startswith("def ") and "self" not in text.split("\n", 1)[0]:
                count += 1
                yield text
                if count >= limit:
                    return


def build(out: Path, n_generated: int = 60000, recipe_repeat: int = 20, stdlib_limit: int = 20000, seed: int = 0) -> dict:
    banned = _held_out_names()
    parts = {"recipes": list(recipes(recipe_repeat)), "generated": list(generated(n_generated, seed, banned)),
             "stdlib": list(stdlib_functions(stdlib_limit))}
    docs = [d for ds in parts.values() for d in ds]
    random.Random(seed).shuffle(docs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(DOC_SEP.join(docs), encoding="utf-8")
    stats = {k: len(v) for k, v in parts.items()}
    stats["chars"] = sum(map(len, docs))
    stats["banned_names"] = len(banned)
    return stats


def main(argv: Sequence[str] = None) -> int:
    p = argparse.ArgumentParser(description="Write the expert-stream training corpus.")
    p.add_argument("--out", default="runs/expert_stream.txt")
    p.add_argument("--generated", type=int, default=60000)
    p.add_argument("--recipe-repeat", type=int, default=20)
    p.add_argument("--stdlib", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    print(json.dumps(build(Path(a.out), a.generated, a.recipe_repeat, a.stdlib, a.seed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
