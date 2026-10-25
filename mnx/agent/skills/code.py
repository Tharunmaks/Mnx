"""The coding agent: write, run and check Python, and only ship tested code.

Strategies, tried in order, every one ending in a test run:

1. **Recipe retrieval.** ``data/code/*.py`` (plus recipes MNX learned and
   saved itself) are small, tested functions. A request like "write a
   function that checks if a string is a palindrome" is matched with BM25,
   the recipe is renamed if you asked for a name, and its tests (or yours)
   are run in a sandboxed subprocess before it is returned.
2. **Your tests decide.** If you give a ``def`` header and ``assert`` lines,
   every candidate (recipes, programs synthesised from the asserts, optional
   neural samples) is run against *your* asserts, and only a passing one is
   returned.
3. **Run / check.** "run this: ```...```" executes code in the sandbox and
   shows the output; "check/debug this code" reports syntax errors, names
   that are never defined, and failing asserts.

Python only. Code that passes is saved as a new recipe (learning from use).
"""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from mnx.agent.base import Answer, Skill
from mnx.agent.retrieval import BM25Index, Chunk, tokenize
from mnx.agent.skills.neural import run_tests

CODE_STOP = set(tokenize("write create make implement build give generate code python function program script method "
                         "class that which for to a an the me please can you could would takes take returns return "
                         "given input output using use in with of how do i def compute computes calculate calculates find finds get gets "
                         "check checks determine determines return returns some"))

#: abbreviations used in function names → the words people type
_NAME_WORDS = {"int": "integer", "str": "string", "dict": "dictionary", "num": "number", "nums": "numbers",
               "len": "length", "dfs": "depth first search", "bfs": "breadth first search", "lcs": "longest common subsequence",
               "gcd": "greatest common divisor", "lcm": "least common multiple", "hcf": "highest common factor"}


def name_words(name: str) -> str:
    parts = [p for p in re.split(r"[_\W]+", name) if p]
    return " ".join(_NAME_WORDS.get(p.lower(), p) for p in parts)


def default_recipe_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "code"


@dataclass
class Recipe:
    name: str
    description: str
    tags: str
    code: str
    tests: List[str]
    path: str

    @classmethod
    def parse(cls, text: str, path: str) -> "Recipe":
        doc_m = re.match(r'\s*"""(.*?)"""\s*\n', text, re.S)
        doc = doc_m.group(1).strip() if doc_m else ""
        body = text[doc_m.end():] if doc_m else text
        code, _, tests = body.partition("# tests")
        desc, _, tags = doc.partition("tags:")
        tree = ast.parse(code)
        names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        main = Path(path).stem
        name = next((n for n in names if n.lower() == main), names[-1] if names else main)
        return cls(name, desc.strip(), tags.strip(), code.strip() + "\n", [t for t in tests.strip().splitlines()], path)

    def source(self) -> str:
        return f'"""{self.description}\n\ntags: {self.tags}\n"""\n\n{self.code}\n\n# tests\n' + "\n".join(self.tests) + "\n"


class RecipeBook:
    def __init__(self, folders: Iterable[Path]):
        self.recipes: List[Recipe] = []
        self.index = BM25Index()
        self.folders = [Path(f) for f in folders]
        for folder in self.folders:
            if folder.exists():
                for path in sorted(folder.glob("*.py")):
                    try:
                        self.add(Recipe.parse(path.read_text(encoding="utf-8"), str(path)))
                    except (SyntaxError, ValueError):
                        continue

    def add(self, recipe: Recipe) -> None:
        self.recipes.append(recipe)
        words = name_words(recipe.name)
        text = f"{words} {words} {recipe.description} {recipe.tags}"
        self.index.add(Chunk(text, str(len(self.recipes) - 1), recipe.name))

    def search(self, query: str, k: int = 5) -> List[Tuple[float, Recipe]]:
        q = " ".join(t for t in re.findall(r"[a-z0-9]+", query.lower()) if stem_ok(t))
        return [(s, self.recipes[int(c.source)]) for s, c in self.index.search(q, k)]

    def coverage(self, query: str, recipe: Recipe) -> float:
        """Share of the request's content words (IDF-weighted) the recipe covers."""
        q = [t for t in dict.fromkeys(tokenize(query)) if t not in CODE_STOP]
        if not q:
            return 0.0
        have = set(tokenize(name_words(recipe.name) + " " + recipe.description + " " + recipe.tags))
        w = {t: self.index.idf(t) for t in q}
        return sum(v for t, v in w.items() if t in have) / (sum(w.values()) or 1.0)


def stem_ok(tok: str) -> bool:
    return bool(tokenize(tok)) and tokenize(tok)[0] not in CODE_STOP


# ----------------------------------------------------------- code helpers
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> Optional[str]:
    m = _FENCE.search(text)
    if m:
        return m.group(1)
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if re.match(r"\s*(def |class |import |from \S+ import |for |while |print\(|[a-zA-Z_]\w* = )", l):
            return "\n".join(lines[i:])
    return None


def rename(code: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, code)


def undefined_names(code: str) -> List[Tuple[str, int]]:
    """Names that are read but never bound anywhere (scope-insensitive)."""
    tree = ast.parse(code)
    bound = set(dir(builtins)) | {"__name__", "__file__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    seen, out = set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in bound and node.id not in seen:
            seen.add(node.id)
            out.append((node.id, node.lineno))
    return sorted(out, key=lambda x: x[1])


def run_code(code: str, timeout: float = 5.0) -> Tuple[bool, str]:
    """Run code in the sandbox; returns (ok, combined output)."""
    import subprocess
    import sys

    try:
        proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout:.0f}s"
    out = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, out[-3000:]


def parse_asserts(name: str, tests: Sequence[str]) -> List[Tuple[tuple, object]]:
    """``assert f(1, 2) == 3`` → ``((1, 2), 3)`` for literal arguments."""
    out = []
    for t in tests:
        try:
            node = ast.parse(t.strip()).body[0]
        except SyntaxError:
            continue
        if not (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq)):
            continue
        call, rhs = node.test.left, node.test.comparators[0]
        if isinstance(rhs, ast.Call):
            call, rhs = rhs, call
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name):
            continue
        try:
            args = tuple(ast.literal_eval(a) for a in call.args)
            out.append((args, ast.literal_eval(rhs)))
        except ValueError:
            continue
    return out


def _probes(examples: List[Tuple[tuple, object]]) -> List[tuple]:
    """Fresh inputs of the same types, used to test whether candidate programs agree."""
    def pool(v):
        if isinstance(v, bool):
            return [True, False]
        if isinstance(v, int):
            return [-3, 0, 1, 4, 7, 12, 25]
        if isinstance(v, str):
            return ["", "a", "Hello World", "racecar"]
        if isinstance(v, list) and all(isinstance(e, int) for e in v):
            return [[], [0], [3, 1, 2], [5, -4, 8, 8]]
        return []
    first = examples[0][0]
    pools = [pool(a) for a in first]
    if not pools or any(not p for p in pools):
        return []
    seen = [e[0] for e in examples]
    out = [tuple(p[i % len(p)] for p in pools) for i in range(max(len(p) for p in pools))]
    return [o for o in out if o not in seen]


#: operations in synthesised code → words that describe them
_OP_WORDS = [
    (r"\bsum\(", "sum total add"), (r"\bmax\(", "max maximum largest biggest greatest highest"),
    (r"\bmin\(", "min minimum smallest least lowest"), (r"\blen\(", "length count number many size long"),
    (r"\bsorted\(", "sort sorted order ascending"), (r"\[::-1\]", "reverse reversed backwards"),
    (r"% 2 == 0", "even"), (r"% 2 == 1|% 2 != 0", "odd"), (r"< 0\b", "negative below"), (r"> 0\b", "positive"),
    (r"\*\* 2|(\b\w+) \* \1\b", "square squares squared"), (r"\*\* 3", "cube cubed"),
    (r"\.upper\(", "upper uppercase capital case"), (r"\.lower\(", "lower lowercase case"), (r"\.split\(", "word words split"),
    (r"\+ 1\b", "increment incremented incr add one next"), (r"\babs\(", "absolute distance difference"),
    (r"fromkeys|\bset\(", "unique distinct duplicate duplicates"), (r"\bprod\(", "product multiply"),
    (r"\* 2\b", "double twice"), (r"\.count\(", "count occurrences times"), (r"\.join\(", "join concatenate"),
    (r"\[0\]", "first"), (r"\[-1\]", "last"), (r"\.title\(|\.capitalize\(", "capitalize capitalise title"),
    (r"\.strip\(", "strip trim whitespace"), (r"\.replace\(", "replace remove"), (r"\.isdigit\(", "digit digits"),
]


def docstring_fit(code: str, words: str) -> int:
    """How well a program's operations match the description: +1 per described op, -1 per op or
    constant (3 or more) that the description never mentions."""
    vocab = set(re.findall(r"[a-z]+", words))
    score = sum(1 if vocab & set(ws.split()) else -1 for pat, ws in _OP_WORDS if re.search(pat, code))
    for c in set(re.findall(r"(?<![\w.])(\d+)(?![\w.])", code)):
        if int(c) >= 3 and c not in words:
            score -= 1
    return score


def _call_args(name: str, tests: Sequence[str]) -> List[tuple]:
    """Literal argument tuples of every ``name(...)`` call in the asserts."""
    out = []
    for t in tests:
        try:
            tree = ast.parse(t.strip())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
                try:
                    out.append(tuple(ast.literal_eval(a) for a in node.args))
                except ValueError:
                    continue
    return out


def _behaviour(code: str, name: str, probes: Sequence[tuple]) -> str:
    """What a candidate returns on the probe inputs (errors included), as one comparable string."""
    import subprocess
    import sys

    script = (code + "\nres = []\nfor a in " + repr(list(probes)) + ":\n    try:\n        res.append(repr(" + name +
              "(*a)))\n    except Exception as e:\n        res.append(type(e).__name__)\nprint(res)\n")
    try:
        proc = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return "timeout"
    return proc.stdout.strip() or proc.stderr.strip()[-80:]


_CODE_VERB = re.compile(r"\b(write|create|make|implement|build|give me|generate|show me|how do i|how to)\b", re.I)
_CODE_NOUN = re.compile(r"\b(code|function|program|script|class|python|algorithm|method|decorator|snippet)\b", re.I)


class CodeSkill(Skill):
    name = "code"
    description = "write, run and check Python; only tested code is returned"

    def __init__(self, recipe_dirs: Optional[Iterable[Path]] = None, learned_dir: Optional[Path] = None,
                 proposers: Sequence[Callable[[str, List[str]], List[str]]] = ()):
        dirs = list(recipe_dirs) if recipe_dirs is not None else [default_recipe_dir()]
        self.learned_dir = Path(learned_dir) if learned_dir else None
        if self.learned_dir:
            dirs.append(self.learned_dir)
        self.book = RecipeBook(dirs)
        self.proposers = list(proposers)  # e.g. neural sampler: (prompt, tests) -> [code]

    # ------------------------------------------------------------- entry
    def answer(self, question: str) -> Optional[Answer]:
        q = question.strip()
        low = q.lower()
        code = extract_code(q)
        if code and re.match(r"^\s*(please\s+)?(run|execute)\b", low):
            return self._run(code)
        if code and re.match(r"^\s*(please\s+)?(check|debug|fix|review|lint|what's wrong|what is wrong|find the bug)", low):
            return self._check(code)
        header = re.search(r"^\s*def\s+(\w+)\s*\(([^)]*)\)", q, re.M)
        tests = [l.strip() for l in q.splitlines() if l.strip().startswith("assert ")]
        if header and tests:
            return self._solve_with_tests(q, header.group(1), header.group(2), tests)
        if not code and (_CODE_NOUN.search(low) or _CODE_VERB.search(low)):
            return self._write(q, explicit=bool(_CODE_NOUN.search(low)))
        return None

    # ------------------------------------------------------------ write
    def _write(self, request: str, explicit: bool = True) -> Optional[Answer]:
        hits = self.book.search(request, k=5)
        if not hits:
            return None
        req = set(tokenize(request))

        req_seq = tokenize(request)

        def name_fit(r: Recipe) -> float:  # "merge sort" should beat "merge sorted lists"
            parts = tokenize(name_words(r.name))
            return sum(p in req for p in parts) / max(1, len(parts))

        def wrong_direction(r: Recipe) -> bool:
            """"decimal to binary" must not match binary_to_decimal."""
            parts = tokenize(name_words(r.name))
            if "to" not in r.name.lower().split("_"):
                return False
            pos = [req_seq.index(p) for p in parts if p in req_seq]
            return len(pos) >= 2 and pos != sorted(pos)

        scored = []
        for s_, r in hits:
            cov_r = 0.0 if wrong_direction(r) else self.book.coverage(request, r)
            scored.append((cov_r, cov_r + 0.2 * name_fit(r), s_, r))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        cov, _, _, recipe = scored[0]
        trace = [f"recipe search: best match '{recipe.name}' covers {cov:.0%} of the request"]
        trace += [f"  runner-up '{r.name}' ({c:.0%})" for c, _, _, r in scored[1:3]]
        if cov < 0.6:
            if not explicit:  # "make a cake" is not a coding request
                return None
            return Answer("I don't have tested code for that yet. Give me a `def` header plus a few `assert` lines "
                          "and I'll search for a program that passes them.", self.name, 0.3, False, trace, failed=True)
        code, tests = recipe.code, list(recipe.tests)
        m = re.search(r"\b(?:called|named)\s+[`'\"]?([A-Za-z_]\w*)", request)
        if m and m.group(1) != recipe.name:
            code = rename(code, recipe.name, m.group(1))
            tests = [rename(t, recipe.name, m.group(1)) for t in tests]
            trace.append(f"renamed {recipe.name} → {m.group(1)}")
        ok, err = run_tests(code, tests, timeout=10)
        trace.append(f"ran {len(tests)} test line(s) in a sandboxed process: {'passed' if ok else 'FAILED ' + err}")
        if not ok:
            return Answer(f"I found a recipe but it failed its tests ({err}), so I won't return it.", self.name, 0.2, False, trace, failed=True)
        shown_tests = "\n".join(tests)
        text = f"```python\n{code}```\nTests it passed:\n```python\n{shown_tests}\n```"
        return Answer(text, self.name, 0.5 + 0.45 * cov, True, trace, [Path(recipe.path).name])

    # ------------------------------------------------------- with tests
    def _solve_with_tests(self, q: str, name: str, params: str, tests: List[str]) -> Answer:
        trace = [f"target: {name}({params}) must pass {len(tests)} assert(s)"]
        doc = " ".join(l for l in q.splitlines() if not l.strip().startswith("assert "))
        # 1. recipes, renamed to the requested function name
        for _, r in self.book.search(doc + " " + " ".join(re.split(r"[_\W]+", name)), k=6):
            code = rename(r.code, r.name, name)
            ok, err = run_tests(code, tests, timeout=10)
            trace.append(f"recipe '{r.name}': {'passed' if ok else 'failed'}")
            if ok:
                return self._ship(code, tests, trace, f"adapted the tested recipe '{r.name}'", learn_from=q)
        # 2. program synthesis from the asserts
        examples = parse_asserts(name, tests)
        if len(examples) >= 2:
            from mnx.agent.skills.synth import ARG_NAMES, synthesize

            arg_names = [p.split(":")[0].split("=")[0].strip() for p in params.split(",") if p.strip()]
            ambiguous = False
            probes = _probes(examples)
            res = synthesize(examples, probes, time_limit=4.0)
            if res is not None and len(arg_names) == len(examples[0][0]):
                if probes:
                    from mnx.agent.skills.synth import vote

                    agree = min(vote(res.hits, i)[1] for i in range(len(probes)))
                    trace.append(f"{len(res.hits)} programs fit; on {len(probes)} new probe inputs they agree ≥{agree:.0%}")
                    expr_code = res.code
                    if agree < 0.8 and ('"""' in q or "'''" in q):
                        # let the description vote too: programs whose operations match its words count for more
                        words = (doc + " " + name.replace("_", " ")).lower()
                        hits = [(c, size - 2 * docstring_fit(c, words), outs) for c, size, outs in res.hits]
                        agree = min(vote(hits, i)[1] for i in range(len(probes)))
                        outs = [vote(hits, i)[0] for i in range(len(probes))]
                        fitting = sorted((h for h in hits if list(h[2]) == outs), key=lambda h: h[1])
                        if agree >= 0.8 and fitting:
                            expr_code = fitting[0][0]
                            trace.append(f"the description breaks the tie: weighted by it, the programs agree ≥{agree:.0%}")
                        else:
                            ambiguous = True
                            trace.append("ambiguous: different programs fit your asserts; add more asserts to pin it down")
                    elif agree < 0.8:
                        ambiguous = True
                        trace.append("ambiguous: different programs fit your asserts; add more asserts to pin it down")
                else:
                    expr_code = res.code
                described = '"""' in q or "'''" in q
                if described and docstring_fit(expr_code, (doc + " " + name.replace("_", " ")).lower()) < 0:
                    # the program uses operations the description never mentions: likely a coincidence
                    ambiguous = True
                    trace.append("the program uses operations your description doesn't mention, so it may fit by coincidence")
                expr = expr_code
                for dsl, real in zip(ARG_NAMES, arg_names):
                    expr = re.sub(rf"\b{dsl}\b", real, expr)
                code = f"def {name}({', '.join(arg_names)}):\n    return {expr}\n"
                ok, err = run_tests(code, tests, timeout=10)
                trace.append(f"synthesised from {len(examples)} examples: {'passed' if ok else 'failed ' + err}")
                if ok and ambiguous and self.proposers:
                    # the asserts don't pin the program down; a model that read the docstring breaks the tie
                    neural = self._propose(q, tests, trace)
                    if neural:
                        return self._ship(neural, tests, trace, "neural proposal (it reads the docstring) that passed your asserts",
                                          learn_from=q)
                if ok and ambiguous:
                    # several programs fit and disagree elsewhere: a guess, so it is not shipped as an answer
                    trace.append("not returned as an answer: the asserts don't decide between the programs that fit")
                    return Answer("I'm not sure. Several different programs pass your asserts and disagree on other "
                                  f"inputs. My best guess, checked only against your asserts:\n```python\n{code}```\n"
                                  "Add a few more asserts (especially edge cases) and I can pin it down.",
                                  self.name, 0.4, False, trace, failed=True)
                if ok:
                    return self._ship(code, tests, trace, "searched for the smallest program that fits your asserts",
                                      learn_from=q)
            else:
                trace.append("program search found nothing within its limits")
        # 3. other proposers (e.g. the neural model)
        code = self._propose(q, tests, trace)
        if code:
            return self._ship(code, tests, trace, "neural proposal that passed your asserts", learn_from=q)
        return Answer("Nothing I tried passes all of your asserts, so I won't hand you untested code. "
                      "Adding more examples or a simpler spec usually helps.", self.name, 0.35, False, trace, failed=True)

    def _propose(self, q: str, tests: List[str], trace: List[str]) -> Optional[str]:
        """Sample proposals, keep those that pass, and return one only if most of them agree.

        A small model can pass your asserts by accident (``return n == 4``), so the passing
        samples are run on fresh probe inputs and grouped by behaviour (AlphaCode-style
        clustering). The largest group must hold at least two samples and half of the passes.
        """
        prompt = "\n".join(l for l in q.splitlines() if not l.strip().startswith("assert ")).rstrip() + "\n"
        passing: List[str] = []
        for propose in self.proposers:
            for code in propose(prompt, tests):
                if code not in passing and run_tests(code, tests, timeout=10)[0]:
                    passing.append(code)
        if not passing:
            trace.append("no neural proposal passed")
            return None
        m = re.search(r"def\s+(\w+)\s*\(", prompt)
        probes = _probes([(args, None) for args in _call_args(m.group(1), tests)]) if m else []
        if not probes:
            trace.append(f"{len(passing)} neural proposal(s) passed, but there are no probe inputs to check them")
            return passing[0] if len(passing) >= 2 else None
        groups: dict = {}
        for code in passing:
            groups.setdefault(_behaviour(code, m.group(1), probes), []).append(code)
        best = max(groups.values(), key=len)
        trace.append(f"{len(passing)} neural proposals passed; the largest group that behaves the same on "
                     f"{len(probes)} probe inputs has {len(best)}")
        if len(best) >= 2 and len(best) * 2 >= len(passing):
            return best[0]
        trace.append("the passing proposals disagree on new inputs, so none is returned")
        return None

    def _ship(self, code: str, tests: List[str], trace: List[str], how: str, learn_from: str, note: str = "") -> Answer:
        trace.append(how)
        saved = None if note else self.learn(code, tests, learn_from)  # don't learn from ambiguous fits
        if saved:
            trace.append(f"saved as a new recipe: {saved}")
        return Answer(f"```python\n{code}```\nPasses all {len(tests)} of your assert(s).{note}", self.name,
                      0.75 if note else 0.95, True, trace)

    def learn(self, code: str, tests: List[str], request: str) -> Optional[str]:
        """Store verified code as a new recipe so similar requests are instant next time."""
        if not self.learned_dir:
            return None
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        if not names:
            return None
        name = names[0]
        if any(r.name == name and r.code.strip() == code.strip() for r in self.book.recipes):
            return None
        desc = next((l.strip().strip('"').strip() for l in request.splitlines() if l.strip().startswith(('"""', "'''", "#"))), "")
        recipe = Recipe(name, desc or f"Learned function {name}.", " ".join(re.split(r"[_\W]+", name)), code, tests,
                        str(self.learned_dir / f"{name.lower()}.py"))
        self.learned_dir.mkdir(parents=True, exist_ok=True)
        Path(recipe.path).write_text(recipe.source(), encoding="utf-8")
        self.book.add(recipe)
        return Path(recipe.path).name

    # --------------------------------------------------------- run/check
    def _run(self, code: str) -> Answer:
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return Answer(f"Syntax error on line {exc.lineno}: {exc.msg}", self.name, 0.95, True, ["parsed with ast"])
        ok, out = run_code(code)
        status = "ran successfully" if ok else "failed"
        return Answer(f"The code {status}. Output:\n```\n{out or '(no output)'}\n```", self.name, 0.95, True,
                      ["executed in an isolated subprocess (python -I) with a 5 s timeout"])

    def _check(self, code: str) -> Answer:
        problems: List[str] = []
        trace = ["static checks with ast, then a sandboxed run"]
        try:
            ast.parse(code)
        except SyntaxError as exc:
            line = (exc.text or "").rstrip()
            return Answer(f"Syntax error on line {exc.lineno}: {exc.msg}\n```\n{line}\n```", self.name, 0.95, True, trace)
        for name, line in undefined_names(code):
            problems.append(f"line {line}: `{name}` is used but never defined")
        ok, out = run_code(code)
        if not ok:
            last = out.splitlines()[-1] if out else "error"
            m = re.findall(r'line (\d+)', out)
            where = f" (line {m[-1]})" if m else ""
            problems.append(f"running it raises {last}{where}")
        if not problems:
            return Answer("I found no problems: it parses, every name is defined, and it runs without errors"
                          + (" (including its asserts)." if "assert" in code else "."), self.name, 0.85, True, trace)
        return Answer("Problems found:\n" + "\n".join(f"- {p}" for p in problems), self.name, 0.9, True, trace)
