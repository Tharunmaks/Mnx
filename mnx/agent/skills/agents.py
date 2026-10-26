"""The registry skill: pick one of thousands of small agents, run it, show the proof.

`mnx/agent/registry.py` holds ten thousand tiny agents. This skill is how MNX1
uses them. It searches them by what the question is about, re-runs the chosen
agent's own asserts, applies it to the arguments in the question, and answers
with the result and the tests that backed it. When the question is about the
standard library instead ("what can sort a list in place"), it points at the
right function and says plainly that it did not run anything.

It is deliberately cautious: if it cannot read arguments out of the question
that match the agent's signature, it returns nothing and lets another skill
answer.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from mnx.agent.base import Answer, Skill
from mnx.agent.registry import Agent, Registry, call, check

_LIST = re.compile(r"\[[^\[\]]*\]")
_STR = re.compile(r"'([^']*)'|\"([^\"]*)\"")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
#: tokens that are data in their own right and must not be read as numbers:
#: a dotted quad like 192.168.0.1, an address with an @, a #rrggbb colour.
#: "3.14159" is deliberately not one of them: that is a number.
_TOKEN = re.compile(r"(?<![\w.])(?:\d+(?:\.\d+){2,}|[^\s@]+@[^\s@]+|#[0-9a-fA-F]{3,8})(?![\w.])")
#: the parameter an agent bakes into its own name, wherever it sits in it:
#: the 25 of percent_25_of as much as the 7 of to_base_7.
_PARAM = re.compile(r"\d+")
_LOOKUP = re.compile(r"\b(what|which|is there|anything|something|function|module|library|stdlib|standard library)\b", re.I)
#: a request to write code belongs to the coder, which tests what it writes
#: against the user's own asserts. The registry must not answer over it.
_IS_CODE_REQUEST = re.compile(r"^\s*(?:def |class |```|assert |import |from )|\n\s*assert |\bwrite (?:a |me )?(?:python )?(?:function|code|class|script|program)\b", re.I)


def params_of(code: str, name: str) -> Optional[List[str]]:
    """The parameter names of the agent's function, from its own source."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [a.arg for a in node.args.args]
    return None


def _bracketed(text: str) -> List[Tuple[int, int]]:
    """The outermost ``[...]`` spans, so a matrix comes back whole.

    A regex for a bracket pair with no brackets inside it only ever matches the
    innermost brackets, which turned
    ``[[1,2],[3,4]]`` into the single row ``[1, 2]``. Counting depth gives the
    whole literal.
    """
    spans, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c == "[":
            if depth == 0:
                start = i
            depth += 1
        elif c == "]" and depth:
            depth -= 1
            if depth == 0:
                spans.append((start, i + 1))
    return spans


def _blank_out(text: str, spans: Sequence[Tuple[int, int]]) -> str:
    out, at = [], 0
    for start, end in spans:
        out.append(text[at:start])
        out.append(" ")
        at = end
    out.append(text[at:])
    return "".join(out)


def arguments_in(question: str, arity: int, parameter: str = "") -> Optional[list]:
    """Read ``arity`` arguments out of a question, or give up.

    List literals are taken first and then removed, so the numbers inside them
    are not mistaken for separate arguments. ``parameter`` is the number already
    baked into the agent's name: in "to base 7 of 100" the 7 chose
    ``to_base_7`` and must not also become its argument, so one occurrence of it
    is dropped from what is left.
    """
    args: list = []
    spans = _bracketed(question)
    for start, end in spans:
        try:
            args.append(ast.literal_eval(question[start:end]))
        except (ValueError, SyntaxError):
            continue
    rest = _blank_out(question, spans)
    if parameter.isdigit():
        rest = re.sub(rf"(?<![\d.]){re.escape(parameter)}(?![\d.])", " ", rest, count=1)
    for m in _ISO.finditer(rest):
        args.append(m.group(0))
    rest = _ISO.sub(" ", rest)
    for m in _STR.finditer(rest):
        args.append(m.group(1) if m.group(1) is not None else m.group(2))
    rest = _STR.sub(" ", rest)
    for m in _TOKEN.finditer(rest):
        args.append(m.group(0))
    rest = _TOKEN.sub(" ", rest)
    for m in _NUM.finditer(rest):
        raw = m.group(0)
        args.append(float(raw) if "." in raw else int(raw))
    return args[:arity] if len(args) >= arity else None


class RegistrySkill(Skill):
    name = "registry"
    description = "ten thousand small agents: pick one, run it, show the tests it passed"

    def __init__(self, registry_dir: Optional[Path] = None, min_margin: float = 1.5,
                 max_tried: int = 4):
        self._dir = registry_dir
        self._registry: Optional[Registry] = None
        self.min_margin = min_margin  # the best agent must beat the runner-up by this much
        self.max_tried = max_tried    # how many of the best matches actually get run

    @property
    def registry(self) -> Optional[Registry]:
        if self._registry is None:
            try:
                self._registry = Registry.load(self._dir) if self._dir else Registry.load()
            except FileNotFoundError:
                return None
        return self._registry

    def answer(self, question: str) -> Optional[Answer]:
        reg = self.registry
        if reg is None:
            return None
        q = question.strip()
        if not q or len(q) < 4:
            return None
        if _IS_CODE_REQUEST.search(q):
            return None  # the coder owns this, and it verifies against the user's own tests
        hits = self._candidates(reg, q)
        trace = [f"registry: {len(reg)} agents, best matches " +
                 ", ".join(f"{a.id} ({s:.1f})" for s, a in hits[:3])]
        if hits:
            hits = self._prefer(q, hits)
            trace.append("after the question's own words and numbers: " +
                         ", ".join(f"{a.id} ({s:.1f})" for s, a in hits[:3]))
            answered = self._run_best(q, hits, trace)
            if answered is not None:
                return answered
        if _LOOKUP.search(q):
            return self._point(q, reg, trace)
        return None

    # ------------------------------------------------------------- choosing
    @staticmethod
    def _candidates(reg: Registry, q: str) -> List[Tuple[float, Agent]]:
        """Search twice: with the question's numbers, and without them.

        A number is a rare token, so BM25 treats it as highly informative and a
        parametric agent that merely happens to bake that number in outranks
        the agent the question is actually about: "prime factors of 360" put
        ``is_coprime_with_360`` first and ``prime_factors_of`` nowhere. Searching
        the words on their own puts the second one back in the running, and the
        re-ranking below decides between them.
        """
        words_only = _NUM.sub(" ", q)
        merged: Dict[str, Tuple[float, Agent]] = {}
        for query in (q, words_only):
            for score, agent in reg.search(query, k=12, tier="verified"):
                keep = merged.get(agent.id)
                if keep is None or score > keep[0]:
                    merged[agent.id] = (score, agent)
        return sorted(merged.values(), key=lambda sa: -sa[0])

    @staticmethod
    def _prefer(q: str, hits: Sequence[Tuple[float, Agent]]) -> List[Tuple[float, Agent]]:
        """Re-rank near-ties using the question's own words and numbers.

        BM25 cannot tell ``celsius_to_fahrenheit`` from ``fahrenheit_to_celsius``,
        or ``percentile_90`` from ``percentile_9``: the words are the same. Two
        cheap signals settle it. A parameter in the agent's name that appears as
        a number in the question counts for a lot, and name words that appear in
        the question *in the same order* count for a little.
        """
        # numbers inside a list literal are the data, not the parameter
        outside = _LIST.sub(" ", q)
        numbers = set(_NUM.findall(outside))
        words = [w[:-1] if len(w) > 3 and w.endswith("s") else w
                 for w in re.findall(r"[a-z0-9]+", q.lower())]

        def bonus(agent: Agent) -> float:
            out = 0.0
            parts = [w[:-1] if len(w) > 3 and w.endswith("s") else w
                     for w in re.split(r"[_\W]+", agent.name.lower()) if w and not w.isdigit()]
            seen = [words.index(w) for w in parts if w in words]
            baked = _PARAM.search(agent.name)
            if baked is not None:
                tail = baked.group(0)
                # the number alone is not evidence: "prime factors of 360" is not
                # a question about 360-coprimality. Something the agent is *called*
                # has to appear in the question before its parameter counts.
                out += 4.0 if (tail in numbers and seen) else -2.0
            if len(seen) >= 2:
                out += 2.0 if seen == sorted(seen) else -2.0
            return out

        return sorted(((s + bonus(a), a) for s, a in hits), key=lambda sa: -sa[0])

    # ------------------------------------------------------------- answering
    @staticmethod
    def _runnable(q: str, hits: Sequence[Tuple[float, Agent]]) -> List[Tuple[float, Agent, list]]:
        """The hits whose arguments the question actually supplies, best first.

        A parameter baked into an agent's name is not also its argument, so
        ``is_coprime_with_360`` has nothing left to run on in "prime factors of
        360" — it matched the number, not the question. Dropping those instead
        of giving up on the whole question is what lets the agent that does fit
        be found behind them.
        """
        out = []
        for score, agent in hits:
            names = params_of(agent.code, agent.name)
            if names is None:
                continue
            baked = _PARAM.search(agent.name)
            args = arguments_in(q, len(names), parameter=baked.group(0) if baked else "")
            if args is None:
                continue
            out.append((score, agent, args))
        return out

    def _run_best(self, q: str, hits: Sequence[Tuple[float, Agent]], trace: List[str]) -> Optional[Answer]:
        candidates = self._runnable(q, hits)
        if not candidates:
            trace.append("no agent that matched can be run on what the question gives: not answering")
            return None
        if len(candidates) < len(hits):
            skipped = [a.id for (_, a) in hits if all(a.id != c[1].id for c in candidates)]
            trace.append("skipped, nothing to run them on: " + ", ".join(skipped[:3]))

        # run the best few. An agent that fails its own tests or raises on these
        # arguments is the wrong agent, not a reason to give up on the question:
        # "to roman numeral 1994" matches from_roman_numeral first, and that one
        # raises on a number, so the next one gets its turn.
        ran: List[Tuple[float, Agent, list, object]] = []
        for score, agent, args in candidates[:self.max_tried]:
            ok, why = check(agent)
            if not ok:
                trace.append(f"{agent.id} failed its own tests ({why}): skipped")
                continue
            try:
                result = call(agent, *args)
            except Exception as exc:  # noqa: BLE001 - a crash is an answer we must not give
                trace.append(f"{agent.id} raised {type(exc).__name__} on {args!r}: skipped")
                continue
            ran.append((score, agent, args, result))
        if not ran:
            return None

        score, agent, args, result = ran[0]
        if len(ran) > 1 and score - ran[1][0] < self.min_margin:
            # a near-tie is only a problem if the two disagree: two agents that
            # give the same answer are two witnesses, not an ambiguity. Each one
            # reads its own arguments out of the question, so in "25 percent of
            # 80" this compares 25% of 80 with 80% of 25.
            second, second_result = ran[1][1], ran[1][3]
            if result != second_result:
                trace.append(f"{agent.id} and {second.id} match about equally well and disagree: not answering")
                return None
            trace.append(f"{agent.id} and {second.id} match equally well and agree on this input")
        trace.append(f"ran {agent.id}({', '.join(repr(a) for a in args)}) after re-running its "
                     f"{len(agent.tests)} test(s)")
        shown = "\n".join(agent.tests[:3])
        return Answer(f"{result!r}\n\nFrom the agent `{agent.id}` — {agent.description}\n"
                      f"```python\n{agent.code}```\nTests it passed just now:\n```python\n{shown}\n```",
                      self.name, 0.5, True, trace, [agent.id])

    def _point(self, q: str, reg: Registry, trace: List[str]) -> Optional[Answer]:
        refs = reg.search(q, k=3, tier="reference")
        if not refs:
            return None
        trace.append("answering with pointers from the reference tier: nothing was run")
        lines = [f"- `{a.module}.{a.name}` — {a.description}" for _, a in refs]
        return Answer("From the standard library (I haven't run these, so check them):\n" + "\n".join(lines),
                      self.name, 0.45, False, trace, [a.id for _, a in refs])
