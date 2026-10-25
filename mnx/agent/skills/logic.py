"""Deductive reasoning over simple English premises.

Premises such as ``All men are mortal.``, ``No reptiles have fur.``,
``Socrates is a man.`` or ``If something is a bird then it can fly.`` are
parsed into unary predicates and implications, closed under modus ponens and
modus tollens by forward chaining, and the question is answered **yes**,
**no** or **cannot be determined**, with the chain of steps as a proof. The
engine is sound for this fragment: it never answers yes/no without a
derivation, which is exactly where large language models tend to bluff.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from mnx.agent.base import Answer, Skill

_IRREGULAR = {
    "men": "man", "women": "woman", "people": "person", "children": "child", "mice": "mouse",
    "geese": "goose", "feet": "foot", "teeth": "tooth", "wolves": "wolf", "leaves": "leaf",
    "knives": "knife", "lives": "life", "fish": "fish", "sheep": "sheep", "deer": "deer",
    "fungi": "fungus", "cacti": "cactus", "oxen": "ox", "data": "data", "species": "species",
}
_KEEP = ("ss", "us", "is", "ous", "ics")


def singular(word: str) -> str:
    w = word.lower()
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) <= 3 or w.endswith(_KEEP):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "ches", "shes", "xes", "zes")):
        return w[:-2]
    if w.endswith("s"):
        return w[:-1]
    return w


def _phrase(text: str) -> str:
    words = re.sub(r"^(?:a|an|the)\s+", "", text.strip().lower()).split()
    if not words:
        return ""
    return " ".join(words[:-1] + [singular(words[-1])])


def _predicate(verb: str, rest: str) -> Tuple[str, bool]:
    """Canonical predicate name and polarity (True = positive) for ``verb rest``."""
    verb = verb.lower()
    rest = rest.strip().lower()
    neg = False
    if verb in ("can't", "cannot", "can not"):
        verb, neg = "can", True
    elif verb in ("don't", "doesn't", "do not", "does not"):
        m = re.match(r"([a-z]+)\s*(.*)", rest)
        if not m:
            raise ValueError("unsupported negation")
        verb, rest, neg = m.group(1), m.group(2), True
    if rest.startswith("not "):
        neg, rest = True, rest[4:]
    if verb in ("is", "are", "am"):
        return _phrase(rest), not neg
    if verb in ("have", "has"):
        return "has " + rest, not neg
    if verb == "can":
        return "can " + rest, not neg
    return f"{verb_base(verb)} {rest}".strip(), not neg


def verb_base(verb: str) -> str:
    """breathes → breathe, flies → fly, catches → catch, swim → swim."""
    if verb.endswith("ies") and len(verb) > 4:
        return verb[:-3] + "y"
    if verb.endswith(("ches", "shes", "sses", "xes", "zes")):
        return verb[:-2]
    if verb.endswith("s") and not verb.endswith("ss"):
        return verb[:-1]
    return verb


_VERB = r"(is|are|am|has|have|can't|cannot|can not|can|don't|doesn't|do not|does not|lays?|eats?|likes?)"
#: fallback: any single word as the verb (subject must then be a single word)
_ANYVERB = r"([a-z]+)"


@dataclass(frozen=True)
class Rule:
    body: str
    body_pos: bool
    head: str
    head_pos: bool
    source: str


def _lit(pred: str, pos: bool) -> str:
    return pred if pos else f"not {pred}"


def parse_premise(sentence: str, individuals: Set[str] = frozenset()) -> Tuple[str, object]:
    """Return ``("rule", Rule)`` or ``("fact", (individual, pred, pos))``.

    ``individuals`` are names known to denote individuals (e.g. the subject
    of the question), which resolves "Bubbles cannot swim" vs "Birds can fly".
    """
    s = sentence.strip().rstrip(".!")
    low = s.lower()
    m = re.match(rf"^(?:all|every|each|any)\s+(.+?)\s+{_VERB}\s+(.+)$", low) or \
        re.match(rf"^(?:all|every|each|any)\s+(\S+)\s+{_ANYVERB}\s*(.*)$", low)
    if m:
        pred, pos = _predicate(m.group(2), m.group(3))
        return "rule", Rule(_phrase(m.group(1)), True, pred, pos, s)
    m = re.match(rf"^no\s+(.+?)\s+{_VERB}\s+(.+)$", low) or re.match(rf"^no\s+(\S+)\s+{_ANYVERB}\s*(.*)$", low)
    if m:
        pred, pos = _predicate(m.group(2), m.group(3))
        return "rule", Rule(_phrase(m.group(1)), True, pred, not pos, s)
    m = re.match(
        rf"^if\s+(?:something|anything|someone|anyone|a thing|an animal|x)\s+{_VERB}\s+(.+?),?\s+then\s+"
        rf"(?:it|they|he|she|x|that thing)\s+{_VERB}\s+(.+)$",
        low,
    )
    if m:
        b, bpos = _predicate(m.group(1), m.group(2))
        h, hpos = _predicate(m.group(3), m.group(4))
        return "rule", Rule(b, bpos, h, hpos, s)
    m = re.match(rf"^([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)?)\s+(?i:{_VERB})\s+(.+)$", s) or \
        re.match(rf"^([A-Z][\w\-]*)\s+{_ANYVERB}\s*(.*)$", s)
    if m and m.group(1).lower() not in ("all", "every", "no", "some", "if", "each", "any", "the", "a", "an"):
        subj, verb = m.group(1), m.group(2).lower()
        pred, pos = _predicate(verb, m.group(3))
        if subj.lower() in individuals:
            return "fact", (subj.lower(), pred, pos)
        plural = singular(subj.split()[-1]) != subj.split()[-1].lower()
        if verb in ("are", "have", "don't", "do not") or (plural and not verb.endswith("s") and verb != "is"):
            return "rule", Rule(_phrase(subj), True, pred, pos, s)
        return "fact", (subj.lower(), pred, pos)
    m = re.match(r"^([a-z][a-z\- ]*?s)\s+(are|have|can't|cannot|can|don't|do not|lay|eat|like)\s+(.+)$", low)
    if m:  # generic plural: "Birds can fly", "Cats are mammals"
        pred, pos = _predicate(m.group(2), m.group(3))
        return "rule", Rule(_phrase(m.group(1)), True, pred, pos, s)
    raise ValueError(f"cannot parse premise: {sentence!r}")


def parse_question(q: str) -> Tuple[str, Optional[str], str, bool]:
    """Return ``(kind, subject, predicate, positive)``.

    ``kind`` is ``"individual"`` (Is Socrates mortal?) or ``"universal"``
    (Are all men mortal?).
    """
    s = q.strip().rstrip("?").strip()
    low = s.lower()
    m = re.match(r"^(?:are|is)\s+(?:all|every|each)\s+(.+?)\s+(not\s+)?(?:a\s+|an\s+)?([a-z][\w\- ]*)$", low)
    if m:
        return "universal", _phrase(m.group(1)), _phrase(m.group(3)), not m.group(2)
    m = re.match(r"^can\s+(?:all|every)\s+(\S+)\s+(.+)$", low)
    if m:
        return "universal", _phrase(m.group(1)), "can " + m.group(2), True
    m = re.match(r"^(?i:is|was)\s+([A-Z][\w\-]*)\s+(not\s+)?(?:a\s+|an\s+)?([a-z][\w\- ]*)$", s)
    if m:
        return "individual", m.group(1).lower(), _phrase(m.group(3)), not m.group(2)
    m = re.match(r"^(?i:can)\s+([A-Z][\w\-]*)\s+(.+)$", s)
    if m:
        return "individual", m.group(1).lower(), "can " + m.group(2).lower(), True
    m = re.match(r"^(?i:does|do)\s+([A-Z][\w\-]*)\s+([a-z]+)\s*(.*)$", s)
    if m:
        pred, _ = _predicate(m.group(2), m.group(3))
        return "individual", m.group(1).lower(), pred, True
    m = re.match(r"^do\s+(?:all|every)\s+(\S+)\s+([a-z]+)\s*(.*)$", low)
    if m:
        pred, _ = _predicate(m.group(2), m.group(3))
        return "universal", _phrase(m.group(1)), pred, True
    raise ValueError("cannot parse question")


def _nouns(premises: List[str], rules: List[Rule]) -> Set[str]:
    """Predicates used as nouns ("a man", "all cats"), for natural phrasing."""
    nouns = {r.body for r in rules if not r.source.lower().startswith("if ")}
    for p in premises:
        nouns.update(_phrase(w) for w in re.findall(r"\b(?:a|an)\s+([a-z][\w\-]*)", p.lower()))
        nouns.update(singular(w) for w in re.findall(r"\bare\s+(?:not\s+)?([a-z][\w\-]*s)\b", p.lower()))
    return nouns


def _say(who: str, pred: str, pos: bool, nouns: Set[str]) -> str:
    if pred.startswith("has "):
        return f"{who} {'has' if pos else 'does not have'} {pred[4:]}"
    if pred.startswith("can "):
        return f"{who} {'can' if pos else 'cannot'} {pred[4:]}"
    if pred not in nouns and " " in pred:
        verb, rest = pred.split(" ", 1)
        third = verb + "es" if verb.endswith(("s", "sh", "ch", "x", "z")) else verb + "s"
        return f"{who} {third if pos else 'does not ' + verb} {rest}"
    art = ("an " if pred[0] in "aeiou" else "a ") if pred in nouns else ""
    return f"{who} is {'' if pos else 'not '}{art}{pred}"


class Reasoner:
    def __init__(self, rules: List[Rule]):
        self.rules = list(rules)
        # modus tollens: A → B  gives  ¬B → ¬A
        self.rules += [Rule(r.head, not r.head_pos, r.body, not r.body_pos, f"contrapositive of: {r.source}") for r in rules]

    def closure(self, facts: Dict[Tuple[str, bool], str]) -> Dict[Tuple[str, bool], str]:
        """Forward chaining; ``facts`` maps ``(pred, pos)`` → explanation."""
        known = dict(facts)
        changed = True
        while changed:
            changed = False
            for r in self.rules:
                if (r.body, r.body_pos) in known and (r.head, r.head_pos) not in known:
                    known[(r.head, r.head_pos)] = f"{_lit(r.body, r.body_pos)} → {_lit(r.head, r.head_pos)}  ({r.source})"
                    changed = True
        return known


class LogicSkill(Skill):
    name = "logic"
    description = "syllogisms and if-then deduction with proofs"

    def answer(self, question: str) -> Optional[Answer]:
        text = question.strip()
        if "?" not in text:
            return None
        body, _, tail = text.rpartition("?")
        sentences = [s for s in re.split(r"(?<=[.!;])\s+|\n+", body.strip()) if s.strip()]
        if len(sentences) < 2:
            return None
        q_text = sentences[-1]
        q_text = re.sub(r"^(?:so|therefore|then|thus|hence),?\s+", "", q_text, flags=re.I)
        premises = sentences[:-1]
        try:
            kind, subject, pred, pos = parse_question(q_text)
        except ValueError:
            return None
        rules: List[Rule] = []
        facts: Dict[str, Dict[Tuple[str, bool], str]] = {}
        skipped: List[str] = []
        for p in premises:
            try:
                k, v = parse_premise(p, {subject} if kind == "individual" else set())
            except ValueError:
                skipped.append(p.strip())
                continue
            if k == "rule":
                rules.append(v)
            else:
                ind, fp, fpos = v
                facts.setdefault(ind, {})[(fp, fpos)] = f"given: {p.strip().rstrip('.')}"
        if not rules and not facts:
            return None
        reasoner = Reasoner(rules)
        if kind == "universal":
            start = {(subject, True): f"suppose something is a {subject}"}
        else:
            start = facts.get(subject, {})
            if not start:
                return Answer(f"Cannot be determined: nothing is stated about {subject.title()}.", self.name, 0.7, True,
                              [f"no premise mentions {subject}"])
        known = reasoner.closure(start)
        conflicts = sorted({p for (p, s) in known if (p, not s) in known})
        trace = [f"premise ignored (not understood): {s}" for s in skipped]
        trace.append(f"parsed {len(rules)} rule(s); forward-chained with modus ponens and modus tollens")
        if conflicts:
            trace.append("contradiction on: " + ", ".join(conflicts))
            return Answer(f"The premises are contradictory (both '{conflicts[0]}' and 'not {conflicts[0]}' follow).",
                          self.name, 0.85, True, trace)
        who = subject.title() if kind == "individual" else f"every {subject}"
        if (pred, pos) in known:
            verdict = "Yes"
        elif (pred, not pos) in known:
            verdict = "No"
        else:
            verdict = None
        if verdict is None:
            conf = 0.6 if skipped else 0.85
            claim = _say(who, pred, pos, _nouns(premises, rules))
            return Answer(f"Cannot be determined: the premises neither prove nor rule out that {claim}.", self.name, conf, not skipped, trace)
        trace += self._proof(known, (pred, pos if verdict == "Yes" else not pos))
        nouns = _nouns(premises, rules)
        holds = pos if verdict == "Yes" else not pos
        claim = _say(who, pred, holds, nouns)
        return Answer(f"{verdict}, {claim}.", self.name, 0.95 if not skipped else 0.75, True, trace)

    @staticmethod
    def _proof(known: Dict[Tuple[str, bool], str], goal: Tuple[str, bool]) -> List[str]:
        steps: List[str] = []
        seen: Set[Tuple[str, bool]] = set()
        frontier = [goal]
        while frontier:
            lit = frontier.pop()
            if lit in seen:
                continue
            seen.add(lit)
            why = known[lit]
            steps.append(f"{_lit(*lit)}: {why}")
            m = re.match(r"^(not )?(.+?) → ", why)
            if m:
                frontier.append((m.group(2), not m.group(1)))
        return ["proof:"] + [f"  {s}" for s in reversed(steps)]
