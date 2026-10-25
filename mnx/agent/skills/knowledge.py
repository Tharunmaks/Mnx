"""Retrieval-backed skills: answering from the knowledge base, and memory."""

from __future__ import annotations

import re
from typing import Optional

from mnx.agent.base import Answer, Skill
from mnx.agent.retrieval import BM25Index, Chunk, Memory, best_sentences

_PRONOUNS = [
    (r"\bi am\b", "you are"), (r"\bi'm\b", "you're"), (r"\bi was\b", "you were"), (r"\bi\b", "you"),
    (r"\bmy\b", "your"), (r"\bmine\b", "yours"), (r"\bme\b", "you"), (r"\bmyself\b", "yourself"),
]


def second_person(text: str) -> str:
    out = text
    for pat, rep in _PRONOUNS:
        out = re.sub(pat, rep, out, flags=re.I)
    return out[:1].upper() + out[1:]


class KnowledgeSkill(Skill):
    name = "knowledge"
    description = "answer factual questions from local documents (BM25 retrieval + sentence extraction)"

    #: minimum share of the question's IDF mass the answer sentence must contain
    min_coverage = 0.6

    def __init__(self, index: BM25Index):
        self.index = index

    def answer(self, question: str) -> Optional[Answer]:
        extracts = best_sentences(self.index, question)
        if not extracts:
            return None
        best = extracts[0]
        trace = [f"BM25 over {len(self.index)} chunks; best sentence covers {best.coverage:.0%} of the question's weighted terms"]
        for e in extracts[1:3]:
            trace.append(f"runner-up ({e.coverage:.0%}, {e.source}): {e.sentence}")
        if best.coverage < self.min_coverage:
            return Answer(f"I'm not sure. The closest thing I found is: \"{best.sentence}\"", self.name,
                          0.25 * best.coverage, False, trace, [best.source], failed=True)
        text = second_person(best.sentence) if best.source == "memory" else best.sentence
        # two sources agreeing (same top sentence wording aside) is weak corroboration
        conf = 0.35 + 0.55 * best.coverage
        if len(extracts) > 1 and extracts[1].coverage >= best.coverage - 0.05 and extracts[1].source != best.source:
            conf -= 0.1
            trace.append("a different source scores almost as well; lowering confidence")
        return Answer(f"{text}  [source: {best.source}]", self.name, conf, False, trace, [best.source])


class MemorySkill(Skill):
    name = "memory"
    description = "remember facts the user teaches ('remember that ...')"

    _PAT = re.compile(
        r"^(?:please\s+)?(?:remember|note|don't forget|do not forget|keep in mind|save this|learn)"
        r"(?:\s+that)?\s*[:,-]?\s*(.+)$",
        re.I | re.S,
    )

    def __init__(self, index: BM25Index, memory: Memory):
        self.index = index
        self.memory = memory

    def answer(self, question: str) -> Optional[Answer]:
        m = self._PAT.match(question.strip())
        if not m:
            return None
        fact = m.group(1).strip().rstrip(".!")
        if len(fact) < 3:
            return None
        self.memory.add(fact)
        self.index.add(Chunk(fact, "memory", "memory"))
        where = f" (saved to {self.memory.path})" if self.memory.path else " for this session"
        return Answer(f"Got it. I'll remember: {second_person(fact)}.", self.name, 0.95, True,
                      [f"added to the retrieval index{where}"])
