"""Learning from use: what MNX1 keeps from every conversation.

No gradient descent on a giant network. MNX1 learns in ways you can read
and undo, all stored as plain files in a state directory (``~/.mnx`` by
default):

``transcript.jsonl``
    every question, the answer, which agent and skill answered, and whether
    it was verified, so you can always see what it said.
``learned_intents.tsv``
    new router examples. They come from two places: answers you confirm
    ("correct", "good"), and *self-training*, where a verified,
    high-confidence answer teaches the router that phrasing. The router is
    updated online with :meth:`NaiveBayesRouter.partial_fit`.
``corrections.jsonl``
    "wrong, the answer is X": the question and your answer, retrieved for
    similar questions later and credited to you.
``experience.jsonl``
    verified answers from the slow agents (program search, coding), replayed
    instantly when the same question comes back.
``recipes/``
    code that passed tests (yours or the recipe's), saved by the coding agent
    as new recipes. This is how MNX1 learns *from its agents*.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mnx.agent.base import Answer
from mnx.agent.retrieval import BM25Index, Chunk, tokenize

_POSITIVE = re.compile(r"^\s*(correct|right|good( answer| job)?|great|perfect|exactly|yes,? (that's|that is) (right|correct)|"
                       r"that's (right|correct)|nice|well done|👍)\s*[.!]*\s*$", re.I)
_NEGATIVE = re.compile(r"^\s*(no[,.!]?|nope|wrong|incorrect|that's (wrong|incorrect|not right)|that is (wrong|incorrect)|"
                       r"not (right|correct)|bad answer|👎)", re.I)
_CORRECTION = re.compile(r"(?:the\s+)?(?:correct|right|real|actual)?\s*answer\s+is\s*[:\-]?\s*(.+)$|"
                         r"\b(?:it(?:'s| is)|actually)\s+(.+)$", re.I | re.S)

#: skills whose answers are worth replaying from experience (they are slow)
SLOW_SKILLS = {"synth", "code", "neural-code"}
#: never self-train the router on these
NO_SELF_TRAIN = {"chat", "memory", "abstain", "learned", "experience", "feedback"}


def normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower()).rstrip(" ?.!")


class Learner:
    def __init__(self, state_dir: Optional[Path] = None, max_self_train: int = 5000):
        self.dir = Path(state_dir).expanduser() if state_dir else None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.max_self_train = max_self_train
        self.last: Optional[Tuple[str, Answer]] = None
        self.intents: List[Tuple[str, str]] = self._read_tsv("learned_intents.tsv")
        self.corrections: List[Dict] = self._read_jsonl("corrections.jsonl")
        self.experience: Dict[str, Dict] = {e["key"]: e for e in self._read_jsonl("experience.jsonl")}
        self.correction_index = BM25Index()
        for c in self.corrections:
            self.correction_index.add(Chunk(c["question"], str(len(self.correction_index)), "correction"))
        self.stats = {"self_trained": 0, "confirmed": 0, "corrected": 0, "replayed": 0}

    # ---------------------------------------------------------------- files
    @property
    def recipe_dir(self) -> Optional[Path]:
        return self.dir / "recipes" if self.dir else None

    def _path(self, name: str) -> Optional[Path]:
        return self.dir / name if self.dir else None

    def _read_jsonl(self, name: str) -> List[Dict]:
        p = self._path(name)
        if not p or not p.exists():
            return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    def _append_jsonl(self, name: str, row: Dict) -> None:
        p = self._path(name)
        if p:
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _read_tsv(self, name: str) -> List[Tuple[str, str]]:
        p = self._path(name)
        if not p or not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                label, text = line.split("\t", 1)
                out.append((label, text.replace("\\n", "\n")))
        return out

    # ------------------------------------------------------------- feedback
    def handle_feedback(self, text: str, router) -> Optional[Answer]:
        """React to "correct" / "wrong, the answer is X" about the previous answer."""
        if self.last is None:
            return None
        question, prev = self.last
        if _POSITIVE.match(text):
            if prev.skill not in NO_SELF_TRAIN:
                self._learn_intent(router, prev.skill, question)
            self.stats["confirmed"] += 1
            self._append_jsonl("feedback.jsonl", {"t": time.time(), "question": question, "answer": prev.text, "verdict": "good"})
            return Answer("Thanks! I'll route questions like that the same way next time.", "feedback", 1.0, True,
                          [f"reinforced router: '{question[:60]}' → {prev.skill}"])
        if _NEGATIVE.match(text):
            rest = _NEGATIVE.sub("", text, count=1).strip(" ,.:;-")
            m = _CORRECTION.search(rest) if rest else None
            fix = (m.group(1) or m.group(2)).strip() if m else (rest if len(rest.split()) >= 1 and rest else "")
            self._append_jsonl("feedback.jsonl", {"t": time.time(), "question": question, "answer": prev.text,
                                                  "verdict": "bad", "correction": fix})
            self.experience.pop(normalize_question(question), None)
            if fix:
                self.add_correction(question, fix)
                return Answer(f"Thanks for correcting me. I've learned that the answer to \"{question}\" is: {fix}",
                              "feedback", 1.0, True, ["saved to corrections; similar questions will use your answer"])
            return Answer("Sorry about that. Tell me the right answer (\"wrong, the answer is ...\") and I'll remember it.",
                          "feedback", 1.0, True, ["logged negative feedback"])
        return None

    def add_correction(self, question: str, answer: str) -> None:
        row = {"t": time.time(), "question": question, "answer": answer}
        self.corrections.append(row)
        self.correction_index.add(Chunk(question, str(len(self.correction_index)), "correction"))
        self._append_jsonl("corrections.jsonl", row)
        self.stats["corrected"] += 1

    def lookup_correction(self, question: str, min_overlap: float = 0.8) -> Optional[Answer]:
        hits = self.correction_index.search(question, k=1)
        if not hits:
            return None
        _, chunk = hits[0]
        row = self.corrections[int(chunk.source)]
        a, b = set(tokenize(question)), set(tokenize(row["question"]))
        overlap = len(a & b) / max(1, len(a | b))
        if overlap < min_overlap:
            return None
        return Answer(f"{row['answer']}  [learned from you]", "learned", 0.9, False,
                      [f"matches a question you corrected me on ({overlap:.0%} word overlap): {row['question']}"], ["corrections"])

    # ------------------------------------------------------------ experience
    def replay(self, question: str) -> Optional[Answer]:
        e = self.experience.get(normalize_question(question))
        if not e:
            return None
        self.stats["replayed"] += 1
        return Answer(e["text"], e["skill"], e["confidence"], True, ["recalled a verified answer from experience"] + e["trace"][:6])

    # --------------------------------------------------------------- observe
    def observe(self, question: str, ans: Answer, router, agent_name: str) -> None:
        """Log every exchange; learn from verified, confident ones."""
        self._append_jsonl("transcript.jsonl", {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": question, "answer": ans.text, "agent": agent_name,
            "skill": ans.skill, "verified": ans.verified, "confidence": round(ans.confidence, 3),
        })
        if ans.skill == "feedback":
            return
        self.last = (question, ans)
        if ans.verified and ans.confidence >= 0.9 and ans.skill not in NO_SELF_TRAIN and not ans.failed \
                and len(self.intents) < self.max_self_train:
            self._learn_intent(router, ans.skill, question)
            self.stats["self_trained"] += 1
        key = normalize_question(question)
        if ans.skill in SLOW_SKILLS and ans.verified and not ans.failed and key not in self.experience:
            row = {"key": key, "text": ans.text, "skill": ans.skill, "confidence": ans.confidence, "trace": ans.trace}
            self.experience[key] = row
            self._append_jsonl("experience.jsonl", row)

    def _learn_intent(self, router, label: str, text: str) -> None:
        if (label, text) in self.intents:
            return
        self.intents.append((label, text))
        router.partial_fit(label, text)
        p = self._path("learned_intents.tsv")
        if p:
            with p.open("a", encoding="utf-8") as f:
                f.write(f"{label}\t{text.replace(chr(10), chr(92) + 'n')}\n")
