"""A from-scratch intent router: multinomial naive Bayes on words and shapes.

The router is the only *learned* component of the default agent. It is
trained in a few milliseconds from ``data/router/intents.tsv`` and gives a
prior over skills; the skills themselves still decide whether they can
answer, so a routing mistake costs time, not correctness.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

_MONTHS = r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"


def features(text: str) -> List[str]:
    """Word unigrams + bigrams + coarse 'shape' tokens (numbers, operators...)."""
    low = text.lower()
    shapes = []
    if re.search(r"\d", low):
        shapes.append("<num>")
    if re.search(r"\d\s*[-+*/^%]\s*[\d(a-z]|[a-z]\^\d|\d[a-z]\b", low):
        shapes.append("<op>")
    if "=" in low and "->" not in low:
        shapes.append("<eq>")
    if re.search(r"->|→|=>", low):
        shapes.append("<arrow>")
    if re.search(r"[a-z]\s*\([^)]*\)\s*=", low):
        shapes.append("<fcall_eq>")
    if re.search(r"-?\d+\s*,\s*-?\d+\s*,\s*-?\d+", low):
        shapes.append("<numlist>")
    if re.search(rf"\d{{4}}-\d{{1,2}}-\d{{1,2}}|\b({_MONTHS})[a-z]*\.?\s+\d", low):
        shapes.append("<date>")
    if re.search(r"\b(all|every|no)\b.*\.\s.*\?", low):
        shapes.append("<syllogism>")
    if re.search(r"[\[\]'\"]", low):
        shapes.append("<literal>")
    if re.search(r"^\s*def \w+\(|```|\bassert\b|^\s*(import|class) ", low, re.M):
        shapes.append("<code>")
    words = re.findall(r"[a-z]+", low)
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams + shapes * 2


class NaiveBayesRouter:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.labels: List[str] = []
        self.log_prior: Dict[str, float] = {}
        self.counts: Dict[str, Counter] = {}
        self.totals: Dict[str, int] = {}
        self.vocab: set = set()
        self.docs: Counter = Counter()

    def fit(self, examples: Iterable[Tuple[str, str]]) -> "NaiveBayesRouter":
        docs = defaultdict(list)
        for label, text in examples:
            docs[label].append(text)
        n = sum(len(v) for v in docs.values())
        self.labels = sorted(docs)
        for label, texts in docs.items():
            c = Counter()
            for t in texts:
                c.update(features(t))
            self.counts[label] = c
            self.totals[label] = sum(c.values())
            self.vocab.update(c)
            self.docs[label] = len(texts)
            self.log_prior[label] = math.log(len(texts) / n)
        return self

    def partial_fit(self, label: str, text: str) -> None:
        """Learn one more example online (used when MNX learns from chats)."""
        c = self.counts.setdefault(label, Counter())
        feats = features(text)
        c.update(feats)
        self.totals[label] = self.totals.get(label, 0) + len(feats)
        self.vocab.update(feats)
        self.docs[label] += 1
        if label not in self.labels:
            self.labels = sorted(self.labels + [label])
        n = sum(self.docs.values())
        self.log_prior = {k: math.log(self.docs[k] / n) for k in self.labels}

    def predict(self, text: str) -> List[Tuple[str, float]]:
        """Posterior over labels, highest first."""
        feats = [f for f in features(text) if f in self.vocab]
        v = len(self.vocab)
        logp = {}
        for label in self.labels:
            denom = math.log(self.totals[label] + self.alpha * v)
            c = self.counts[label]
            logp[label] = self.log_prior[label] + sum(math.log(c.get(f, 0) + self.alpha) - denom for f in feats)
        m = max(logp.values())
        exp = {k: math.exp(x - m) for k, x in logp.items()}
        z = sum(exp.values())
        return sorted(((k, e / z) for k, e in exp.items()), key=lambda kv: kv[1], reverse=True)


def load_intents(path: Path) -> List[Tuple[str, str]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        label, text = line.split("\t", 1)
        out.append((label.strip(), text.strip()))
    return out


def default_intents_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "router" / "intents.tsv"


def train_default_router(path: Path | None = None) -> NaiveBayesRouter:
    return NaiveBayesRouter().fit(load_intents(path or default_intents_path()))


def cross_validate(examples: Sequence[Tuple[str, str]], folds: int = 5) -> float:
    """k-fold accuracy, used by the evaluation script to keep the router honest."""
    correct = 0
    for k in range(folds):
        train = [e for i, e in enumerate(examples) if i % folds != k]
        test = [e for i, e in enumerate(examples) if i % folds == k]
        r = NaiveBayesRouter().fit(train)
        correct += sum(r.predict(t)[0][0] == label for label, t in test)
    return correct / len(examples)
