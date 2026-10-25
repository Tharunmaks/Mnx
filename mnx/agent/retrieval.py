"""BM25 retrieval and extractive answering over local text files.

Facts live in plain ``.md`` / ``.txt`` files instead of model weights, so the
knowledge base is inspectable, editable and citable: every retrieved answer
says which file it came from. Nothing here is learned from a pretrained
model; it is classic information retrieval (Robertson & Zaragoza's BM25)
plus a sentence scorer.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have having
    he her here hers herself him himself his how i if in into is it its itself just me more most my myself no
    nor not now of off on once only or other our ours out over own same she should so some such than that the
    their theirs them themselves then there these they this those through to too under until up very was we
    were what when where which while who whom why will with would you your yours yourself tell me please know
    whats what's who's does did many much world""".split()
)


def stem(word: str) -> str:
    """Tiny suffix-stripping stemmer (a few Porter step-1 rules)."""
    w = word
    if len(w) > 6 and w.endswith("ence"):
        return w[:-4] + "ent"
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 4 and w.endswith("es") and w[-3] in "sxz":
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    return w


def tokenize(text: str) -> List[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower().replace("'s ", " ")) if t not in STOPWORDS]


@dataclass
class Chunk:
    text: str
    source: str
    heading: str


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def split_sentences(text: str) -> List[str]:
    out: List[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if line:
            out.extend(s.strip() for s in _SENT.split(line) if s.strip())
    return out


def chunk_markdown(text: str, source: str) -> List[Chunk]:
    """Split on headings and blank lines; each chunk remembers its heading."""
    chunks: List[Chunk] = []
    heading = Path(source).stem
    buf: List[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            # long paragraphs become ~3-sentence windows for sharper retrieval
            sents = split_sentences(body)
            for i in range(0, len(sents), 3):
                chunks.append(Chunk(" ".join(sents[i:i + 3]), source, heading))
        buf.clear()

    for line in text.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            flush()
            heading = m.group(1).strip()
        elif not line.strip():
            flush()
        else:
            buf.append(line)
    flush()
    return chunks


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: List[Chunk] = []
        self.tfs: List[Counter] = []
        self.df: Counter = Counter()
        self.total_len = 0

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunk: Chunk) -> None:
        toks = tokenize(chunk.text + " " + chunk.heading)
        tf = Counter(toks)
        self.chunks.append(chunk)
        self.tfs.append(tf)
        self.df.update(tf.keys())
        self.total_len += len(toks)

    def add_text(self, text: str, source: str) -> int:
        new = chunk_markdown(text, source)
        for c in new:
            self.add(c)
        return len(new)

    def add_folder(self, folder: Path) -> int:
        n = 0
        for path in sorted(Path(folder).rglob("*")):
            if path.suffix.lower() in (".md", ".txt") and path.is_file():
                n += self.add_text(path.read_text(encoding="utf-8", errors="replace"), str(path.name))
        return n

    def idf(self, term: str) -> float:
        n, df = len(self.chunks), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> List[Tuple[float, Chunk]]:
        q = tokenize(query)
        if not q or not self.chunks:
            return []
        avg = self.total_len / len(self.chunks)
        scores = []
        for i, tf in enumerate(self.tfs):
            dl = sum(tf.values())
            s = 0.0
            for t in set(q):
                f = tf.get(t, 0)
                if f:
                    s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avg))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [(s, self.chunks[i]) for s, i in scores[:k]]


#: light query expansion so "how far" can match "km from", "invented" can match "created", ...
EXPANSIONS: Dict[str, set] = {
    "far": {"distance", "km", "kilometr", "mile", "away"},
    "distance": {"far", "km"},
    "fast": {"speed"},
    "big": {"largest", "size", "area"},
    "large": {"largest", "biggest", "size"},
    "tall": {"highest", "height", "tallest", "metr"},
    "high": {"highest", "height", "tallest"},
    "old": {"age", "year"},
    "hot": {"hottest", "temperature"},
    "invent": {"creat", "discover", "introduc", "develop"},
    "creat": {"invent", "develop", "found"},
    "wrote": {"writ", "author", "wrote"},
    "write": {"wrote", "author"},
    "author": {"wrote"},
    "discover": {"discover", "found", "describ"},
    "found": {"founded", "establish"},
    "end": {"lasted", "ended", "until"},
    "gain": {"became", "won"},
    "start": {"began", "lasted", "begin"},
    "begin": {"began", "lasted", "start"},
    "body": {"skeleton", "human"},
    "biggest": {"largest"},
    "largest": {"biggest"},
    "tallest": {"highest"},
    "highest": {"tallest"},
    "torn": {"fell"},
    "fall": {"fell"},
    "destroy": {"fell"},
}


def _expand(question: str) -> str:
    extra = [w for t in tokenize(question) for w in EXPANSIONS.get(t, ())]
    return question + " " + " ".join(extra)


@dataclass
class Extract:
    sentence: str
    source: str
    coverage: float  # fraction of the question's IDF mass found in the sentence
    score: float


def best_sentences(index: BM25Index, question: str, k_chunks: int = 4) -> List[Extract]:
    """Rank sentences from the top chunks by how much of the question they cover."""
    q_terms = list(dict.fromkeys(tokenize(question)))
    if not q_terms:
        return []
    weights = {t: index.idf(t) for t in q_terms}
    total = sum(weights.values()) or 1.0
    lowq = question.lower()
    want_year = bool(re.search(r"\b(when|what year|which year)\b", lowq))
    want_number = bool(re.search(r"\b(how many|how much|how far|how long|how fast|how old|how big|how tall|how deep|how high)\b", lowq))
    want_person = bool(re.search(r"\bwho\b", lowq))
    personal = bool(re.search(r"\b(my|me|mine|i)\b", lowq))
    alts = {t: {t} | EXPANSIONS.get(t, set()) for t in q_terms}
    out: List[Extract] = []
    for chunk_score, chunk in index.search(_expand(question), k_chunks):
        head = set(tokenize(chunk.heading))
        for sent in split_sentences(chunk.text):
            seq = tokenize(sent)
            toks = set(seq)
            matched = [t for t in q_terms if alts[t] & toks]
            got = sum(weights[t] for t in matched)
            got += 0.3 * sum(w for t, w in weights.items() if t in head and t not in matched)
            cov = min(1.0, got / total)
            bonus = 0.2 * len(matched) / max(1, len(seq))
            if seq and alts[q_terms[0]] & set(seq[:3]):
                bonus += 0.1  # definitional sentences name the subject first
            if personal and chunk.source == "memory":
                bonus += 0.3
            if want_year and re.search(r"\b\d{3,4}\b", sent):
                bonus += 0.15
            if want_number and re.search(r"\d", sent):
                bonus += 0.15
            if want_person and re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+", sent):
                bonus += 0.1
            out.append(Extract(sent, chunk.source, cov, cov + bonus + 0.02 * chunk_score))
    out.sort(key=lambda e: e.score, reverse=True)
    seen, uniq = set(), []
    for e in out:
        if e.sentence not in seen:
            seen.add(e.sentence)
            uniq.append(e)
    return uniq


class Memory:
    """User-taught facts, optionally persisted as JSON lines."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.facts: List[str] = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.facts.append(json.loads(line)["fact"])

    def add(self, fact: str) -> None:
        self.facts.append(fact)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"fact": fact}) + "\n")


def default_knowledge_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "knowledge"


def build_index(folders: Iterable[Path] = (), facts: Sequence[str] = ()) -> BM25Index:
    idx = BM25Index()
    for folder in folders:
        if Path(folder).exists():
            idx.add_folder(Path(folder))
    for fact in facts:
        idx.add(Chunk(fact, "memory", "memory"))
    return idx
