"""Byte-level Byte Pair Encoding tokenizer, written from scratch.

The tokenizer works on raw UTF-8 bytes so that any input string can be encoded
without unknown tokens. Text is first split into "pre-tokens" with a regular
expression tuned for source code (identifiers, numbers, punctuation runs and
whitespace runs including indentation), then BPE merges are applied inside each
pre-token.

Vocabulary layout
-----------------
    [0, 256)                      raw bytes
    [256, 256 + n_special)        special tokens (see ``SPECIAL_TOKENS``)
    [256 + n_special, vocab_size) learned merges, in rank order

The tokenizer is serialised to a self-contained JSON file.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BOS = "<|bos|>"
EOS = "<|eos|>"
PAD = "<|pad|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"

SPECIAL_TOKENS: Tuple[str, ...] = (BOS, EOS, PAD, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX)

N_BYTES = 256

# Pre-tokenisation pattern. Order matters: contractions, words (optionally
# preceded by a single space), numbers, punctuation runs, then whitespace. The
# ``\s+(?!\S)`` branch keeps leading indentation separate from the token that
# follows it, which lets indentation levels become their own tokens.
DEFAULT_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)"
    r"| ?[A-Za-z_][A-Za-z0-9_]*"
    r"| ?\d+"
    r"| ?[^\sA-Za-z0-9_]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

Pair = Tuple[int, int]


def _pair_counts(word: Sequence[int], freq: int, counts: Dict[Pair, int]) -> None:
    for a, b in zip(word, word[1:]):
        counts[(a, b)] += freq


def _merge_word(word: List[int], pair: Pair, new_id: int) -> List[int]:
    out: List[int] = []
    i = 0
    n = len(word)
    a, b = pair
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


class BPETokenizer:
    """A byte-level BPE tokenizer with special tokens for code and FIM."""

    def __init__(
        self,
        merges: Optional[Sequence[Pair]] = None,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        pattern: str = DEFAULT_PATTERN,
    ) -> None:
        self.pattern = pattern
        self._regex = re.compile(pattern)
        self.special_tokens: List[str] = list(special_tokens)
        self.special_to_id: Dict[str, int] = {
            tok: N_BYTES + i for i, tok in enumerate(self.special_tokens)
        }
        self.id_to_special: Dict[int, str] = {v: k for k, v in self.special_to_id.items()}
        self._special_regex = (
            re.compile("|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)))
            if self.special_tokens
            else None
        )

        self.merges: List[Pair] = []
        self.merge_ranks: Dict[Pair, int] = {}
        # id -> bytes for every non-special token
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(N_BYTES)}
        self._cache: Dict[str, List[int]] = {}
        for pair in merges or []:
            self._add_merge((int(pair[0]), int(pair[1])))

    # ------------------------------------------------------------------ props
    @property
    def n_special(self) -> int:
        return len(self.special_tokens)

    @property
    def merge_base(self) -> int:
        """First id used by learned merges."""
        return N_BYTES + self.n_special

    @property
    def vocab_size(self) -> int:
        return self.merge_base + len(self.merges)

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def bos_id(self) -> int:
        return self.special_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.special_to_id[EOS]

    @property
    def pad_id(self) -> int:
        return self.special_to_id[PAD]

    @property
    def fim_prefix_id(self) -> int:
        return self.special_to_id[FIM_PREFIX]

    @property
    def fim_middle_id(self) -> int:
        return self.special_to_id[FIM_MIDDLE]

    @property
    def fim_suffix_id(self) -> int:
        return self.special_to_id[FIM_SUFFIX]

    # --------------------------------------------------------------- training
    def _add_merge(self, pair: Pair) -> int:
        new_id = self.vocab_size
        self.merges.append(pair)
        self.merge_ranks[pair] = len(self.merges) - 1
        self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self._cache.clear()
        return new_id

    def train(self, texts: Iterable[str], vocab_size: int, min_frequency: int = 2, verbose: bool = False) -> None:
        """Learn BPE merges from an iterable of strings.

        ``vocab_size`` is the *total* target size including bytes and special
        tokens. Training stops early if no pair occurs at least
        ``min_frequency`` times.
        """
        if vocab_size < self.merge_base:
            raise ValueError(f"vocab_size must be >= {self.merge_base}, got {vocab_size}")
        n_merges = vocab_size - self.merge_base
        if n_merges <= 0:
            return

        # Pre-token frequency table, each pre-token as a list of byte ids.
        word_freq: Counter = Counter()
        for text in texts:
            for chunk in self._split_specials(text):
                if chunk in self.special_to_id:
                    continue
                for m in self._regex.finditer(chunk):
                    word_freq[m.group(0)] += 1

        words: List[List[int]] = [list(w.encode("utf-8")) for w in word_freq]
        freqs: List[int] = list(word_freq.values())

        # Pair statistics plus an index of which words contain each pair.
        counts: Dict[Pair, int] = defaultdict(int)
        where: Dict[Pair, set] = defaultdict(set)
        for wi, (word, f) in enumerate(zip(words, freqs)):
            for a, b in zip(word, word[1:]):
                counts[(a, b)] += f
                where[(a, b)].add(wi)

        for step in range(n_merges):
            if not counts:
                break
            best = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            pair, best_count = best
            if best_count < min_frequency:
                break
            new_id = self._add_merge(pair)
            if verbose and (step % 100 == 0 or step == n_merges - 1):
                print(f"merge {step + 1}/{n_merges}: {pair} -> {new_id} (count={best_count})")

            affected = list(where.pop(pair))
            counts.pop(pair, None)
            for wi in affected:
                old = words[wi]
                f = freqs[wi]
                for a, b in zip(old, old[1:]):
                    p = (a, b)
                    if p in counts:
                        counts[p] -= f
                        if counts[p] <= 0:
                            del counts[p]
                            where.pop(p, None)
                        else:
                            where[p].discard(wi)
                new = _merge_word(old, pair, new_id)
                words[wi] = new
                for a, b in zip(new, new[1:]):
                    p = (a, b)
                    counts[p] += f
                    where[p].add(wi)

    @classmethod
    def train_from_files(
        cls,
        paths: Iterable[Path],
        vocab_size: int,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> "BPETokenizer":
        tok = cls()
        texts = []
        for p in paths:
            try:
                texts.append(Path(p).read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                continue
        tok.train(texts, vocab_size=vocab_size, min_frequency=min_frequency, verbose=verbose)
        return tok

    # --------------------------------------------------------------- encoding
    def _split_specials(self, text: str) -> List[str]:
        """Split text so that special-token strings become their own chunks."""
        if self._special_regex is None:
            return [text]
        chunks: List[str] = []
        last = 0
        for m in self._special_regex.finditer(text):
            if m.start() > last:
                chunks.append(text[last : m.start()])
            chunks.append(m.group(0))
            last = m.end()
        if last < len(text):
            chunks.append(text[last:])
        return chunks

    def _encode_word(self, word: str) -> List[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = list(word.encode("utf-8"))
        if self.merge_ranks:
            while len(ids) >= 2:
                best_rank = None
                best_pair = None
                for pair in zip(ids, ids[1:]):
                    rank = self.merge_ranks.get(pair)
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank, best_pair = rank, pair
                if best_pair is None:
                    break
                ids = _merge_word(ids, best_pair, self.merge_base + best_rank)
        self._cache[word] = ids
        return ids

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        allow_special: bool = True,
    ) -> List[int]:
        """Encode a string to token ids.

        If ``allow_special`` is true, literal occurrences of special-token
        strings in ``text`` are mapped to their ids; otherwise they are encoded
        as ordinary text.
        """
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        chunks = self._split_specials(text) if allow_special else [text]
        for chunk in chunks:
            sid = self.special_to_id.get(chunk) if allow_special else None
            if sid is not None:
                ids.append(sid)
                continue
            for m in self._regex.finditer(chunk):
                ids.extend(self._encode_word(m.group(0)))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        buf = bytearray()
        parts: List[str] = []
        for i in ids:
            i = int(i)
            if i in self.id_to_special:
                if skip_special:
                    continue
                if buf:
                    parts.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                parts.append(self.id_to_special[i])
            else:
                piece = self.vocab.get(i)
                if piece is None:
                    raise ValueError(f"token id {i} is out of vocabulary (size {self.vocab_size})")
                buf.extend(piece)
        if buf:
            parts.append(buf.decode("utf-8", errors="replace"))
        return "".join(parts)

    def token_str(self, i: int) -> str:
        """Human-readable form of a single token (for debugging)."""
        if i in self.id_to_special:
            return self.id_to_special[i]
        return self.vocab[i].decode("utf-8", errors="replace")

    # -------------------------------------------------------------- save/load
    def to_dict(self) -> dict:
        return {
            "model": "mnx-bpe",
            "version": 1,
            "pattern": self.pattern,
            "special_tokens": self.special_tokens,
            "merges": [list(p) for p in self.merges],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        if d.get("model") != "mnx-bpe":
            raise ValueError("not an mnx-bpe tokenizer file")
        return cls(
            merges=[tuple(p) for p in d["merges"]],
            special_tokens=d["special_tokens"],
            pattern=d.get("pattern", DEFAULT_PATTERN),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ----------------------------------------------------------------------- CLI
def iter_source_files(root: Path, extensions: Optional[Sequence[str]] = None) -> List[Path]:
    """Return all files under ``root`` (or ``root`` itself if it's a file)."""
    root = Path(root)
    if root.is_file():
        return [root]
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if extensions:
        exts = {e if e.startswith(".") else "." + e for e in extensions}
        files = [p for p in files if p.suffix in exts]
    return files


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train or inspect an MNX byte-level BPE tokenizer.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train a tokenizer on a directory or file")
    p_train.add_argument("--data", required=True, help="directory of source files or a single text file")
    p_train.add_argument("--vocab-size", type=int, default=1024)
    p_train.add_argument("--min-frequency", type=int, default=2)
    p_train.add_argument("--out", required=True, help="output tokenizer.json path")
    p_train.add_argument("--ext", nargs="*", default=None, help="restrict to file extensions, e.g. .py .js")
    p_train.add_argument("--verbose", action="store_true")

    p_enc = sub.add_parser("encode", help="encode a string and print token ids/pieces")
    p_enc.add_argument("--tokenizer", required=True)
    p_enc.add_argument("--text", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "train":
        files = iter_source_files(Path(args.data), args.ext)
        if not files:
            raise SystemExit(f"no files found under {args.data}")
        tok = BPETokenizer.train_from_files(files, args.vocab_size, args.min_frequency, verbose=args.verbose)
        tok.save(args.out)
        print(f"trained tokenizer on {len(files)} files -> {args.out} (vocab_size={tok.vocab_size})")
    elif args.cmd == "encode":
        tok = BPETokenizer.load(args.tokenizer)
        ids = tok.encode(args.text)
        print(ids)
        print([tok.token_str(i) for i in ids])


if __name__ == "__main__":
    main()
