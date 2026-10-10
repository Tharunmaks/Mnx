"""Tokenise a large corpus into flat binary shards for pretraining.

Example::

    python -m mnx.prepare --tokenizer runs/tok/tokenizer.json \\
        --input /data/code --out /data/mnx-shards --workers 64

Inputs are any mix of:

* source files under a directory (one file = one document), and
* ``.jsonl`` files with one JSON object per line whose ``--text-field``
  (default ``content``, falling back to ``text``) holds a document, which is
  the layout of most Hugging Face code datasets.

Every document becomes ``<|bos|> doc <|eos|>`` (a fraction ``--fim-rate`` is
rewritten into the PSM fill-in-the-middle layout first) and documents are
appended to ``uint16`` shards of roughly ``--shard-tokens`` tokens:
``train_00000.bin``, ``train_00001.bin``, ... plus ``val_00000.bin``.
``meta.json`` records the tokenizer, vocabulary size and token counts.
Shards are read back with ``numpy.memmap`` by :mod:`mnx.pretrain`.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from mnx.data import DEFAULT_CODE_EXTENSIONS, apply_fim
from mnx.tokenizer import BPETokenizer, iter_source_files

DTYPE = np.uint16
META_NAME = "meta.json"

_TOK: Optional[BPETokenizer] = None
_FIM_RATE = 0.0
_VAL_FRACTION = 0.0
_SEED = 0


def _iter_jsonl(path: Path, field: str) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(field) if field in obj else obj.get("text")
            if isinstance(text, str) and text.strip():
                yield text


def iter_documents(inputs: Sequence[str], text_field: str, extensions: Sequence[str]) -> Iterator[str]:
    """Yield documents from directories, source files and ``.jsonl`` files."""
    for item in inputs:
        root = Path(item)
        files = [root] if root.is_file() else iter_source_files(root, (*extensions, ".jsonl"))
        for f in files:
            if f.suffix == ".jsonl":
                yield from _iter_jsonl(f, text_field)
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if text.strip():
                yield text


def _init_worker(tokenizer_path: str, fim_rate: float, val_fraction: float, seed: int) -> None:
    global _TOK, _FIM_RATE, _VAL_FRACTION, _SEED
    _TOK = BPETokenizer.load(tokenizer_path)
    _FIM_RATE = fim_rate
    _VAL_FRACTION = val_fraction
    _SEED = seed


def _encode_doc(text: str) -> Tuple[bool, np.ndarray]:
    """Return ``(is_validation, tokens)`` for one document."""
    tok = _TOK
    assert tok is not None
    if _is_val(text, _VAL_FRACTION, _SEED):
        return True, np.array([tok.bos_id, *tok.encode(text, allow_special=False), tok.eos_id], dtype=DTYPE)
    ids = tok.encode(text, allow_special=False)
    if _FIM_RATE > 0:
        # Seed from the content so results do not depend on worker scheduling.
        digest = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).digest()
        rng = random.Random(int.from_bytes(digest, "little") ^ _SEED)
        if rng.random() < _FIM_RATE:
            ids = apply_fim(ids, tok, rng)
    return False, np.array([tok.bos_id, *ids, tok.eos_id], dtype=DTYPE)


def _is_val(text: str, val_fraction: float, seed: int) -> bool:
    if val_fraction <= 0:
        return False
    digest = hashlib.blake2b(f"{seed}:{text}".encode("utf-8", "replace"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2**64 < val_fraction


class ShardWriter:
    """Append token arrays to ``{prefix}_{n:05d}.bin`` files of bounded size."""

    def __init__(self, out_dir: Path, prefix: str, shard_tokens: int) -> None:
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.shards: List[dict] = []
        self.total = 0
        self._buf: List[np.ndarray] = []
        self._buf_len = 0

    def add(self, arr: np.ndarray) -> None:
        self._buf.append(arr)
        self._buf_len += len(arr)
        self.total += len(arr)
        if self._buf_len >= self.shard_tokens:
            self.flush()

    def flush(self) -> None:
        if not self._buf_len:
            return
        name = f"{self.prefix}_{len(self.shards):05d}.bin"
        np.concatenate(self._buf).astype(DTYPE, copy=False).tofile(self.out_dir / name)
        self.shards.append({"file": name, "tokens": self._buf_len})
        self._buf, self._buf_len = [], 0


def prepare(
    inputs: Sequence[str],
    tokenizer_path: str,
    out: str,
    text_field: str = "content",
    extensions: Sequence[str] = DEFAULT_CODE_EXTENSIONS,
    shard_tokens: int = 100_000_000,
    val_fraction: float = 0.005,
    fim_rate: float = 0.5,
    workers: int = 1,
    seed: int = 0,
    max_docs: Optional[int] = None,
) -> dict:
    """Tokenise ``inputs`` into shards under ``out`` and return the metadata."""
    tok = BPETokenizer.load(tokenizer_path)
    if tok.vocab_size > np.iinfo(DTYPE).max + 1:
        raise ValueError(f"vocab_size {tok.vocab_size} does not fit in {DTYPE.__name__}")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(out_dir / "tokenizer.json")

    train_w = ShardWriter(out_dir, "train", shard_tokens)
    val_w = ShardWriter(out_dir, "val", shard_tokens)
    n_docs = 0
    t0 = time.time()

    def docs() -> Iterator[str]:
        for i, d in enumerate(iter_documents(inputs, text_field, extensions)):
            if max_docs is not None and i >= max_docs:
                return
            yield d

    def consume(results: Iterator[Tuple[bool, np.ndarray]]) -> None:
        nonlocal n_docs
        for is_val, arr in results:
            (val_w if is_val else train_w).add(arr)
            n_docs += 1
            if n_docs % 10_000 == 0:
                rate = (train_w.total + val_w.total) / max(1e-9, time.time() - t0)
                print(f"{n_docs} docs, {train_w.total + val_w.total:,} tokens ({rate:,.0f} tok/s)", flush=True)

    init_args = (tokenizer_path, fim_rate, val_fraction, seed)
    if workers > 1:
        # Pool.imap drains its input eagerly, so feed it bounded batches to
        # keep memory flat on corpora far larger than RAM.
        batch = workers * 512
        with Pool(workers, initializer=_init_worker, initargs=init_args) as pool:
            it = docs()
            while chunk := list(itertools.islice(it, batch)):
                consume(pool.imap(_encode_doc, chunk, chunksize=64))
    else:
        _init_worker(*init_args)
        consume(map(_encode_doc, docs()))

    train_w.flush()
    val_w.flush()
    if not train_w.shards:
        raise ValueError("no training documents found")
    meta = {
        "tokenizer": "tokenizer.json",
        "vocab_size": tok.vocab_size,
        "dtype": "uint16",
        "documents": n_docs,
        "fim_rate": fim_rate,
        "train_tokens": train_w.total,
        "val_tokens": val_w.total,
        "train": train_w.shards,
        "val": val_w.shards,
    }
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2))
    print(
        f"done: {n_docs} docs -> {train_w.total:,} train / {val_w.total:,} val tokens "
        f"in {len(train_w.shards)} + {len(val_w.shards)} shards ({time.time() - t0:.1f}s)"
    )
    return meta


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Tokenise a corpus into binary shards for mnx.pretrain.")
    p.add_argument("--input", nargs="+", required=True, help="directories, source files or .jsonl files")
    p.add_argument("--tokenizer", required=True, help="tokenizer.json from `python -m mnx.tokenizer train`")
    p.add_argument("--out", required=True, help="output directory for shards and meta.json")
    p.add_argument("--text-field", default="content", help="JSONL field holding the document (fallback: text)")
    p.add_argument("--shard-tokens", type=int, default=100_000_000)
    p.add_argument("--val-fraction", type=float, default=0.005)
    p.add_argument("--fim-rate", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-docs", type=int, default=None)
    a = p.parse_args(argv)
    prepare(
        a.input, a.tokenizer, a.out, text_field=a.text_field, shard_tokens=a.shard_tokens,
        val_fraction=a.val_fraction, fim_rate=a.fim_rate, workers=a.workers, seed=a.seed,
        max_docs=a.max_docs,
    )


if __name__ == "__main__":
    main()
