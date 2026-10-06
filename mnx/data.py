"""Dataset utilities: corpus loading, token packing and fill-in-the-middle.

Documents are tokenised, optionally rewritten into the FIM "PSM" layout
(prefix, suffix, middle), separated with ``<|eos|>`` and packed into one long
token stream from which fixed-length training windows are cut.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from mnx.tokenizer import BPETokenizer, iter_source_files

DEFAULT_CODE_EXTENSIONS: Tuple[str, ...] = (
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp", ".cc",
    ".rs", ".go", ".rb", ".php", ".cs", ".swift", ".kt", ".scala", ".sh", ".sql",
    ".html", ".css", ".json", ".yaml", ".yml", ".toml", ".md", ".txt",
)


# ------------------------------------------------------------------ loading
def load_corpus(
    path: str | Path,
    extensions: Optional[Sequence[str]] = DEFAULT_CODE_EXTENSIONS,
    doc_separator: Optional[str] = None,
) -> List[str]:
    """Load documents from a directory of source files or a single text file.

    For a directory every file with a matching extension is one document. For
    a single file the content is one document, or split on ``doc_separator``
    if given (e.g. ``"\\n<|eos|>\\n"``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        text = path.read_text(encoding="utf-8", errors="replace")
        if doc_separator:
            return [d for d in text.split(doc_separator) if d.strip()]
        return [text]
    docs: List[str] = []
    for f in iter_source_files(path, extensions):
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if text.strip():
            docs.append(text)
    if not docs:
        raise ValueError(f"no readable documents found under {path}")
    return docs


# --------------------------------------------------------------------- FIM
def apply_fim(
    ids: Sequence[int],
    tok: BPETokenizer,
    rng: random.Random,
    min_middle: int = 1,
) -> List[int]:
    """Rewrite a token sequence into PSM fill-in-the-middle format.

    ``[prefix | middle | suffix]`` becomes::

        <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|> middle

    The caller is responsible for the trailing ``<|eos|>``. Sequences too
    short to be split are returned unchanged.
    """
    ids = list(ids)
    n = len(ids)
    if n < min_middle + 2:
        return ids
    # Two cut points with at least ``min_middle`` tokens in between.
    a = rng.randint(0, n - min_middle)
    b = rng.randint(a + min_middle, n)
    prefix, middle, suffix = ids[:a], ids[a:b], ids[b:]
    return [tok.fim_prefix_id, *prefix, tok.fim_suffix_id, *suffix, tok.fim_middle_id, *middle]


def make_fim_prompt(prefix: str, suffix: str, tok: BPETokenizer) -> List[int]:
    """Build the inference-time FIM prompt; the model then emits the middle."""
    return [
        tok.fim_prefix_id,
        *tok.encode(prefix, allow_special=False),
        tok.fim_suffix_id,
        *tok.encode(suffix, allow_special=False),
        tok.fim_middle_id,
    ]


def split_fim_output(ids: Sequence[int], tok: BPETokenizer) -> List[int]:
    """Return the generated middle from a full FIM sequence (prompt + output)."""
    ids = list(ids)
    if tok.fim_middle_id in ids:
        ids = ids[ids.index(tok.fim_middle_id) + 1 :]
    if tok.eos_id in ids:
        ids = ids[: ids.index(tok.eos_id)]
    return ids


# ----------------------------------------------------------------- packing
def tokenize_documents(docs: Sequence[str], tok: BPETokenizer) -> List[List[int]]:
    return [tok.encode(d, allow_special=False) for d in docs]


def pack_documents(
    token_docs: Sequence[Sequence[int]],
    tok: BPETokenizer,
    fim_rate: float = 0.0,
    seed: int = 0,
    shuffle: bool = True,
) -> List[int]:
    """Concatenate documents into a single stream, each followed by ``<|eos|>``.

    A fraction ``fim_rate`` of documents is transformed with :func:`apply_fim`.
    """
    rng = random.Random(seed)
    order = list(range(len(token_docs)))
    if shuffle:
        rng.shuffle(order)
    stream: List[int] = []
    for i in order:
        ids = list(token_docs[i])
        if fim_rate > 0 and rng.random() < fim_rate:
            ids = apply_fim(ids, tok, rng)
        stream.extend(ids)
        stream.append(tok.eos_id)
    return stream


class CodeDataset(Dataset):
    """Fixed-length ``(input, target)`` windows cut from a packed token stream.

    Call :meth:`reshuffle` between epochs to re-sample document order and FIM
    transformations.
    """

    def __init__(
        self,
        token_docs: Sequence[Sequence[int]],
        tok: BPETokenizer,
        seq_len: int,
        fim_rate: float = 0.0,
        stride: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.token_docs = [list(d) for d in token_docs]
        self.tok = tok
        self.seq_len = seq_len
        self.fim_rate = fim_rate
        self.stride = stride or seq_len
        self.seed = seed
        self.stream: torch.Tensor = torch.empty(0, dtype=torch.long)
        self.reshuffle(seed)

    def reshuffle(self, seed: int) -> None:
        stream = pack_documents(self.token_docs, self.tok, self.fim_rate, seed=seed)
        if len(stream) < self.seq_len + 1:
            # Repeat a tiny corpus so at least one full window exists.
            reps = (self.seq_len + 1) // max(1, len(stream)) + 1
            stream = stream * reps
        self.stream = torch.tensor(stream, dtype=torch.long)

    @property
    def num_tokens(self) -> int:
        return int(self.stream.numel())

    def __len__(self) -> int:
        return max(1, (self.num_tokens - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = i * self.stride
        chunk = self.stream[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]

    def random_batch(self, batch_size: int, generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample ``batch_size`` windows at uniformly random offsets."""
        hi = self.num_tokens - self.seq_len - 1
        starts = torch.randint(0, hi + 1, (batch_size,), generator=generator)
        xs = torch.stack([self.stream[s : s + self.seq_len] for s in starts.tolist()])
        ys = torch.stack([self.stream[s + 1 : s + self.seq_len + 1] for s in starts.tolist()])
        return xs, ys


def split_documents(docs: Sequence, val_fraction: float, seed: int = 0) -> Tuple[list, list]:
    """Deterministically split documents into train/validation lists."""
    docs = list(docs)
    if len(docs) < 2 or val_fraction <= 0:
        return docs, []
    rng = random.Random(seed)
    order = list(range(len(docs)))
    rng.shuffle(order)
    n_val = max(1, int(round(len(docs) * val_fraction)))
    val_idx = set(order[:n_val])
    train = [docs[i] for i in range(len(docs)) if i not in val_idx]
    val = [docs[i] for i in range(len(docs)) if i in val_idx]
    return train, val


def iter_batches(dataset: CodeDataset, batch_size: int, shuffle: bool = True, seed: int = 0) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield mini-batches covering every window once (one epoch)."""
    idx = list(range(len(dataset)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        items = [dataset[j] for j in idx[i : i + batch_size]]
        xs = torch.stack([x for x, _ in items])
        ys = torch.stack([y for _, y in items])
        yield xs, ys
