import random

import torch

from mnx.data import (
    CodeDataset,
    apply_fim,
    iter_batches,
    load_corpus,
    make_fim_prompt,
    pack_documents,
    split_documents,
    split_fim_output,
    tokenize_documents,
)


def test_load_corpus(sample_dir):
    docs = load_corpus(sample_dir)
    assert len(docs) >= 30
    assert all(isinstance(d, str) and d.strip() for d in docs)


def test_apply_fim_psm_layout(tokenizer):
    ids = list(range(10, 30))  # 20 ordinary tokens
    rng = random.Random(3)
    out = apply_fim(ids, tokenizer, rng)
    # Exactly three special tokens inserted, content preserved as a multiset.
    assert len(out) == len(ids) + 3
    assert out[0] == tokenizer.fim_prefix_id
    assert out.count(tokenizer.fim_suffix_id) == 1
    assert out.count(tokenizer.fim_middle_id) == 1
    s = out.index(tokenizer.fim_suffix_id)
    m = out.index(tokenizer.fim_middle_id)
    assert 0 < s < m
    prefix, suffix, middle = out[1:s], out[s + 1 : m], out[m + 1 :]
    assert len(middle) >= 1
    assert prefix + middle + suffix == ids


def test_apply_fim_is_random_but_seeded(tokenizer):
    ids = list(range(50, 90))
    a = apply_fim(ids, tokenizer, random.Random(0))
    b = apply_fim(ids, tokenizer, random.Random(0))
    c = apply_fim(ids, tokenizer, random.Random(1))
    assert a == b
    assert a != c


def test_apply_fim_short_sequence_unchanged(tokenizer):
    assert apply_fim([1], tokenizer, random.Random(0)) == [1]
    assert apply_fim([], tokenizer, random.Random(0)) == []


def test_fim_prompt_and_split(tokenizer):
    prompt = make_fim_prompt("def f(x):\n    ", "\n    return y\n", tokenizer)
    assert prompt[0] == tokenizer.fim_prefix_id
    assert prompt[-1] == tokenizer.fim_middle_id
    middle = tokenizer.encode("y = x * 2")
    full = prompt + middle + [tokenizer.eos_id, 99, 98]
    assert split_fim_output(full, tokenizer) == middle


def test_pack_documents_adds_eos_and_fim(tokenizer):
    docs = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    stream = pack_documents(docs, tokenizer, fim_rate=0.0, seed=0, shuffle=False)
    assert stream == [1, 2, 3, 4, 5, 6, tokenizer.eos_id, 7, 8, 9, 10, 11, 12, tokenizer.eos_id]
    fim_stream = pack_documents(docs, tokenizer, fim_rate=1.0, seed=0, shuffle=False)
    assert fim_stream.count(tokenizer.eos_id) == 2
    assert fim_stream.count(tokenizer.fim_prefix_id) == 2
    assert len(fim_stream) == len(stream) + 2 * 3


def test_code_dataset_windows(tokenizer, sample_dir):
    docs = tokenize_documents(load_corpus(sample_dir), tokenizer)
    ds = CodeDataset(docs, tokenizer, seq_len=32, fim_rate=0.5, seed=0)
    assert len(ds) > 10
    x, y = ds[0]
    assert x.shape == (32,) and y.shape == (32,)
    assert torch.equal(x[1:], y[:-1])  # targets are inputs shifted by one
    xb, yb = ds.random_batch(4, torch.Generator().manual_seed(0))
    assert xb.shape == (4, 32) and yb.shape == (4, 32)
    assert torch.equal(xb[:, 1:], yb[:, :-1])
    before = ds.stream.clone()
    ds.reshuffle(1)
    assert ds.stream.numel() > 0 and not torch.equal(before[: min(100, ds.num_tokens)], ds.stream[:100])


def test_code_dataset_repeats_tiny_corpus(tokenizer):
    ds = CodeDataset([[1, 2, 3]], tokenizer, seq_len=16)
    assert ds.num_tokens >= 17
    x, y = ds[0]
    assert x.shape == (16,)


def test_split_documents_and_iter_batches(tokenizer):
    docs = [[i] * 8 for i in range(10)]
    train, val = split_documents(docs, 0.2, seed=0)
    assert len(train) == 8 and len(val) == 2
    ds = CodeDataset(train, tokenizer, seq_len=8)
    batches = list(iter_batches(ds, batch_size=3, seed=0))
    assert sum(b[0].shape[0] for b in batches) == len(ds)
