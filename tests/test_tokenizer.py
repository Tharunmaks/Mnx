import json

import pytest

from mnx.tokenizer import N_BYTES, SPECIAL_TOKENS, BPETokenizer


def test_untrained_tokenizer_is_pure_bytes():
    tok = BPETokenizer()
    assert tok.vocab_size == N_BYTES + len(SPECIAL_TOKENS)
    text = "hello"
    assert tok.encode(text) == list(text.encode("utf-8"))
    assert tok.decode(tok.encode(text)) == text


@pytest.mark.parametrize(
    "text",
    [
        "def add(a, b):\n    return a + b\n",
        "class Foo:\n\tdef __init__(self):\n\t\tself.x = 1\n",
        "x = [i ** 2 for i in range(10)]  # comment",
        "unicode: héllo wörld 🎉 — ünïcödé",
        "   leading spaces and trailing   ",
        "",
        "\n\n\n",
    ],
)
def test_round_trip(tokenizer, text):
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert all(0 <= i < tokenizer.vocab_size for i in ids)


def test_training_learns_merges_and_compresses(tokenizer, sample_dir):
    assert tokenizer.vocab_size == 512
    assert len(tokenizer.merges) == 512 - tokenizer.merge_base
    text = (sample_dir / "fibonacci.py").read_text()
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(tokenizer.encode(text))
    assert n_tokens < n_bytes * 0.6, "BPE should compress code noticeably"
    # Common keyword should become a single token.
    assert len(tokenizer.encode("def")) == 1
    assert len(tokenizer.encode(" return")) == 1


def test_special_tokens(tokenizer):
    ids = tokenizer.encode("a<|eos|>b", allow_special=True)
    assert ids == [ord("a"), tokenizer.eos_id, ord("b")]
    assert tokenizer.decode(ids) == "a<|eos|>b"
    assert tokenizer.decode(ids, skip_special=True) == "ab"

    # With allow_special=False the literal string is encoded as plain text.
    plain = tokenizer.encode("<|eos|>", allow_special=False)
    assert tokenizer.eos_id not in plain
    assert tokenizer.decode(plain) == "<|eos|>"

    ids = tokenizer.encode("x", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id and ids[-1] == tokenizer.eos_id

    fim = {tokenizer.fim_prefix_id, tokenizer.fim_middle_id, tokenizer.fim_suffix_id}
    assert len(fim) == 3
    assert all(N_BYTES <= i < tokenizer.merge_base for i in fim)


def test_save_and_load(tokenizer, tmp_path):
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    data = json.loads(path.read_text())
    assert data["model"] == "mnx-bpe"
    loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tokenizer.vocab_size
    text = "def foo(bar):\n    return bar * 2\n"
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_encode_is_deterministic_and_cached(tokenizer):
    text = "for i in range(10):\n    print(i)\n"
    assert tokenizer.encode(text) == tokenizer.encode(text)


def test_cli_train_and_encode(tmp_path, sample_dir, capsys):
    from mnx.tokenizer import main

    out = tmp_path / "tokenizer.json"
    main(["train", "--data", str(sample_dir), "--vocab-size", "400", "--out", str(out)])
    assert out.exists()
    main(["encode", "--tokenizer", str(out), "--text", "def f(x): return x"])
    captured = capsys.readouterr().out
    assert "def" in captured
