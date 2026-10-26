"""Expert stream, Q&A database and streaming training (the latter needs torch)."""

import ast
import re
from fractions import Fraction

import pytest

from mnx.agent.expert_stream import _held_out_names, format_doc, generated, prompt_for, recipes
from mnx.agent.qa_stream import GENERATORS, read, stream, write
from mnx.agent.skills.neural import run_tests


def test_generated_code_is_valid_and_passes_its_asserts():
    docs = list(generated(40, seed=1))
    assert len(docs) == 40
    for doc in docs:
        tests = [l[2:] for l in doc.splitlines() if l.startswith("# assert")]
        code = "\n".join(l for l in doc.splitlines() if not l.startswith("#")) + "\n"
        ast.parse(code)
        if tests:
            assert run_tests(code, tests)[0], doc


def test_holdout_names_are_never_generated():
    banned = _held_out_names()
    assert {"cube", "is_even", "add_one", "shout", "factorial"} <= banned
    names = {re.search(r"def (\w+)\(", d).group(1) for d in generated(3000, seed=2, banned=banned)}
    assert not names & banned


def test_prompt_format_matches_training_format():
    doc = format_doc("def f(x):", "Double it.", "    return 2 * x\n", ["assert f(2) == 4"])
    prompt = prompt_for('def f(x):\n    """Double it."""\n', ["assert f(2) == 4"])
    assert doc.startswith(prompt)
    assert next(recipes(1)).strip()


def test_qa_pairs_are_distinct_and_correct():
    items = list(stream(3000, seed=7))
    assert len({it["q"] for it in items}) == 3000
    assert {it["agent"] for it in items} <= {"mathematician", "reasoner", "coder"}
    for it in items:
        if it["kind"] == "linear":
            lhs, rhs = it["q"][len("solve "):].split(" = ")
            x = int(it["a"].split("= ")[1])
            expr = re.sub(r"(\d)x", r"\1*x", lhs).replace("-x", "-1*x")
            assert eval(expr, {"x": x}) == int(rhs)
        if it["kind"] == "arithmetic" and " divided by " not in it["q"] and "/" not in it["q"]:
            expr = re.sub(r"^(what is|calculate) ", "", it["q"]).replace(" = ?", "").replace("plus", "+").replace(
                "minus", "-").replace("times", "*")
            assert str(eval(expr)) == it["a"], it


def test_qa_database_roundtrip(tmp_path):
    stats = write(1200, tmp_path, seed=3, shard_size=500)
    assert stats["pairs"] == 1200 and len(list(tmp_path.glob("qa-*.jsonl.gz"))) == 3
    assert sum(1 for _ in read(tmp_path)) == 1200
    assert abs(sum(w for _, _, w in GENERATORS.values()) - 1.0) < 1e-9


def test_mnx1_agrees_with_the_database(tmp_path):
    from mnx.agent.qa_stream import check

    write(3000, tmp_path, seed=5)
    res = check(tmp_path, sample=60, seed=1)
    assert res["checked"] > 20
    assert res["agreed"] >= 0.9 * res["mnx1_answered"]


def test_stream_dataset_uses_each_token_once():
    torch = pytest.importorskip("torch")
    from mnx.data import StreamDataset
    from mnx.tokenizer import BPETokenizer

    docs = [f"Q: what is {i} + 1\nA: {i + 1}\n" for i in range(400)]
    tok = BPETokenizer()
    tok.train(docs, vocab_size=300)
    ds = StreamDataset(iter(docs), tok, seq_len=16, buffer_docs=50)
    x1, y1 = ds.random_batch(4)
    x2, _ = ds.random_batch(4)
    assert x1.shape == (4, 16) and torch.equal(x1[:, 1:], y1[:, :-1])
    assert not torch.equal(x1, x2)
    assert ds.tokens_seen == 128 and ds.docs_seen >= 50


def test_streaming_training_and_resume(tmp_path):
    pytest.importorskip("torch")
    import json

    from mnx.train import parse_args, train

    qa = tmp_path / "qa"
    write(50000, qa, seed=9)
    out = tmp_path / "run"
    common = ["--config", "tiny", "--stream", str(qa), "--data", "", "--out", str(out), "--batch-size", "2",
              "--seq-len", "64", "--eval-interval", "0", "--log-interval", "5", "--save-interval", "5", "--vocab-size", "300"]
    train(parse_args(common + ["--steps", "5"]))
    state = json.loads((out / "stream_state.json").read_text())
    assert state["step"] == 5 and state["docs_seen"] > 0
    train(parse_args(common + ["--steps", "8", "--resume"]))
    assert json.loads((out / "stream_state.json").read_text())["docs_seen"] > state["docs_seen"]


def test_bugs_found_by_the_database_stay_fixed():
    from mnx.agent.skills.math import MathSkill

    m = MathSkill()
    assert m.answer("19 * 216 = ?").text == "4104"
    assert m.answer("gcd of 413 and 497").text == "7"
    assert m.answer("least common multiple of 60 and 395").text == "4740"


def test_coder_reads_examples_from_a_docstring():
    from mnx.agent.skills.code import CodeSkill, docstring_tests

    q = ('def triple(x):\n    """Return x multiplied by three.\n\n    >>> triple(2)\n    6\n'
         '    >>> triple(5)\n    15\n    >>> triple(0)\n    0\n    """\n')
    assert docstring_tests(q, "triple") == ["assert triple(2) == 6", "assert triple(5) == 15", "assert triple(0) == 0"]
    ans = CodeSkill().answer(q)  # no assert lines: the examples in the docstring are the tests
    assert ans is not None and ans.verified and "def triple(x)" in ans.text
