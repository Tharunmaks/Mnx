"""Tests for the MNX agent (standard library only, no PyTorch needed)."""

import datetime as dt
from fractions import Fraction

import pytest

from mnx.agent import ABSTAIN, MNXAgent
from mnx.agent.expr import MathError, Poly, evaluate, parse, to_poly
from mnx.agent.retrieval import BM25Index, Memory, tokenize
from mnx.agent.router import cross_validate, default_intents_path, load_intents, train_default_router
from mnx.agent.skills.dates import DateSkill, _zeller
from mnx.agent.skills.logic import LogicSkill, singular
from mnx.agent.skills.math import MathSkill, factorize, is_prime
from mnx.agent.skills.neural import NeuralCodeSkill, assemble, run_tests, split_prompt_and_tests
from mnx.agent.skills.sequences import SequenceSkill, predict_next
from mnx.agent.skills.synth import SynthSkill, parse_examples, run_program, synthesize
from mnx.agent.skills.units import UnitSkill


@pytest.fixture(scope="module")
def agent():
    return MNXAgent(today=dt.date(2026, 1, 1), learn=False)


# ------------------------------------------------------------------ algebra
@pytest.mark.parametrize(
    "text,value",
    [("2 + 3 * 4", 14), ("-2^2", -4), ("2^-1", Fraction(1, 2)), ("2(3 + 4)", 14), ("5!", 120),
     ("sqrt(49)", 7), ("1/3 + 1/6", Fraction(1, 2)), ("gcd(12, 18)", 6), ("10 % 4", 2), ("8^(1/3)", 2)],
)
def test_evaluate_exact(text, value):
    assert evaluate(parse(text)) == value


def test_evaluate_refuses_bad_input():
    with pytest.raises(MathError):
        evaluate(parse("1/0"))
    with pytest.raises(MathError):
        evaluate(parse("10^10^10"))
    with pytest.raises(MathError):
        parse("2 apples")


def test_poly_expand_and_calculus():
    p = to_poly(parse("(x + 1)^3"))
    assert str(p) == "x^3 + 3x^2 + 3x + 1"
    assert str(p.derivative("x")) == "3x^2 + 6x + 3"
    assert p.integral("x").derivative("x") == p
    assert to_poly(parse("(x+y)(x-y)")) == to_poly(parse("x^2 - y^2"))
    assert Poly.const(Fraction(0)).terms == {}


@pytest.mark.parametrize(
    "q,expected",
    [
        ("what is 15% of 80", "12"),
        ("solve x^2 - 5x + 6 = 0", "x = 2 or x = 3"),
        ("x^2 + 1 = 0", "x = i or x = -i"),
        ("solve x + y = 10, x - y = 2", "x = 6, y = 4"),
        ("x^3 - 6x^2 + 11x - 6 = 0", "x = 1 or x = 2 or x = 3"),
        ("factor 2x^2 + 3x - 2", "(2x - 1)(x + 2)"),
        ("derivative of x^3 + 2x", "3x^2 + 2"),
        ("what is 3 plus 4 times 2", "11"),
        ("100!", "9332621544394415268169923885626670049071596826438162146859296389521759999322991560894146397615651828625369792082722375825118521091686400000000000000000000000"),
    ],
)
def test_math_skill(q, expected):
    ans = MathSkill().answer(q)
    assert ans is not None and expected in ans.text
    assert ans.verified


def test_math_skill_verifies_irrational_and_numeric_roots():
    ans = MathSkill().answer("x^2 = 2")
    assert ans.verified and "1.414213562" in ans.text
    ans = MathSkill().answer("x^5 - x - 1 = 0")  # no closed form: Durand-Kerner + residual check
    assert ans.verified and "1.167303978" in ans.text


def test_math_skill_ignores_non_math():
    assert MathSkill().answer("what is the capital of France") is None


def test_primes_and_factorization():
    assert is_prime(2**61 - 1) and not is_prime(561)
    n = 600851475143
    f = factorize(n)
    assert f == {71: 1, 839: 1, 1471: 1, 6857: 1}
    assert factorize(2**64 + 1) == {274177: 1, 67280421310721: 1}


# --------------------------------------------------------- units and dates
@pytest.mark.parametrize(
    "q,expected",
    [("convert 5 km to miles", "3.10686 miles"), ("100 F in C", "37.7778"), ("how many feet are in a mile", "5280 feet"),
     ("1 GB in MB", "1000 MB"), ("what is 0 kelvin in fahrenheit", "-459.67")],
)
def test_units(q, expected):
    ans = UnitSkill().answer(q)
    assert ans is not None and expected in ans.text and ans.verified


def test_units_dimension_mismatch():
    assert "different dimensions" in UnitSkill().answer("5 km to kg").text


def test_dates():
    d = DateSkill(today=dt.date(2026, 1, 1))
    assert "Thursday" in d.answer("What day of the week was July 4, 1776?").text
    assert "359 days" in d.answer("how many days between 2024-01-01 and 2024-12-25").text
    assert "April 10, 2024" in d.answer("what date is 100 days after 2024-01-01").text
    assert d.answer("is 1900 a leap year").text.startswith("No")
    assert d.answer("is 2000 a leap year").text.startswith("Yes")
    assert "valid date" in d.answer("what day of the week was 2023-02-30").text


def test_zeller_matches_datetime():
    day = dt.date(1900, 1, 1)
    for _ in range(0, 80000, 97):
        assert _zeller(day) == day.strftime("%A")
        day += dt.timedelta(days=97)


# -------------------------------------------------------------------- logic
@pytest.mark.parametrize(
    "q,expected",
    [
        ("All men are mortal. Socrates is a man. Is Socrates mortal?", "Yes, Socrates is mortal."),
        ("No reptiles have fur. All snakes are reptiles. Kaa is a snake. Does Kaa have fur?", "No, Kaa does not have fur."),
        ("If something is a dog then it is loyal. Rex is not loyal. Is Rex a dog?", "No, Rex is not a dog."),
        ("All cats are mammals. All mammals are animals. Are all cats animals?", "Yes, every cat is an animal."),
        ("All dolphins are mammals. All mammals breathe air. Flipper is a dolphin. Does Flipper breathe air?", "Yes, Flipper breathes air."),
        ("All cats are mammals. Tom is a dog. Is Tom a mammal?", "Cannot be determined"),
        ("Every dog is an animal. Fido is a dog. Fido is not an animal. Is Fido a dog?", "contradictory"),
    ],
)
def test_logic(q, expected):
    ans = LogicSkill().answer(q)
    assert ans is not None and expected in ans.text


def test_logic_proof_trace():
    ans = LogicSkill().answer("All A are B. All B are C. X is an A. Is X a C?")
    assert ans.text.startswith("Yes")
    assert any("proof" in t for t in ans.trace)


def test_singular():
    assert [singular(w) for w in ["men", "cats", "bodies", "boxes", "glass", "species"]] == ["man", "cat", "body", "box", "glass", "species"]


# ---------------------------------------------------------------- sequences
@pytest.mark.parametrize(
    "terms,nxt",
    [([2, 4, 8, 16], 32), ([1, 1, 2, 3, 5, 8], 13), ([1, 4, 9, 16, 25], 36), ([3, 7, 11, 15], 19),
     ([1, 3, 7, 15, 31], 63), ([1, 2, 4, 7, 11, 16], 22), ([2, 3, 5, 7, 11], 13), ([1, 2, 6, 24, 120], 720)],
)
def test_sequences(terms, nxt):
    res = predict_next([Fraction(t) for t in terms])
    assert res is not None and res[0] == nxt and res[2]


def test_sequence_skill_admits_failure():
    ans = SequenceSkill().answer("1, 5, 2, 8, 3 next")
    assert ans.failed and not ans.verified


# ---------------------------------------------------------------- synthesis
def test_parse_examples():
    ex, qs = parse_examples("f(1, 2) = 3, f(2, 2) = 4, f(5, 5) = ?")
    assert ex == [((1, 2), 3), ((2, 2), 4)] and qs == [(5, 5)]
    ex, qs = parse_examples("[1,2] -> [2,1]; 'a, b' -> 'b, a'")
    assert ex == [(([1, 2],), [2, 1]), (("a, b",), "b, a")] and qs == []


@pytest.mark.parametrize(
    "examples,query,expected",
    [
        ([((1,), 2), ((2,), 4), ((3,), 6)], (10,), 20),
        ([(([3, 1, 2],), [1, 2, 3]), (([9, 8],), [8, 9])], ([5, 4, 6],), [4, 5, 6]),
        ([(("hello",), "HELLO"), (("abc",), "ABC")], ("xyz",), "XYZ"),
        ([(([1, 2, 3, 4],), [2, 4]), (([5, 6, 7, 8, 10],), [6, 8, 10])], ([11, 12],), [12]),
        ([((3,), 6), ((4,), 24), ((5,), 120)], (6,), 720),
        ([(("John Smith",), "JS"), (("Ada Lovelace",), "AL")], ("Alan Turing",), "AT"),
    ],
)
def test_synthesize(examples, query, expected):
    res = synthesize(examples, [query], time_limit=10)
    assert res is not None
    for args, out in examples:
        assert run_program(res.source, args) == out
    assert run_program(res.source, query) == expected


def test_synth_voting_prefers_majority_behaviour():
    ans = SynthSkill().answer("f(1)=0, f(2)=1, f(3)=0, f(4)=1, f(9)=?")
    assert "f(9) = 0" in ans.text and ans.verified


def test_synth_sandbox_has_no_builtins():
    with pytest.raises(Exception):
        run_program("def f(x):\n    return __import__('os')\n", (1,))


# ---------------------------------------------------------------- retrieval
def test_bm25_ranks_relevant_chunk_first():
    idx = BM25Index()
    idx.add_text("# Fruit\n\nBananas are yellow.\n\n# Space\n\nMars is red.", "t.md")
    (score, top), *_ = idx.search("what colour is mars")
    assert "Mars" in top.text and top.heading == "Space"
    assert tokenize("Running dogs") == ["runn", "dog"]


def test_memory_persists(tmp_path):
    path = tmp_path / "mem.jsonl"
    a = MNXAgent(knowledge_dirs=[], memory_path=path, learn=False)
    assert "remember" in a.ask("remember that my name is Ada").text
    b = MNXAgent(knowledge_dirs=[], memory_path=path, learn=False)
    assert "Your name is Ada" in b.ask("what is my name?").text
    assert Memory(path).facts == ["my name is Ada"]


# ------------------------------------------------------------------- router
def test_router_learns_intents():
    r = train_default_router()
    assert r.predict("solve 3x + 1 = 10")[0][0] == "math"
    assert r.predict("what is the capital of peru")[0][0] == "knowledge"
    assert r.predict("convert 3 miles to km")[0][0] == "units"
    assert cross_validate(load_intents(default_intents_path())) > 0.7


# -------------------------------------------------------------------- agent
@pytest.mark.parametrize(
    "q,expected,skill",
    [
        ("What is 2+2?", "4", "math"),
        ("What is the capital of France?", "Paris", "knowledge"),
        ("convert 5 km to miles", "3.10686", "units"),
        ("All men are mortal. Socrates is a man. Is Socrates mortal?", "Yes", "logic"),
        ("2, 4, 8, 16, ?", "32", "sequence"),
        ("[1,2,3] -> [2,4,6], [5] -> [10], [7] -> ?", "[14]", "synth"),
        ("hello", "Hello", "chat"),
    ],
)
def test_agent_routes_and_answers(agent, q, expected, skill):
    ans = agent.ask(q)
    assert ans.skill == skill and expected in ans.text
    assert ans.trace[0].startswith("agent:") and ans.trace[1].startswith("router:")


@pytest.mark.parametrize("q", ["Write a poem about the sea", "Who will win the next election?", "What is the capital of Atlantis?"])
def test_agent_abstains_instead_of_guessing(agent, q):
    ans = agent.ask(q)
    assert ans.skill == "abstain" and ans.text == ABSTAIN


def test_benchmark_has_no_wrong_answers():
    from mnx.agent.evaluate import load_benchmark, run

    stats = run(load_benchmark())
    assert sum(s["wrong"] for s in stats.values()) == 0


def test_cli(capsys):
    from mnx.agent.__main__ import main

    assert main(["--no-learn", "--trace", "solve", "2x", "+", "3", "=", "11"]) == 0
    out = capsys.readouterr().out
    assert "x = 4" in out and "verified" in out


# ------------------------------------------------------- neural proposer
def test_neural_skill_keeps_only_passing_samples():
    q = "def square(x):\n    \"\"\"Return x squared.\"\"\"\nassert square(3) == 9\nassert square(-2) == 4\n"
    prompt, tests = split_prompt_and_tests(q)
    assert tests == ["assert square(3) == 9", "assert square(-2) == 4"]
    samples = ["    return x + x\n", "    return x * x\nprint('trailing junk')\n"]
    ans = NeuralCodeSkill(lambda p, k: samples, k=2).answer(q)
    assert ans.verified and "return x * x" in ans.text and "trailing junk" not in ans.text
    assert assemble(prompt, samples[1]).endswith("return x * x\n")


def test_run_tests_sandbox_times_out():
    ok, err = run_tests("def f():\n    while True: pass\n", ["assert f()"], timeout=1)
    assert not ok and err == "timeout"


# ------------------------------------------------------------ coding agent
from mnx.agent.skills.code import CodeSkill, RecipeBook, default_recipe_dir, parse_asserts, undefined_names  # noqa: E402


def test_every_recipe_passes_its_own_tests():
    book = RecipeBook([default_recipe_dir()])
    assert len(book.recipes) >= 70
    for r in book.recipes:
        ok, err = run_tests(r.code, r.tests, timeout=20)
        assert ok, f"{r.name}: {err}"


@pytest.mark.parametrize(
    "q,expected",
    [
        ("write a python function that checks if a string is a palindrome", "def is_palindrome"),
        ("implement merge sort", "def merge_sort"),
        ("write code that converts a decimal number to binary", "def decimal_to_binary"),
        ("write a function that converts an integer to a roman numeral", "def int_to_roman"),
        ("create a stack class", "class Stack"),
        ("write a function called fib that returns the nth fibonacci number", "def fib("),
    ],
)
def test_code_agent_writes_tested_code(q, expected):
    ans = CodeSkill().answer(q)
    assert ans.verified and expected in ans.text and "Tests it passed" in ans.text


def test_code_agent_declines_instead_of_guessing():
    ans = CodeSkill().answer("write a function that sends an email via smtp")
    assert ans.failed and not ans.verified
    assert CodeSkill().answer("make a cake") is None


def test_code_agent_solves_from_asserts_and_learns(tmp_path):
    skill = CodeSkill(learned_dir=tmp_path)
    q = "def mystery(a, b):\nassert mystery(2, 3) == 13\nassert mystery(1, 1) == 2\nassert mystery(0, 4) == 16\nassert mystery(3, 0) == 9"
    guess = skill.answer(q)  # programs that fit 4 asserts still disagree elsewhere: a labelled guess, not an answer
    assert guess.failed and not guess.verified and "best guess" in guess.text and "def mystery(a, b)" in guess.text
    # enough varied asserts that the answer doesn't depend on how far the 4-second search gets on this machine
    more = "\nassert mystery(5, 2) == 29\nassert mystery(-1, 2) == 5\nassert mystery(-3, -4) == 25\nassert mystery(7, 1) == 50\nassert mystery(4, 6) == 52"
    ans = skill.answer(q + more)
    assert ans.verified and "def mystery(a, b)" in ans.text and "Passes all 9" in ans.text
    assert parse_asserts("mystery", ["assert mystery(2, 3) == 13"]) == [((2, 3), 13)]


def test_code_agent_run_and_check():
    skill = CodeSkill()
    assert "45" in skill.answer("run this:\n```python\nprint(sum(range(10)))\n```").text
    report = skill.answer("check this code:\n```python\ndef f(x):\n    return x + y\n```").text
    assert "`y` is used but never defined" in report
    assert "Syntax error on line 1" in skill.answer("debug:\n```python\ndef f(x)\n    return x\n```").text
    assert undefined_names("import os\nfor i in range(3):\n    print(os.sep, i, j)") == [("j", 3)]


# ---------------------------------------------------------------- learning
def test_learning_from_feedback_and_corrections(tmp_path):
    a = MNXAgent(state_dir=tmp_path, today=dt.date(2026, 1, 1))
    assert a.ask("what is the capital of Wakanda?").skill == "abstain"
    assert "learned" in a.ask("wrong, the answer is Birnin Zana").text
    assert "Birnin Zana" in a.ask("What's the capital of Wakanda").text
    a.ask("solve x^2 - 9 = 0")
    assert a.ask("correct").skill == "feedback"
    # everything is on disk, so a fresh agent remembers
    b = MNXAgent(state_dir=tmp_path)
    assert "Birnin Zana" in b.ask("capital of wakanda?").text
    rows = (tmp_path / "transcript.jsonl").read_text().splitlines()
    assert len(rows) == 6 and "learned_intents.tsv" in {p.name for p in tmp_path.iterdir()}


def test_router_partial_fit_changes_prediction():
    r = train_default_router()
    q = "zorblat the quux"
    before = dict(r.predict(q))["logic"]
    for _ in range(5):
        r.partial_fit("logic", q)
    assert dict(r.predict(q))["logic"] > before


def test_slow_answers_are_replayed(tmp_path):
    a = MNXAgent(state_dir=tmp_path)
    q = "[1,2,3] -> [2,4,6], [5] -> [10], [7] -> ?"
    first = a.ask(q)
    again = a.ask(q)
    assert again.text == first.text and any("experience" in t for t in again.trace)


def test_agent_refuses_images(agent):
    ans = agent.ask("generate an image of a cat")
    assert ans.skill == "chat" and "can't create images" in ans.text


def test_code_holdout_has_no_wrong_answers():
    from mnx.agent.evaluate import default_benchmark, load_benchmark, run

    stats = run(load_benchmark(default_benchmark().with_name("code_holdout.jsonl")))
    assert stats["code"]["wrong"] == 0
