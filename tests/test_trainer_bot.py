"""The 24/7 CPU trainer bot: grading, replay priority, curriculum, hot folder."""

from __future__ import annotations

import json

import pytest

from mnx.trainer_bot import (LEVELS, Item, PrioritizedReplayBuffer, QAGeneratorHub, grade_prediction,
                             numeric_match, status, token_f1)


def test_grading_is_exact_on_numbers_and_forgiving_on_words():
    assert grade_prediction("42", "42") == 1.0
    assert grade_prediction("41", "42") == 0.0           # a wrong number scores nothing
    assert grade_prediction("yes", "yes") == 1.0
    assert 0.5 < grade_prediction("the answer is 42", "42") < 1.0  # right number, extra words
    assert grade_prediction("", "42") == 0.0


def test_token_f1_and_numeric_match():
    assert token_f1("a b", "a b") == 1.0
    assert 0 < token_f1("a b c", "a b") < 1.0
    assert numeric_match("no numbers", "also none") is None
    assert numeric_match("1 and 2", "2 and 1") == 1.0
    assert numeric_match("1", "1 and 2") == 0.5


def test_a_missed_question_outranks_a_known_one():
    missed = Item("q1", "a", "arithmetic", last_loss=2.0, last_score=0.1, times_seen=1)
    known = Item("q2", "a", "arithmetic", last_loss=0.1, last_score=0.95, times_seen=20)
    assert missed.priority > known.priority
    seen_often = Item("q3", "a", "arithmetic", last_loss=0.1, last_score=0.95, times_seen=1)
    assert seen_often.priority > known.priority  # novelty decays with every showing


def test_the_replay_buffer_keeps_the_hardest_when_full():
    buf = PrioritizedReplayBuffer(capacity=2)
    buf.add(Item("hard", "a", "arithmetic", last_loss=3.0, last_score=0.0))
    buf.add(Item("easy", "a", "arithmetic", last_loss=0.01, last_score=1.0, times_seen=50))
    buf.add(Item("medium", "a", "arithmetic", last_loss=1.0, last_score=0.5))
    assert len(buf) == 2 and "hard" in buf.items and "easy" not in buf.items
    assert all(i.kind == "arithmetic" for i in buf.sample(2, ("arithmetic",)))
    assert buf.sample(2, ("code",)) == []
    assert len(buf.sample(2)) == 2  # with no filter, replay draws from every kind


def test_the_buffer_survives_a_round_trip():
    buf = PrioritizedReplayBuffer()
    buf.add(Item("q", "a", "code", last_loss=0.5, last_score=0.25, times_seen=3))
    other = PrioritizedReplayBuffer()
    other.load_json(json.loads(json.dumps(buf.to_json())))
    assert other.items["q"].times_seen == 3 and other.items["q"].last_score == 0.25


def test_the_curriculum_grows_from_arithmetic_to_code():
    assert "code" not in LEVELS[1] and "code" in LEVELS[5]
    for low, high in zip(sorted(LEVELS)[:-1], sorted(LEVELS)[1:]):
        assert set(LEVELS[low]) <= set(LEVELS[high])


def test_the_hub_respects_the_level_and_reads_a_hot_folder(tmp_path):
    hub = QAGeneratorHub(seed=3)
    items = hub.take(8, LEVELS[1])
    assert len(items) == 8 and all(i.kind in LEVELS[1] for i in items)

    drop = tmp_path / "extra.jsonl"
    drop.write_text(json.dumps({"q": "What is the capital of France?", "a": "Paris"}) + "\nnot json\n")
    hub2 = QAGeneratorHub(seed=3, data_path=drop)
    first = hub2.take(1, LEVELS[1])[0]
    assert first.a == "Paris" and first.kind == "custom"


def test_status_of_an_empty_workdir_says_nothing_happened(tmp_path):
    out = status(tmp_path)
    assert out["workdir"] == str(tmp_path) and "last" not in out


def test_a_question_the_model_keeps_failing_is_retired():
    buf = PrioritizedReplayBuffer(retire_after=3)
    hopeless = Item("13 * 47 = ?", "611", "arithmetic", failures=3)
    buf.add(hopeless)
    assert len(buf) == 0 and "13 * 47 = ?" in buf.retired
    buf.add(Item("13 * 47 = ?", "611", "arithmetic"))  # and it stays out
    assert len(buf) == 0
    buf2 = PrioritizedReplayBuffer(retire_after=0)   # 0 keeps the original behaviour
    buf2.add(Item("q", "a", "arithmetic", failures=99))
    assert len(buf2) == 1


def test_replay_crosses_the_curriculum_so_the_model_does_not_forget():
    """The bug that mattered: level-filtered replay cost the model everything else."""
    buf = PrioritizedReplayBuffer()
    buf.add(Item("code q", "a", "code", last_loss=0.2, last_score=0.9, times_seen=5))
    buf.add(Item("logic q", "a", "syllogism", last_loss=0.2, last_score=0.9, times_seen=5))
    buf.add(Item("sum q", "a", "arithmetic", last_loss=2.0, last_score=0.0))
    kinds = {i.kind for i in buf.sample(3)}
    assert kinds == {"code", "syllogism", "arithmetic"}
    assert {i.kind for i in buf.sample(3, LEVELS[1])} == {"arithmetic"}
