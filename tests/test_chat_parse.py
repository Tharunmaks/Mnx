"""What the chat understands without a model: the fixed patterns that route a message to a Bee."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import cloud_chat  # noqa: E402


@pytest.mark.parametrize("text,kind,query,purpose,chips,steps,prompt,engine", [
    ("run my stories model on the lpu", "on_lpu", "stories", "run", 1, None, "", ""),
    ("run stories on 8 lpu chips", "on_lpu", "stories", "run", 8, None, "", ""),
    ('run the 1.28M model on 4 lpu chips with prompt "The cat"', "on_lpu", "1.28M", "run", 4, None, "The cat", ""),
    ("simulate my stories model on a virtual chip", "on_lpu", "stories", "run", 1, None, "", ""),
    ("run stories on the groq chip", "on_lpu", "stories", "run", 1, None, "", ""),
    ("train my stories model on the lpu for 8 steps", "on_lpu", "stories", "train", 1, 8, "", ""),
    ("train stories on 2 lpu chips for 20 steps", "on_lpu", "stories", "train", 2, 20, "", ""),
    ("run my stories model on my phone", "on_phone", "stories", "run", None, None, "", ""),
    ("run my stories model on my phone on the lpu", "on_phone", "stories", "run", None, None, "", "lpu"),
    ("train my stories model on my phone for 5 steps", "on_phone", "stories", "train", None, None, "", ""),
])
def test_phone_and_lpu_intents(text, kind, query, purpose, chips, steps, prompt, engine):
    i = cloud_chat.parse(text)
    assert i is not None and i.kind == kind
    assert i.model_query == query and i.purpose == purpose and i.dataset == prompt and i.engine == engine
    if kind == "on_lpu":
        assert int(i.size_b) == chips and i.steps == steps
    if kind == "on_phone" and purpose == "train":
        assert i.size_b == 5  # the phone flow carries its step count here


@pytest.mark.parametrize("text,kind", [
    ("run my model", "run"),
    ("stop training", "stop"),
    ("stop renting the gpu", "rent_stop"),
    ("create a 10M model for stories", "create"),
    ("how is training going", "status"),
])
def test_other_intents_still_route(text, kind):
    i = cloud_chat.parse(text)
    assert i is not None and i.kind == kind


def test_small_talk_is_not_a_cloud_request():
    assert cloud_chat.parse("hello, how are you?") is None
