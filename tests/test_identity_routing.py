"""Ordinary requests reach the model (not the make-a-model flows), and the model answers as the Mnx model it is."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import bots, brain, cloud_chat  # noqa: E402


@pytest.mark.parametrize("text", [
    "teach me python", "make an AI chatbot in python", "build a neural network in numpy",
    "give me some example text for a story", "add data validation to this form", "create a brain teaser",
    "train for a marathon plan", "help me customize my vim config", "build a website from scratch",
    "make me a todo app in react", "create a landing page", "write a snake game", "how do I train a model?",
    "explain how transformers work", "make a website that uses AI", "create a python script to rename files",
    "give me ideas for a story", "add a dark mode to my site", "teach me AI", "what model are you",
    "add a character to my story", "create a character for my novel", "build a chatbot for my website",
    "make a telegram bot that sends reminders", "start a bot conversation",
])
def test_ordinary_requests_go_to_the_model(text):
    assert cloud_chat.parse(text) is None
    assert bots.parse(text) is None


@pytest.mark.parametrize("text,kind", [
    ("train Qwen 2.5 Coder", "train"), ("fine-tune Qwen/Qwen2.5-1.5B-Instruct", "train"), ("train my model", "train"),
    ("train the 100B blueprint", "train"), ("fine-tune my model on tharun/data", "train"),
    ("create a 10M model for stories", "create"), ("create a 1B model", "create"),
    ("create a 101.3 billion parameter model AI", "create"), ("create a new ai model for coding", "create"),
    ("add data to my model", "add_data"), ("feed these pages into my stories model", "add_data"),
    ("continue training", "add_data"), ("run my model", "run"),
])
def test_model_requests_still_route(text, kind):
    i = cloud_chat.parse(text)
    assert i is not None and i.kind == kind


def test_making_a_bot_still_works():
    assert bots.parse("make a bot called Maya that is a friendly Tamil tutor")["name"] == "Maya"


@pytest.mark.parametrize("reply,want", [
    ("I am Qwen, a large language model created by Alibaba Cloud. I can help you.",
     "I am Mnx 1.5 Max Coder, a large language model created by Mnx. I can help you."),
    ("I'm Qwen2.5-Coder, developed by Alibaba Cloud's Tongyi Lab.\n\nAsk me anything.",
     "I'm Mnx 1.5 Max Coder, developed by Mnx.\n\nAsk me anything."),
    ("我是通义千问，由阿里云开发。", "我是Mnx 1.5 Max Coder，由Mnx开发。"),
    ("Qwen is a model family from Alibaba.", "Qwen is a model family from Alibaba."),  # about Qwen, not as Qwen
    ("Sure:\n```py\nprint(1)\n```", "Sure:\n```py\nprint(1)\n```"),
])
def test_the_model_uses_its_own_name(reply, want):
    assert brain.own_name(reply, "Mnx 1.5 Max Coder") == want


def test_every_conversation_carries_the_identity_and_history(monkeypatch):
    seen = {}

    def answer(request: httpx.Request) -> httpx.Response:
        import json
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "I am Qwen, created by Alibaba Cloud."}}]})

    monkeypatch.setattr(brain.brain, "transport", httpx.MockTransport(answer))
    from hive import models
    token = models.selected.set("mnx-1.5-max-coder")
    try:
        monkeypatch.setattr(models.models, "using", _fake_using)
        reply = asyncio.run(brain.ask("and now in rust", history=[{"role": "user", "text": "write fizzbuzz"},
                                                                  {"role": "bot", "text": "print('fizz')"}]))
    finally:
        models.selected.reset(token)
    msgs = seen["body"]["messages"]
    assert msgs[0]["role"] == "system" and "Mnx 1.5 Max Coder" in msgs[0]["content"] and "Alibaba" in msgs[0]["content"]
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"] and msgs[-1]["content"] == "and now in rust"
    assert reply == "I am Mnx 1.5 Max Coder, created by Mnx."


class _fake_using:
    def __init__(self, _model):
        pass

    async def __aenter__(self):
        return "http://model.test"

    async def __aexit__(self, *exc):
        return False
