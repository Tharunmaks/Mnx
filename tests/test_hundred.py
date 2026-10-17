"""A hundred things people ask, each through the whole Hive (hive/queen.py): every one reaches what can do it, gets an
answer, and nothing crashes. The slow actions (browsing, deploying, training) are recorded instead of run."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import apps, bots, brain as brain_mod, calls, cloud_chat, expert, modes, plans, queen, respond, sites  # noqa: E402
from hive.core import Task  # noqa: E402

M, W = "model", "web"  # answered by the model; answered by the model after a web search
HUNDRED = [
    # questions, writing, code, maths, plans: the model
    ("teach me python", M), ("explain recursion with an example", M), ("write a poem about the monsoon in Chennai", M),
    ("translate 'good morning, how are you' to Tamil", M), ("what is 17 times 23", M), ("give me a 7 day vegetarian diet plan", M),
    ("make me a todo app in react", M), ("write a python script to rename all jpg files in a folder", M),
    ("build a neural network in numpy", M), ("how do I order on swiggy", M), ("what is blinkit", M),
    ("explain how transformers work", M), ("write a cover letter for a junior web developer job", M),
    ("summarise the plot of the Mahabharata in 5 lines", M), ("help me plan a 3 day trip to Ooty", M),
    ("what's the difference between TCP and UDP", M), ("write a SQL query to find duplicate emails", M),
    ("create a brain teaser", M), ("teach me AI", M), ("give me some example text for a story", M),
    ("add data validation to this form", M), ("help me customize my vim config", M), ("how do I train a model?", M),
    ("make a website that loads in python flask", M), ("fix this error: TypeError: undefined is not a function", M),
    ("who are you", M), ("what can you do", M), ("tell me a joke in Tanglish", M),
    ("convert 5 feet 8 inches to centimetres", M), ("write a regex for Indian mobile numbers", M),
    ("explain compound interest with a table", M), ("write a bash one-liner to find the 10 biggest files", M),
    ("write a haiku about filter coffee", M), ("explain git rebase vs merge", M),
    # today's facts: the model, after a web search
    ("what's the weather in Chennai today", W), ("latest news about ISRO", W), ("who won the IPL final yesterday", W),
    ("gold rate in Chennai today", W), ("bitcoin price right now", W), ("search the web for the best budget phones", W),
    ("petrol price in Bangalore today", W), ("what's trending on twitter this week", W), ("nifty 50 today", W),
    ("release date of the next iPhone", W), ("who is the current chief minister of Tamil Nadu", W),
    # apps and websites: a Bee uses them
    ("add KitKat in blinkit", "apps"), ("order 2 milk from zepto", "apps"), ("search for running shoes on amazon", "apps"),
    ("order a pizza from dominos", "apps"), ("book a movie on bookmyshow", "apps"), ("search for bus tickets on redbus", "apps"),
    ("play arijit songs on spotify", "apps"), ("find cheap flights on skyscanner.co.in", "apps"), ("order a cake on Bakingo app", "apps"),
    ("pay my electricity bill on the BESCOM website", "apps"), ("book a hotel on airbnb", "apps"), ("check my order on flipkart.com", "apps"),
    ("send hi to amma on whatsapp", "apps"), ("buy a phone cover on meesho", "apps"), ("order biryani from zomato", "apps"),
    # websites the Bees build
    ("build a website for my bakery where people can order cakes", "sites"), ("create a landing page for my app with a waitlist", "sites"),
    ("make a portfolio site for my photos", "sites"), ("create a full stack web app called Tasky for team todos", "sites"),
    ("update my bakery site: add a contact page", "sites"), ("list my sites", "sites"), ("build a website from scratch", "sites"), ("make a blog for my travel stories", "sites"),
    # Expert Craft and Expert Stream
    ("expert craft plan 70B for 2 hours", "expert"), ("train a 70B model with expert craft called stories for 3 hours", "expert"),
    ("expert craft status stories", "expert"), ("ask my expert craft model stories: Once upon a time", "expert"),
    ("set up expert stream", "expert"),
    # Bees
    ("make a bee called Scout that watches prices", "bees"), ("make a bot called Maya that is a friendly Tamil tutor", "bees"),
    ("create a bee that translates Tamil", "bees"), ("list my bees", "cloud:bee"), ("what computer do my bees get?", "cloud:bee"),
    # plans
    ("upgrade to pro", "plans"), ("what plan am I on", "plans"), ("cancel my subscription", "plans"), ("get plus", "plans"),
    # calls
    ("call me", "calls"), ("call me in 10 minutes", "calls"), ("call me at 9am and tell me my plan for the day", "calls"),
    # the GPU
    ("run train.py on the gpu", "gpu"), ("gpu status", "gpu"),
    # training and models
    ("train Qwen 2.5 Coder", "cloud:train"), ("create a 1B model", "cloud:create"), ("create a 10M model for stories", "cloud:create"),
    ("add data to my model", "cloud:add_data"), ("how is training going", "cloud:status"), ("stop training", "cloud:stop"),
    ("run my model", "cloud:run"), ("show my usage", "cloud:usage"), ("I have 1500 INR, which GPU should I rent?", "cloud:gpu_budget"),
    ("run my stories model on the lpu", "cloud:on_lpu"), ("fine-tune Qwen/Qwen2.5-1.5B-Instruct", "cloud:train"),
    # Rush (all your Bees together)
    ("roll call", "rush"), ("everyone run df -h", "rush"),
]


def test_there_are_a_hundred():
    assert len(HUNDRED) == 100 and len({t for t, _ in HUNDRED}) == 100


@pytest.fixture
def hive(monkeypatch, tmp_path):
    """The real router, with the slow work recorded instead of run, and a model that echoes what it was asked."""
    seen = {}

    def record(route):
        async def handler(task, *a, **k):
            seen["route"] = route
            await task.answer(f"[{route}] done")
            return True
        return handler
    monkeypatch.setattr(apps, "handle", record("apps"))
    monkeypatch.setattr(sites, "handle", record("sites"))
    monkeypatch.setattr(expert, "handle", record("expert"))
    monkeypatch.setattr(bots, "create_flow", record("bees"))
    monkeypatch.setattr(plans, "handle", record("plans"))
    monkeypatch.setattr(queen, "gpu_chat", record("gpu"))

    async def call_chat(task):
        if calls.parse_call(task.text, "Asia/Kolkata"):
            return await record("calls")(task)
        return False
    monkeypatch.setattr(calls, "handle_chat", call_chat)

    async def cloud(task):
        i = cloud_chat.parse(task.text)
        if not i:
            return False
        return await record(f"cloud:{i.kind}")(task)
    monkeypatch.setattr(cloud_chat, "handle", cloud)

    async def rush(task):
        if not modes.rush_kind(task.text):
            return False
        return await record("rush")(task)
    monkeypatch.setattr(modes, "rush", rush)

    async def research(query, pages=3):
        seen["searched"] = query
        return [{"n": 1, "title": "A news site", "url": "https://news.example/today", "snippet": "fresh facts", "text": "Today: fresh facts."}]
    monkeypatch.setattr(respond.websearch, "research", research)

    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})

    def model(req):
        body = json.loads(req.content)
        web = "Web results for this question" in body["messages"][0]["content"]
        seen["route"] = W if web else M
        return httpx.Response(200, json={"choices": [{"message": {"content": f"A full answer to: {body['messages'][-1]['content']}"}}]})
    b.transport = httpx.MockTransport(model)
    monkeypatch.setattr(brain_mod, "brain", b)
    return seen


@pytest.mark.parametrize("text,route", HUNDRED, ids=[t[:40] for t, _ in HUNDRED])
def test_each_request_reaches_what_can_do_it(hive, text, route):
    events = []

    async def emit(ev):
        events.append(ev)
    t = Task("t1", "c1", text, emit, mode="rush" if route == "rush" else None, user="u_hundred00001")
    asyncio.run(queen.handle(t))
    assert hive.get("route") == route, (text, hive.get("route"))
    answers = [e for e in events if e["type"] == "answer"]
    assert len(answers) == 1 and answers[0]["text"], text
    assert not any(e["type"] in ("task_error",) for e in events)
    if route == W:
        assert "Sources: [1](https://news.example/today)" in answers[0]["text"] and hive["searched"]


def test_a_message_never_ends_without_an_answer(hive, monkeypatch):
    """A flow that stops at a card you cancel (it used to end silently) still answers."""
    async def quiet(task, *a, **k):
        return True
    monkeypatch.setattr(sites, "handle", quiet)
    events = []

    async def emit(ev):
        events.append(ev)
    asyncio.run(queen.handle(Task("t1", "c1", "build a website for my bakery", emit, user="u_hundred00001")))
    assert [e["text"] for e in events if e["type"] == "answer"] == ["Okay, I stopped there; nothing was started."]


def test_expert_craft_runs_one_command_at_a_time_per_bee(monkeypatch):
    order = []

    async def slow(task, intent, b):
        order.append(("start", intent["kind"]))
        await asyncio.sleep(0.05)
        order.append(("end", intent["kind"]))
        await task.answer("ok")
    monkeypatch.setattr(expert, "_craft", slow)
    monkeypatch.setattr(expert, "_pick_bee", lambda task: {"id": "webby", "name": "Webby"})

    async def both():
        async def emit(ev):
            pass
        await asyncio.gather(expert.handle(Task("a", "c", "x", emit), {"kind": "plan"}), expert.handle(Task("b", "c", "y", emit), {"kind": "train"}))
    asyncio.run(both())
    assert order == [("start", "plan"), ("end", "plan"), ("start", "train"), ("end", "train")]
