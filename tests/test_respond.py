"""Answers: streamed as they're written (in the model's own name), web research for today's facts, and its results read."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import brain as brain_mod, respond, websearch  # noqa: E402
from hive.core import Task  # noqa: E402


@pytest.mark.parametrize("text,web", [
    ("what's the weather in Chennai today", True), ("who is the current chief minister of Tamil Nadu", True),
    ("latest python version", True), ("search the web for the best budget phones", True), ("gold rate in Chennai today", True),
    ("explain electric current", False), ("what is the current in a 10 ohm resistor at 5 V", False),
    ("write a story about a news reporter", False), ("write a poem about today", False), ("teach me python", False),
    ("translate 'what's the weather' to Hindi", False),
])
def test_when_to_search(text, web):
    assert respond.needs_web(text) is web


DDG = """<div class="result"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fweather.example%2Fchennai&amp;rut=x">Chennai <b>weather</b> today</a>
<a class="result__snippet" href="#">Sunny, 34 &deg;C, humid.</a></div>
<div class="result"><a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad=1">An ad</a></div>
<div class="result"><a rel="nofollow" class="result__a" href="https://imd.example/forecast">IMD forecast</a><a class="result__snippet">Rain later.</a></div>"""


def test_reading_search_results_and_pages():
    rs = websearch.parse_results(DDG)
    assert [r["url"] for r in rs] == ["https://weather.example/chennai", "https://imd.example/forecast"]  # no ads
    assert rs[0]["title"] == "Chennai weather today" and rs[0]["snippet"] == "Sunny, 34 °C, humid."
    page = "<html><script>var x=1</script><nav>menu</nav><main><h1>Chennai</h1><p>Sunny &amp; 34 °C today." + " More." * 80 + "</p></main></html>"
    assert websearch.page_text(page).startswith("Chennai Sunny & 34 °C today.") and "var x" not in websearch.page_text(page)


def test_private_addresses_are_never_fetched(monkeypatch):
    hits = []
    monkeypatch.setattr(websearch, "transport", httpx.MockTransport(lambda r: hits.append(r) or httpx.Response(200, text="secret")))
    assert asyncio.run(websearch.read("http://127.0.0.1:8000/api/lab")) == "" and not hits


def _stream(chunks):
    body = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def test_answers_stream_whole_sentences_in_the_models_own_name(tmp_path, monkeypatch):
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})
    sent = []
    b.transport = httpx.MockTransport(lambda r: sent.append(json.loads(r.content)) or
                                      _stream(["I am Qw", "en, created by Alibaba Cloud. ", "Here is ", "the plan:\n", "1. Rest"]))
    monkeypatch.setattr(brain_mod, "brain", b)
    events = []

    async def emit(ev):
        events.append(ev)
    t = Task("t1", "c1", "who are you", emit)
    asyncio.run(respond.respond(t, brain_mod.system_prompt()))
    deltas = [e["text"] for e in events if e["type"] == "answer_delta"]
    assert sent[0]["stream"] is True and "Start with the answer itself" in sent[0]["messages"][0]["content"]
    assert deltas[0] == "I am Mnx, created by Mnx." and "Qwen" not in "".join(deltas) and "Alibaba" not in "".join(deltas)
    assert "".join(deltas).endswith("1. Rest") and events[-1]["type"] == "answer"
    assert events[-1]["text"] == "I am Mnx, created by Mnx. Here is the plan:\n1. Rest"
    assert t.tokens_out < 40  # the pieces aren't counted twice


def test_a_question_about_today_is_researched_and_cited(tmp_path, monkeypatch):
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})
    asked = []

    def model(req):
        asked.append(json.loads(req.content))
        return _stream(["It's sunny and 34 °C in Chennai today [1]."])
    b.transport = httpx.MockTransport(model)
    monkeypatch.setattr(brain_mod, "brain", b)

    async def research(q, pages=3):
        return [{"n": 1, "title": "Chennai weather", "url": "https://weather.example/chennai", "snippet": "Sunny", "text": "Sunny, 34 °C"}]
    monkeypatch.setattr(websearch, "research", research)
    events = []

    async def emit(ev):
        events.append(ev)
    t = Task("t1", "c1", "what's the weather in Chennai today", emit)
    asyncio.run(respond.respond(t, brain_mod.system_prompt()))
    assert "Sunny, 34 °C" in asked[0]["messages"][0]["content"]
    assert any(e["type"] == "searching" for e in events)
    assert events[-1]["text"] == "It's sunny and 34 °C in Chennai today [1].\n\nSources: [1](https://weather.example/chennai)"


def test_when_the_search_fails_it_still_answers(tmp_path, monkeypatch):
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})
    b.transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "I can't check today's weather."}}]}))
    monkeypatch.setattr(brain_mod, "brain", b)
    monkeypatch.setattr(websearch, "transport", httpx.MockTransport(lambda r: httpx.Response(503)))
    events = []

    async def emit(ev):
        events.append(ev)
    asyncio.run(respond.respond(Task("t1", "c1", "weather in Chennai today", emit), brain_mod.system_prompt()))
    assert events[-1]["text"] == "I can't check today's weather." and any(e.get("status") == "error" for e in events)
