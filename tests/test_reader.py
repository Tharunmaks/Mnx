"""The Bees' text-mode browser (for machines with no Chromium, like a phone), and the "What happened" report."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import apps, brain as brain_mod, reader  # noqa: E402
from hive.core import Task  # noqa: E402

HOME = """<html><head><title>Wiki (fake)</title><script>var tracking = 1;</script></head><body>
<nav><a href="/about">About</a></nav>
<h1>The free encyclopedia</h1><p>Search millions of articles.</p>
<form action="/search" method="get"><input name="q" placeholder="Search Wiki"><button>Search</button></form>
<form action="/login" method="post"><input name="user" placeholder="User name"><input type="password" name="pw" placeholder="Password">
<input type="hidden" name="csrf" value="t0k"><button name="go" value="1">Log in</button></form>
</body></html>"""
RESULTS = """<html><head><title>Search: Chennai</title></head><body><h1>Results</h1>
<a href="/wiki/Chennai">Chennai</a><p>Capital of Tamil Nadu</p><a href="/wiki/Chennai_Central">Chennai Central</a></body></html>"""
ARTICLE = """<html><head><title>Chennai - Wiki</title></head><body><main><h1>Chennai</h1><p>Chennai is the capital of Tamil Nadu, on the
Coromandel Coast of the Bay of Bengal. Its population is about 11 million.</p>""" + "<p>More history.</p>" * 300 + "</main></body></html>"


class FakeWeb:
    def __init__(self):
        self.seen = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.seen.append((req.method, req.url.path, dict(req.url.params), req.content.decode(), req.headers.get("cookie")))
        p = req.url.path
        if p == "/":
            return httpx.Response(200, html=HOME)
        if p == "/search":
            return httpx.Response(200, html=RESULTS if req.url.params.get("q") == "Chennai" else "<title>Nothing</title>")
        if p == "/wiki/Chennai":
            return httpx.Response(200, html=ARTICLE)
        if p == "/login":
            return httpx.Response(302, headers={"location": "/me", "set-cookie": "sid=abc; Path=/"})
        if p == "/me":
            return httpx.Response(200, html="<title>Your account</title><p>Welcome back, asha</p>" if req.headers.get("cookie") == "sid=abc"
                                  else "<title>Login</title>")
        if p == "/app":
            return httpx.Response(200, html="<html><head><title>Shop</title></head><body><div id=root></div><script src=app.js></script></body></html>")
        return httpx.Response(404, html="<title>Not found</title>")


@pytest.fixture
def web(monkeypatch):
    fake = FakeWeb()
    monkeypatch.setattr(reader, "transport", httpx.MockTransport(fake.handler))
    monkeypatch.setattr("hive.browser._blocked_host", lambda host: host in ("127.0.0.1", "localhost"))
    return fake


def run(coro):
    return asyncio.run(coro)


def test_reading_searching_following_and_going_back(web):
    r = reader.ReaderSurface()
    run(r.open("wiki.example"))
    obs = run(r.observe())
    assert obs["title"] == "Wiki (fake)" and "The free encyclopedia" in obs["text"] and "tracking" not in obs["text"]
    labels = [e["label"] for e in obs["elements"]]
    assert labels[:4] == ["Search Wiki", "Search", "User name", "Password"] and "About" in labels and "t0k" not in json.dumps(obs)
    run(r.type(0, "Chennai", enter=True))
    obs = run(r.observe())
    assert obs["title"] == "Search: Chennai" and web.seen[-1][2] == {"q": "Chennai"}
    run(r.click(next(i for i, e in enumerate(obs["elements"]) if e["label"] == "Chennai")))
    obs = run(r.observe())
    assert obs["title"] == "Chennai - Wiki" and "population is about 11 million" in obs["text"] and "more characters below" in obs["text"]
    run(r.scroll())
    assert "population" not in run(r.observe())["text"]
    run(r.back())
    assert r.title == "Search: Chennai" and r.history == ["https://wiki.example"]


def test_forms_post_and_the_session_is_remembered(web):
    r = reader.ReaderSurface()
    run(r.open("https://wiki.example/"))
    run(r.observe())
    run(r.type(2, "asha"))
    run(r.type(3, "s3cret"))
    run(r.click(4))  # Log in
    method, path, _, body, _ = next(x for x in web.seen if x[1] == "/login")
    assert method == "POST" and "user=asha" in body and "pw=s3cret" in body and "csrf=t0k" in body and "go=1" in body
    assert r.title == "Your account" and "Welcome back" in r.text  # the cookie went along to the next page


def test_pages_that_need_javascript_say_so_and_errors_are_words(web):
    r = reader.ReaderSurface()
    run(r.open("https://wiki.example/app"))
    assert "needs the full browser" in run(r.observe())["text"]
    with pytest.raises(reader.ReaderError, match="404"):
        run(r.open("https://wiki.example/nowhere"))
    with pytest.raises(reader.ReaderError, match="private network"):
        run(r.open("http://127.0.0.1:8000/api/lab"))
    assert reader.explain_browser("Page.goto: net::ERR_NAME_NOT_RESOLVED at https://x") == "that address doesn't exist (no such website)"
    assert "no internet" in reader.explain_browser("net::ERR_INTERNET_DISCONNECTED")


def test_a_bee_uses_the_text_browser_and_tells_what_happened(web, tmp_path, monkeypatch):
    steps = [{"do": "type", "id": 0, "text": "Chennai", "enter": True}, "click:Chennai",
             {"do": "done", "summary": "Chennai is the capital of Tamil Nadu, with about 11 million people."}]
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})

    def model(req):
        nxt = steps.pop(0)
        if isinstance(nxt, str):  # click the element with this label on the screen it was shown
            screen = json.loads(req.content)["messages"][-1]["content"]
            row = next(ln for ln in screen.splitlines() if ln.endswith(json.dumps(nxt.split(":", 1)[1])))
            nxt = {"do": "click", "id": int(row[1:row.index("]")])}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(nxt)}}]})
    b.transport = httpx.MockTransport(model)
    monkeypatch.setattr(brain_mod, "brain", b)
    events = []

    async def emit(ev):
        events.append(ev)
    t = Task("t1", "c1", "open wiki.example and tell me about Chennai", emit)
    r = reader.ReaderSurface()

    async def go():
        await r.open("https://wiki.example/")
        await apps.run(t, r, {"app": None, "url": "https://wiki.example", "goal": t.text}, "Scout", None)
    run(go())
    answer = [e["text"] for e in events if e["type"] == "answer"][-1]
    told = next(e for e in events if e.get("badge") == "Report")
    assert answer == "Chennai is the capital of Tamil Nadu, with about 11 million people."
    assert [d[1] for d in told["details"][:5]] == ["Opened “Wiki (fake)” (wiki.example/)", "Typed Chennai into “Search Wiki”",
                                                   "Opened “Search: Chennai” (wiki.example/search?q=Chennai)", "Tapped “Chennai”",
                                                   "Opened “Chennai - Wiki” (wiki.example/wiki/Chennai)"]
    assert ["Now on", "Chennai - Wiki · wiki.example/wiki/Chennai"] in told["details"]
    assert not any(e["type"] == "screen" for e in events)  # text mode: no screenshots


def test_no_chromium_means_the_text_browser(web, monkeypatch, tmp_path):
    from hive import bees, cells as cells_mod
    from hive.browser import BrowserError
    monkeypatch.setattr(cells_mod, "CELLS", tmp_path / "cells")
    monkeypatch.setattr(cells_mod.cells, "cells", {})
    monkeypatch.setattr(cells_mod.cells, "mode", "local")
    r = bees.Registry(tmp_path / "hive.json", tmp_path / "keys.json")

    async def nothing(*a):
        return None
    r.on_created = r.on_deleted = r.on_computer = nothing
    monkeypatch.setattr(bees, "registry", r)
    run(r.create("Scout", "reads the web for me", owner="u_reader000001"))

    async def no_chromium(self, action, **a):
        raise BrowserError("The Bee's browser couldn't start: this machine has no Chromium for it.")
    monkeypatch.setattr("hive.browser.BrowserSession.do", no_chromium)
    monkeypatch.setattr(brain_mod, "is_connected", lambda: False)
    events = []

    async def emit(ev):
        events.append(ev)
    t = Task("t1", "c1", "open wiki.example.com", emit, user="u_reader000001")
    run(apps.handle(t, apps.parse(t.text)))
    texts = [e.get("text", "") for e in events]
    assert any("text mode" in x for x in texts)
    told = next(e for e in events if e.get("badge") == "Report")
    assert any(d[0] == "It says" and "The free encyclopedia" in d[1] for d in told["details"])
    assert "No model is running" in [e["text"] for e in events if e["type"] == "answer"][-1]
