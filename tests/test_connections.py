"""Connected apps: Blinkit's delivery place and sign-in, what's missing said plainly, and the Apps page.

The browser tests run the real flow in real Chromium against a stand-in Blinkit that is built like the real thing (a page of plain
clickable divs: no buttons, no links), and are skipped on a machine with no Chromium.
"""

from __future__ import annotations

import asyncio
import json
import socketserver
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway import auth  # noqa: E402
from hive import apps, brain as brain_mod, connections  # noqa: E402
from hive.browser import BrowserSession, find_chromium, stop_playwright  # noqa: E402
from hive.core import Task  # noqa: E402

PHONE, OTP = "9876543210", "123456"

MOCK = """<!doctype html><meta charset=utf-8><title>blinkit</title>
<style>.btn,.row{cursor:pointer;display:inline-block;padding:8px 12px;margin:4px;background:#0c831f;color:#fff;border-radius:6px}.row{display:block;background:#eee;color:#000}
.modal{border:2px solid #333;padding:12px;margin:10px;background:#fff}input{padding:6px;margin:4px}</style>
<div id=app></div>
<script>
const PLACES = ["Anna Nagar, Chennai 600040", "Adyar, Chennai 600020", "T Nagar, Chennai 600017", "Koramangala, Bengaluru 560034"];
const S = { place: localStorage.place || "", logged: localStorage.logged === "1", step: "", cart: +(localStorage.cart || 0), typed: "", q: new URLSearchParams(location.search).get("q") || "" };
function draw() {
  let h = "";
  if (!S.place) {
    h += `<div class=modal><h2>Welcome to blinkit</h2><div class=btn id=detect>Detect my location</div>
      <input id=loc placeholder="search delivery location" value="${S.typed}">`;
    if (S.typed) for (const p of PLACES.filter((p) => p.toLowerCase().includes(S.typed.toLowerCase()))) h += `<div class=row data-place="${p}">${p}</div>`;
    h += `</div>`;
  }
  h += `<header><b>blinkit</b> ${S.place ? `<span>Delivery in 8 minutes ${S.place}</span>` : ""}
    <input id=q placeholder="Search 'milk'" value="${S.q}"> ${S.logged ? "<div class=btn id=account>Account</div>" : "<div class=btn id=login>Login</div>"}
    <div class=btn id=cart>Cart ${S.cart}</div></header>`;
  if (S.step === "phone") h += `<div class=modal><h2>Log in</h2><input id=phone placeholder="Enter mobile number"><div class=btn id=cont>Continue</div></div>`;
  if (S.step === "otp") h += `<div class=modal><h2>Enter the OTP we sent</h2><input id=otp placeholder="OTP"><div class=btn id=verify>Verify</div></div>`;
  if (S.place) h += `<div class=grid><div>Kit Kat 4 pack ₹45 <div class=btn id=add>ADD</div></div><div>Dairy Milk ₹60 <div class=btn>ADD</div></div></div>`;
  const keep = document.activeElement && document.activeElement.id;
  app.innerHTML = h;
  if (keep && document.getElementById(keep)) { const el = document.getElementById(keep); el.focus(); el.setSelectionRange && el.setSelectionRange(el.value.length, el.value.length); }
}
document.addEventListener("input", (e) => { if (e.target.id === "loc") { S.typed = e.target.value; draw(); } });
document.addEventListener("click", (e) => {
  const t = e.target.closest("[id],[data-place]"); if (!t) return;
  if (t.dataset && t.dataset.place) { S.place = localStorage.place = t.dataset.place; }
  else if (t.id === "login") S.step = "phone";
  else if (t.id === "cont") { if (document.getElementById("phone").value.replace(/\\D/g, "").length === 10) S.step = "otp"; }
  else if (t.id === "verify") { if (document.getElementById("otp").value === "%(otp)s") { S.logged = true; localStorage.logged = "1"; S.step = ""; } }
  else if (t.id === "add") { S.cart++; localStorage.cart = S.cart; }
  draw();
});
draw();
</script>""" % {"otp": OTP}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = MOCK if not self.path.startswith("/blocked") else "<title>Access Denied</title><h1>Access Denied</h1><p>You don't have permission to access this server.</p>"
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def shop(tmp_path, monkeypatch):
    monkeypatch.setattr(connections, "FILE", tmp_path / "connections.json")
    srv = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setitem(apps.APPS["blinkit"], "web", f"http://127.0.0.1:{port}/")
    monkeypatch.setitem(apps.APPS["blinkit"], "search", f"http://127.0.0.1:{port}/?q={{q}}")
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


needs_chromium = pytest.mark.skipif(not find_chromium(), reason="no Chromium on this machine")


def run(coro):
    return asyncio.run(coro)


async def close(session):
    """Each test has its own event loop, so Playwright (cached for the whole server) must be stopped in this one."""
    await session.shutdown()
    await stop_playwright()


def _task(answers, user="u_shopper00001", text="connect blinkit"):
    events, box = [], {}

    async def emit(ev):
        events.append(ev)
        if ev["type"] in ("clarify", "confirm"):
            action, values = answers.pop(0)
            asyncio.get_running_loop().call_soon(lambda: box["t"].resolve(ev["id"], action, values))
    t = Task("t_conn", "c1", text, emit, user=user)
    box["t"] = t
    return t, events


# ---------------------------------------------------------------- reading a page
def page(text="", els=(), title=""):
    return {"place": "", "title": title, "text": text, "elements": [{"label": l, "kind": k} for l, k in els]}


@pytest.mark.parametrize("obs,state", [
    (page("Access Denied. You don't have permission to access this server.", [("Home", "a")], "Access Denied"), "blocked"),
    (page("", []), "empty"),
    (page("Welcome. Select delivery location", [("Detect my location", "div"), ("search delivery location", "input:text"), ("Search 'milk'", "input:text")]), "location"),
    (page("Delivery in 8 minutes", [("Search 'milk'", "input:text"), ("Login", "div"), ("Cart", "div")]), "signed_out"),
    (page("Delivery in 8 minutes", [("Search 'milk'", "input:text"), ("Account", "div"), ("Cart", "div")]), "ready"),
    (page("Hello, pick a place to deliver. Select your location", [("Home", "a")]), "location"),
])
def test_what_a_page_is_waiting_for(obs, state):
    assert connections.classify(obs) == state


@pytest.mark.parametrize("text,want", [
    ("connect blinkit", ("blinkit", "connect")), ("set my delivery location on blinkit", ("blinkit", "connect")),
    ("sign in to zepto", ("zepto", "connect")), ("is blinkit ready?", ("blinkit", "status")), ("blinkit status", ("blinkit", "status")),
    ("disconnect blinkit", ("blinkit", "disconnect")), ("is blinkit ready to order", ("blinkit", "status")),
    ("add KitKat in blinkit", None), ("order 2 milk from zepto", None), ("login to zomato and order biryani", None),
    ("what is blinkit", None), ("connect to the internet", None), ("search for shoes on amazon", None),
])
def test_what_is_a_connect_request(text, want):
    got = connections.parse(text)
    assert (got and (got["app"], got["action"])) == want


def test_only_the_fact_of_a_connection_is_remembered(tmp_path, monkeypatch):
    monkeypatch.setattr(connections, "FILE", tmp_path / "c.json")
    connections.mark("u_a", "blinkit", bee="shopper", location="Anna Nagar", signed_in=True)
    connections.mark("u_a", "blinkit", signed_in=False)
    assert connections.get("u_a")["blinkit"]["location"] == "Anna Nagar" and connections.get("u_a")["blinkit"]["signed_in"] is False
    assert connections.get("u_b") == {} and oct((tmp_path / "c.json").stat().st_mode)[-3:] == "600"
    assert connections.forget("u_a", "blinkit") and not connections.forget("u_a", "blinkit")


def test_the_apps_page_says_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(connections, "FILE", tmp_path / "c.json")
    monkeypatch.setattr(brain_mod, "is_connected", lambda: False)
    monkeypatch.setattr("hive.browser.find_chromium", lambda: None)
    st = connections.status("u_a", [])
    checks = {c["id"]: c for c in st["checks"]}
    assert not any(c["ok"] for c in checks.values()) and "playwright install" in checks["browser"]["detail"] and "Make a Bee" in checks["bee"]["detail"]
    assert st["apps"][0]["key"] == "blinkit" and not st["apps"][0]["connected"]
    connections.mark("u_a", "blinkit", location="Anna Nagar")
    monkeypatch.setattr(brain_mod, "is_connected", lambda: True)
    monkeypatch.setattr("hive.browser.find_chromium", lambda: "/usr/bin/chromium")
    st = connections.status("u_a", [{"name": "Shopper"}])
    assert all(c["ok"] for c in st["checks"]) and st["apps"][0]["connected"] and "Anna Nagar" in st["checks"][3]["detail"]


# ---------------------------------------------------------------- the real flow, in real Chromium
@needs_chromium
def test_a_bee_sets_the_place_and_signs_in_with_secrets_never_kept(shop, tmp_path):
    async def go():
        s = BrowserSession(allow={shop.split("//")[1]})
        surface = apps.WebSurface(s)
        try:
            await surface.open(shop)
            t, events = _task([("go", {"place": "anna nagar"}), ("go", {}), ("go", {"phone": "98765 43210"}), ("go", {"otp": OTP})])
            ok = await connections.ensure_ready(t, surface, "blinkit", "Shopper", None, "shopper")
            obs = await surface.observe()
            return ok, events, obs
        finally:
            await close(s)
    ok, events, obs = run(go())
    assert ok
    rec = connections.get("u_shopper00001")["blinkit"]
    assert rec["location"] == "anna nagar" and rec["signed_in"] is True
    assert any(e["label"] == "Account" for e in obs["elements"]) and "Anna Nagar" in obs["text"]      # the site really got it
    blob = json.dumps(events) + (tmp_path / "connections.json").read_text()
    assert PHONE not in blob and "98765 43210" not in blob and OTP not in blob                         # never in the chat or on disk
    cards = [e for e in events if e["type"] in ("clarify", "confirm")]
    assert [c["title"] for c in cards] == ["Where should Blinkit deliver?", "Sign in to Blinkit?", "Your Blinkit number", "Blinkit OTP"]
    assert cards[3]["fields"][0]["input"] == "password" and cards[3]["fields"][0]["secret"]            # the OTP box is a secret box
    assert [e["step"] if "step" in e else None for e in events if e["type"] == "screen"] or any(e["type"] == "screen" for e in events)


@needs_chromium
def test_a_place_that_isnt_found_is_asked_again_and_signing_in_can_wait(shop):
    async def go():
        s = BrowserSession(allow={shop.split("//")[1]})
        surface = apps.WebSurface(s)
        try:
            await surface.open(shop)
            t, events = _task([("go", {"place": "zzzqqq"}), ("go", {"place": "Adyar"}), ("skip", {})])
            return await connections.ensure_ready(t, surface, "blinkit", "Shopper", None, "shopper"), events
        finally:
            await close(s)
    ok, events = run(go())
    cards = [e for e in events if e["type"] in ("clarify", "confirm")]
    assert ok and len(cards) == 3 and "wasn't found" in cards[1]["text"]
    rec = connections.get("u_shopper00001")["blinkit"]
    assert "Adyar" in rec["location"] and not rec["signed_in"] and rec["skip_login"]


@needs_chromium
def test_a_wrong_otp_leaves_you_signed_out_but_able_to_shop(shop):
    async def go():
        s = BrowserSession(allow={shop.split("//")[1]})
        surface = apps.WebSurface(s)
        try:
            await surface.open(shop)
            t, events = _task([("go", {"place": "Anna"}), ("go", {}), ("go", {"phone": PHONE}), ("go", {"otp": "000000"})])
            return await connections.ensure_ready(t, surface, "blinkit", "Shopper", None, "shopper"), events
        finally:
            await close(s)
    ok, events = run(go())
    assert ok and not connections.get("u_shopper00001")["blinkit"]["signed_in"]
    assert any(e["type"] == "error" and "sign-in didn't finish" in e.get("text", "") for e in events)


@needs_chromium
def test_a_blocked_site_is_explained_not_looped_on(shop):
    async def go():
        s = BrowserSession(allow={shop.split("//")[1]})
        surface = apps.WebSurface(s)
        try:
            await surface.open(shop + "/blocked")
            t, events = _task([])
            return await connections.ensure_ready(t, surface, "blinkit", "Shopper", "browser.html?bee=shopper", "shopper"), events
        finally:
            await close(s)
    ok, events = run(go())
    answer = [e["text"] for e in events if e["type"] == "answer"][-1]
    assert not ok and "refused this browser" in answer and "Watch live" in answer and any(e["type"] == "screen" for e in events)


def test_text_mode_cannot_run_blinkit_and_says_so():
    from hive.reader import ReaderSurface
    t, events = _task([])
    ok = run(connections.ensure_ready(t, ReaderSurface(), "blinkit", "Shopper", None, "shopper"))
    answer = [e["text"] for e in events if e["type"] == "answer"][-1]
    assert not ok and "full browser (Chromium)" in answer and "playwright install" in answer


@needs_chromium
def test_add_kitkat_sets_things_up_first_then_shops(shop, tmp_path, monkeypatch):
    """The whole chat request: the Bee opens Blinkit, gets the place from you, then carries on (here no model is running, so it stops there)."""
    from hive import bees, cells as cells_mod
    monkeypatch.setattr(cells_mod, "CELLS", tmp_path / "cells")
    monkeypatch.setattr(cells_mod.cells, "cells", {})
    monkeypatch.setattr(cells_mod.cells, "mode", "local")
    r = bees.Registry(tmp_path / "hive.json", tmp_path / "keys.json")

    async def nothing(*a):
        return None
    r.on_created = r.on_deleted = r.on_computer = nothing
    monkeypatch.setattr(bees, "registry", r)
    monkeypatch.setattr(brain_mod, "is_connected", lambda: False)

    async def go():
        await r.create("Shopper", "orders my groceries", owner="u_shopper00001")
        s = BrowserSession(allow={shop.split("//")[1]})

        async def open_web(task, b, url, where, name, step=None):
            surface = apps.WebSurface(s)
            await surface.open(url)
            return surface, "browser.html?bee=shopper", step
        monkeypatch.setattr(apps, "open_web", open_web)
        try:
            t, events = _task([("go", {"place": "T Nagar"}), ("skip", {})], text="add KitKat in blinkit")
            await apps.handle(t, apps.parse("add KitKat in blinkit"))
            return events
        finally:
            await close(s)
    events = run(go())
    assert [e["title"] for e in events if e["type"] in ("clarify", "confirm")] == ["Where should Blinkit deliver?", "Sign in to Blinkit?"]
    assert "T Nagar" in connections.get("u_shopper00001")["blinkit"]["location"]
    answer = [e["text"] for e in events if e["type"] == "answer"][-1]
    assert "No model is running" in answer and "KitKat" in answer


@needs_chromium
def test_the_browser_sees_buttons_made_of_divs_in_page_order(shop):
    async def go():
        s = BrowserSession(allow={shop.split("//")[1]})
        await s.do("goto", url=shop)
        try:
            return (await s.do("elements"))["elements"]
        finally:
            await close(s)
    els = run(go())
    labels = [e["label"] for e in els]
    assert "Detect my location" in labels and "Login" in labels and "Cart 0" in labels                        # divs with a hand cursor
    assert labels.index("Detect my location") < labels.index("search delivery location") < labels.index("Login")  # in page order
    box = next(e for e in els if e["label"].startswith("search delivery location"))
    assert box["tag"] == "input" and box["label"] == "search delivery location"                                 # a box keeps its own name


# ---------------------------------------------------------------- the gateway
@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import gateway.api as gw
    monkeypatch.setattr(auth, "USER_TOKENS_FILE", tmp_path / "t.txt")
    (tmp_path / "t.txt").write_text("user-token-a\nuser-token-b\n")
    monkeypatch.setattr(connections, "FILE", tmp_path / "c.json")
    return TestClient(gw.app)


def test_each_person_sees_only_their_own_connections(client):
    a, b = {"Authorization": "Bearer user-token-a"}, {"Authorization": "Bearer user-token-b"}
    connections.mark(auth.user_of("user-token-a"), "blinkit", location="Anna Nagar", signed_in=True)
    st = client.get("/api/apps/status", headers=a).json()
    assert st["apps"][0]["connected"] and st["apps"][0]["location"] == "Anna Nagar" and {c["id"] for c in st["checks"]} == {"bee", "browser", "model", "place"}
    assert not client.get("/api/apps/status", headers=b).json()["apps"][0]["connected"]
    assert client.delete("/api/apps/blinkit", headers=b).json() == {"forgot": False}
    assert client.delete("/api/apps/blinkit", headers=a).json() == {"forgot": True}
    assert client.get("/api/apps/status", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/apps.html").status_code == 200
