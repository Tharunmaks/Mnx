"""Bees make games on their own: write, deploy privately, play-test, fix; and wrong refusals are asked again."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import brain as brain_mod, games, respond, sites  # noqa: E402
from hive.core import Task  # noqa: E402


@pytest.mark.parametrize("text,styles", [
    ("Create me a 3d gta 6 game", ["GTA"]), ("Create a game like gta 6 not gta 6 in 3d", ["GTA"]),
    ("create an open world game in 3d", []), ("build a racing game like need for speed", ["Need for Speed"]),
    ("Use three.js for the game and make it yourself and run it", []),
    ("make a minecraft game in unity", ["Minecraft"]),
])
def test_game_requests(text, styles):
    got = games.parse(text)
    assert got and got["styles"] == styles
    assert got["unity"] is ("unity" in text)


@pytest.mark.parametrize("text", ["make a snake game in python", "explain game theory", "what's the best game of 2024",
                                  "give me video game ideas", "make a landing page for my bakery"])
def test_not_game_requests(text):
    assert games.parse(text) is None


def test_names_and_the_starter_game():
    assert games.name_for("Create me a 3d gta 6 game", ["GTA"]) == "open-world-game"
    assert games.name_for("create a zombie survival game", []) == "zombie-survival-game"
    files = games.template_files()
    assert set(files) >= {"index.html", "game.js", "style.css"}
    assert "/vendor/three/three.module.js" in files["index.html"] and "window.__game" in files["game.js"]


GAME_OUT = """=== FILE: index.html ===
<!doctype html><script type="importmap">{"imports":{"three":"/vendor/three/three.module.js"}}</script><script type="module" src="game.js"></script>
=== END ===
=== FILE: game.js ===
%s
window.__game = { ready: true };
=== END ==="""


@pytest.fixture
def hive(tmp_path, monkeypatch):
    from hive import bees, cells as cells_mod
    monkeypatch.setattr(cells_mod, "CELLS", tmp_path / "cells")
    monkeypatch.setattr(cells_mod.cells, "cells", {})
    monkeypatch.setattr(cells_mod.cells, "mode", "local")
    r = bees.Registry(tmp_path / "hive.json", tmp_path / "keys.json")

    async def nothing(*a):
        return None
    r.on_created = r.on_deleted = r.on_computer = nothing
    monkeypatch.setattr(bees, "registry", r)
    asyncio.run(r.create("Gamer", "makes my games", owner="u_gamer0000001"))
    deployed = []

    async def deploy(uid, name, files, cols, private=False):
        deployed.append((uid, name, dict(files), private))
        return {"name": name, "url": f"/d/dpl_test{len(deployed)}/"}
    monkeypatch.setattr(sites.chroma, "ensure", nothing)
    monkeypatch.setattr(sites.chroma, "deploy", deploy)
    return {"deployed": deployed, "cells": tmp_path / "cells"}


def _run(text, user="u_gamer0000001"):
    events = []

    async def emit(ev):
        events.append(ev)
    t = Task("t1", "c1", text, emit, user=user)
    asyncio.run(games.handle(t, games.parse(text)))
    return events


def test_a_bee_writes_plays_fixes_and_hands_over_a_private_game(hive, monkeypatch, tmp_path):
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})
    asked = []
    outs = [GAME_OUT % "const city = 1; brokn();", GAME_OUT % "const city = 1;"]
    b.transport = httpx.MockTransport(lambda r: asked.append(json.loads(r.content)) or
                                      httpx.Response(200, json={"choices": [{"message": {"content": outs.pop(0)}}]}))
    monkeypatch.setattr(brain_mod, "brain", b)
    plays = [{"ok": False, "ready": False, "errors": ["ReferenceError: brokn is not defined"], "shot": b""},
             {"ok": True, "ready": True, "errors": [], "shot": b"\xff\xd8jpeg"}]

    async def play(url):
        return plays.pop(0)
    monkeypatch.setattr(games, "play", play)
    events = _run("Create me a 3d gta 6 game")

    first, fix = asked
    assert "ORIGINAL game in that style" in first["messages"][0]["content"] and "copy nothing" in first["messages"][1]["content"]
    assert "brokn is not defined" in fix["messages"][1]["content"]
    assert all(d[3] is True for d in hive["deployed"]) and len(hive["deployed"]) == 2  # private every time
    assert "brokn" not in hive["deployed"][-1][2]["game.js"] and "gfx.js" in hive["deployed"][-1][2]  # the fix, with the kit kept
    assert "from scratch" in first["messages"][1]["content"] and "Neon Bay" not in json.dumps(first)   # no premade game sent or used
    card = next(e for e in events if e["type"] == "result")
    assert card["badge"] == "Private" and ["Play", "/d/dpl_test2/"] in card["details"]
    assert any(e["type"] == "screen" for e in events)
    answer = [e["text"] for e in events if e["type"] == "answer"][-1]
    assert "(/d/dpl_test2/)" in answer and "original game in the style of GTA" in answer and "/p/" not in answer
    assert (hive["cells"] / "gamer" / "work" / "games" / "open-world-game" / "game.js").exists()


def _model(monkeypatch, tmp_path, replies):
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})
    asked = []

    def reply(r):
        asked.append(json.loads(r.content))
        out = replies.pop(0)
        return out if isinstance(out, httpx.Response) else httpx.Response(200, json={"choices": [{"message": {"content": out}}]})
    b.transport = httpx.MockTransport(reply)
    monkeypatch.setattr(brain_mod, "brain", b)
    return asked


def test_no_model_means_no_game_never_a_premade_one(hive, monkeypatch):
    monkeypatch.setattr(brain_mod, "is_connected", lambda: False)
    events = _run("create a 3d high graphics train game")
    assert not hive["deployed"] and "no model is running" in events[-1]["text"]


def test_a_model_that_cant_fit_a_game_is_explained_and_nothing_premade_is_given(hive, monkeypatch, tmp_path):
    too_long = httpx.Response(400, json={"error": {"message": "the last message alone is longer than the context; start the server with a larger -c",
                                                   "type": "invalid_request_error"}})
    _model(monkeypatch, tmp_path, [too_long])
    events = _run("create a 3d high graphics train game")
    answer = [e["text"] for e in events if e["type"] == "answer"][-1]
    assert not hive["deployed"] and "context window" in answer and "Nothing was made" in answer


TRAIN = """=== FILE: index.html ===
<!doctype html><script type="importmap">{"imports":{"three":"/vendor/three/three.module.js"}}</script><script type="module" src="game.js"></script>
=== END ===
=== FILE: game.js ===
// CONTROLS: W speed up, S slow down, Space horn, tap the arrows on phones
import { createStage } from "./gfx.js";
const train = 1;
window.__game = { ready: true };
=== END ==="""


def test_a_train_game_is_written_for_trains_and_a_phone_just_isnt_play_tested(hive, monkeypatch, tmp_path):
    asked = _model(monkeypatch, tmp_path, [TRAIN])

    async def play(url):
        return {"ok": None, "errors": [], "shot": b"", "skipped": "this machine has no browser to play-test it"}
    monkeypatch.setattr(games, "play", play)
    events = _run("Create a 3d high graphics train game")
    system, user = asked[0]["messages"][0]["content"], asked[0]["messages"][1]["content"]
    assert "train game" in user and "never turn the request into a different game" in " ".join(system.split()) and len(asked) == 1   # no fix rounds
    deployed = hive["deployed"][-1][2]
    assert deployed["game.js"].startswith("// CONTROLS: W speed up") and "gfx.js" in deployed and "style.css" not in deployed
    card = next(e for e in events if e["type"] == "result")
    assert ["Controls", "W speed up, S slow down, Space horn, tap the arrows on phones"] in card["details"]
    assert ["Tested", "not play-tested (this machine has no browser to play-test it)"] in card["details"]
    assert "wrote it from scratch" in events[-1]["text"] and "cars" not in events[-1]["text"]


def test_still_broken_keeps_the_bees_own_game_never_a_premade_one(hive, monkeypatch, tmp_path):
    _model(monkeypatch, tmp_path, [TRAIN, TRAIN.replace("const train = 1;", "const train = 2;"), TRAIN.replace("const train = 1;", "const train = 3;")])

    async def play(url):
        return {"ok": False, "ready": False, "errors": ["TypeError: x is undefined"], "shot": b""}
    monkeypatch.setattr(games, "play", play)
    events = _run("make a train game")
    assert all("Neon Bay" not in json.dumps(d[2].get("index.html", "")) for d in hive["deployed"])
    assert "const train = 3;" in hive["deployed"][-1][2]["game.js"]
    assert "still has a problem" in events[-1]["text"]


def test_no_bee_yet_says_how(hive):
    events = _run("create an open world game in 3d", user="u_nobody000001")
    assert "Make one first" in events[-1]["text"] and not hive["deployed"]


def _sse(chunks):
    body = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def test_a_wrong_refusal_is_never_shown_and_is_asked_again(tmp_path, monkeypatch):
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})
    sent = []
    replies = [["As an AI, I'm unable to create ", "video games. "], ["Here is ", "a three.js city game:\n", "```js\n```"]]
    b.transport = httpx.MockTransport(lambda r: sent.append(json.loads(r.content)) or _sse(replies.pop(0)))
    monkeypatch.setattr(brain_mod, "brain", b)
    events = []

    async def emit(ev):
        events.append(ev)
    t = Task("t1", "c1", "write the code for a game like gta in three.js", emit)
    asyncio.run(respond.respond(t, brain_mod.system_prompt(), web=False))
    shown = json.dumps(events)
    assert "unable" not in shown and len(sent) == 2 and respond.RETRY in sent[1]["messages"][0]["content"]
    assert events[-1] == {**events[-1], "type": "answer"} and events[-1]["text"].startswith("Here is a three.js city game")
    assert any("Asking again" in (e.get("text") or e.get("title") or "") for e in events)


@pytest.mark.parametrize("text,refused", [
    ("I'm sorry, but I can't help with that.", True), ("As an AI language model, I am unable to create games.", True),
    ("I cannot create a copy of GTA 6.", True), ("Sure! Here's a game.", False), ("I can't wait to show you this game:", False),
])
def test_what_counts_as_a_refusal(text, refused):
    assert bool(respond.REFUSAL.search(text)) is refused


SMALL_MODEL = """Sure! Here is your train game.

```javascript
// CONTROLS: W go, S stop
import * as THREE from "three";
import { createStage } from "./gfx.js";
const stage = createStage(document.getElementById("game"), { time: 9 });
const train = new THREE.Mesh(new THREE.BoxGeometry(2, 1, 6), new THREE.MeshStandardMaterial({ color: 0xcc2222 }));
stage.scene.add(train);
```
Have fun!"""


@pytest.mark.parametrize("raw,ok", [
    (SMALL_MODEL, True), (TRAIN, True), ("I'm sorry, but I can't create games.", False), ("Use Unity and follow these steps: 1. …", False),
    ("```html\n<!doctype html><html><body><canvas id=game></canvas><script type=\"module\">" + "const a = 1;\n" * 30 + "</script></body></html>\n```", True),
])
def test_a_small_models_game_is_taken_in_any_shape(raw, ok):
    files = games.salvage(raw)
    assert bool(files) is ok
    if ok:
        assert "index.html" in files and ("game.js" not in files or "window.__game" in files["game.js"])


def test_a_small_model_that_refuses_is_asked_once_more_for_just_the_code(hive, monkeypatch, tmp_path):
    asked = _model(monkeypatch, tmp_path, ["I'm sorry, I can't make a game. You could try Unity.", SMALL_MODEL])

    async def play(url):
        return {"ok": None, "errors": [], "shot": b"", "skipped": "this machine has no browser to play-test it"}
    monkeypatch.setattr(games, "play", play)
    events = _run("make a train game")
    assert len(asked) == 2 and "one ```js code block" in asked[1]["messages"][-1]["content"]
    game = hive["deployed"][-1][2]
    assert game["index.html"] == games.BASE_HTML and game["game.js"].startswith("// CONTROLS: W go") and "window.__game" in game["game.js"]
    assert "wrote it from scratch" in events[-1]["text"]


def test_a_slow_model_that_times_out_is_explained_never_a_blank_error(hive, monkeypatch, tmp_path):
    def slow(r):
        raise httpx.ReadTimeout("")
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})
    b.transport = httpx.MockTransport(slow)
    monkeypatch.setattr(brain_mod, "brain", b)
    answer = [e["text"] for e in _run("make a train game") if e["type"] == "answer"][-1]
    assert "couldn't make the game: ." not in answer and "too long" in answer and not hive["deployed"]


def test_the_game_streams_so_a_slow_phone_model_keeps_going(hive, monkeypatch, tmp_path):
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})
    sent = []
    b.transport = httpx.MockTransport(lambda r: sent.append(json.loads(r.content)) or _sse([TRAIN[:200], TRAIN[200:]]))
    monkeypatch.setattr(brain_mod, "brain", b)

    async def play(url):
        return {"ok": None, "errors": [], "shot": b"", "skipped": "no browser"}
    monkeypatch.setattr(games, "play", play)
    _run("make a train game")
    assert sent[0]["stream"] is True and "const train = 1;" in hive["deployed"][-1][2]["game.js"]
