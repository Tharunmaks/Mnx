"""Bees make 3D models (a real .glb), Mnx Studio edits them, games use them, the Sandbox lists it all; and "make X"
answered with how to make X is asked again."""

from __future__ import annotations

import asyncio
import json
import math
import struct
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway import auth  # noqa: E402
from hive import brain as brain_mod, games, models3d, respond, sandbox, sites  # noqa: E402
from hive.core import Task  # noqa: E402


def read_glb(data: bytes) -> tuple[dict, bytes]:
    magic, version, total = struct.unpack_from("<III", data)
    assert magic == 0x46546C67 and version == 2 and total == len(data) and total % 4 == 0
    jlen, jtype = struct.unpack_from("<II", data, 12)
    assert jtype == 0x4E4F534A and jlen % 4 == 0
    gltf = json.loads(data[20:20 + jlen])
    blen, btype = struct.unpack_from("<II", data, 20 + jlen)
    assert btype == 0x004E4942
    return gltf, data[28 + jlen:28 + jlen + blen]


def test_a_design_becomes_a_valid_glb():
    spec = models3d.clean(models3d._car())
    gltf, blob = read_glb(models3d.to_glb(spec))
    assert gltf["asset"]["version"] == "2.0" and gltf["buffers"][0]["byteLength"] == len(blob)
    parts = [n for n in gltf["nodes"] if "mesh" in n]
    assert len(parts) == len(spec["parts"]) and gltf["nodes"][gltf["scenes"][0]["nodes"][0]]["children"] == list(range(len(parts)))
    for acc in gltf["accessors"]:  # every accessor fits in its view, and indices point at real vertices
        view = gltf["bufferViews"][acc["bufferView"]]
        width = {"VEC3": 12, "SCALAR": 2}[acc["type"]]
        assert acc["count"] * width == view["byteLength"] and view["byteOffset"] + view["byteLength"] <= len(blob)
    for mesh in gltf["meshes"]:
        prim = mesh["primitives"][0]
        idx_acc, pos_acc = gltf["accessors"][prim["indices"]], gltf["accessors"][prim["attributes"]["POSITION"]]
        v = gltf["bufferViews"][idx_acc["bufferView"]]
        idx = struct.unpack_from(f"<{idx_acc['count']}H", blob, v["byteOffset"])
        assert max(idx) < pos_acc["count"] and len(idx) % 3 == 0
        assert all(-0.5001 <= x <= 0.5001 for x in pos_acc["min"] + pos_acc["max"])  # unit shapes
    wheel = next(n for n in parts if n["name"] == "wheel front left")
    assert wheel["scale"] == [0.72, 0.32, 0.72] and wheel["translation"] == [-0.95, 0.36, 1.35]
    x, y, z, w = wheel["rotation"]  # 90° around z
    assert abs(z - math.sin(math.pi / 4)) < 1e-6 and abs(w - math.cos(math.pi / 4)) < 1e-6 and abs(x) < 1e-9 and abs(y) < 1e-9
    lights = [m for m in gltf["materials"] if "emissiveFactor" in m]
    assert lights and "KHR_materials_emissive_strength" in gltf["extensionsUsed"]
    glass = next(m for m in gltf["materials"] if m.get("alphaMode") == "BLEND")
    assert glass["pbrMetallicRoughness"]["baseColorFactor"][3] == 0.5


def test_whatever_the_model_writes_is_cleaned():
    raw = """Here you go:
```json
{"name": "lamp", "parts": [
  {"shape": "cube", "size": [1, 2], "pos": [0, 1, 0], "color": "f00", "metal": 7},
  {"shape": "teapot", "size": "big", "pos": [0, "x", 0], "rot": [0, 0, 1e9]},
  {"shape": "donut", "color": "#00FF00", "glow": 1.2, "opacity": 0.3},
  "nonsense",
]}
```"""
    spec = models3d.parse_spec(raw)
    a, b, c = spec["parts"]
    assert a == {"name": "box 1", "shape": "box", "size": [1, 2, 1], "pos": [0, 1, 0], "rot": [0, 0, 0], "color": "#ff0000", "metal": 1}
    assert b["shape"] == "box" and b["size"] == [1, 1, 1] and b["pos"] == [0, 0, 0] and b["rot"][2] == 720
    assert c["shape"] == "torus" and c["color"] == "#00ff00" and c["glow"] == 1.2 and c["opacity"] == 0.3
    with pytest.raises(ValueError):
        models3d.parse_spec("I can't draw that.")
    with pytest.raises(ValueError):
        models3d.clean({"parts": []})
    assert len(models3d.clean({"parts": [{"shape": "box"}] * 500})["parts"]) == models3d.MAX_PARTS


@pytest.mark.parametrize("text,want", [
    ("make a 3d model of a sports car", {"goal": "make a 3d model of a sports car"}),
    ("create a low poly house", {"goal": "create a low poly house"}),
    ("design a 3d robot for my game", {"goal": "design a 3d robot for my game"}),
    ("change my sports-car model: make it red", {"goal": "change my sports-car model: make it red", "change": "sports-car"}),
    ("open studio", {"open": True, "goal": "open studio"}),
    ("Create me a 3d gta 6 game", None), ("Create a game like gta 6 not gta 6 in 3d", None),
    ("make a racing game with my sports-car model", None), ("train a model on my data", None), ("make an ai model", None),
    ("make a 3d website for my shop", None), ("make a model of the solar system in python", None),
])
def test_what_is_a_3d_model_request(text, want):
    assert models3d.parse(text) == want


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
    asyncio.run(r.create("Maker", "makes my 3d models and games", owner="u_maker0000001"))
    deployed = []

    async def deploy(uid, name, files, cols, private=False):
        deployed.append((uid, name, dict(files), private))
        return {"name": name, "url": f"/d/dpl_m{len(deployed)}/"}
    monkeypatch.setattr(sites.chroma, "ensure", nothing)
    monkeypatch.setattr(sites.chroma, "deploy", deploy)

    async def play(url):
        return {"ok": True, "ready": True, "errors": [], "shot": b"\xff\xd8jpeg"}
    monkeypatch.setattr(games, "play", play)
    return {"registry": r, "deployed": deployed, "cells": tmp_path / "cells"}


def _run(module, text, user="u_maker0000001"):
    events = []

    async def emit(ev):
        events.append(ev)
    t = Task("t1", "c1", text, emit, user=user)
    asyncio.run(module.handle(t, module.parse(text)))
    return events


def test_a_bee_designs_builds_checks_and_saves_a_model(hive, monkeypatch, tmp_path):
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})
    asked = []
    design = {"name": "x", "parts": [{"name": "trunk", "shape": "cylinder", "size": [0.3, 2, 0.3], "pos": [0, 1, 0], "color": "#6b4423"},
                                     {"name": "crown", "shape": "sphere", "size": [2, 2, 2], "pos": [0, 2.8, 0], "color": "#2f8f3a"}]}
    b.transport = httpx.MockTransport(lambda r: asked.append(json.loads(r.content)) or
                                      httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(design)}}]}))
    monkeypatch.setattr(brain_mod, "brain", b)
    events = _run(models3d, "make a 3d model of an oak tree")
    assert "y is up" in asked[0]["messages"][0]["content"] and "oak tree" in asked[0]["messages"][1]["content"]
    root = hive["cells"] / "maker" / "work" / "models"
    saved = json.loads((root / "oak-tree.json").read_text())
    assert [p["name"] for p in saved["parts"]] == ["trunk", "crown"] and saved["name"] == "oak tree"
    read_glb((root / "oak-tree.glb").read_bytes())
    uid, project, files, private = hive["deployed"][0]
    assert private and project == "oak-tree-model" and files["model.glb"] == (root / "oak-tree.glb").read_bytes()
    assert "oak tree" in files["index.html"] and "GLTFLoader" in files["viewer.js"]
    card = next(e for e in events if e["type"] == "result")
    assert card["badge"] == "Private" and ["Edit in Studio", "/studio.html?bee=maker&model=oak-tree"] in card["details"]
    answer = events[-1]["text"]
    assert "What I did:" in answer and "Designed it from 2 parts" in answer and "(/d/dpl_m1/)" in answer
    assert sandbox.of("u_maker0000001")[0] == {**sandbox.of("u_maker0000001")[0], "kind": "model", "name": "oak-tree", "url": "/d/dpl_m1/"}

    # "change my oak-tree model …": the Bee gets the design it made, and changes it
    design["parts"].append({"name": "apple", "shape": "sphere", "size": [0.2, 0.2, 0.2], "pos": [0.8, 2.5, 0.6], "color": "#d1202a"})
    _run(models3d, "change my oak-tree model: add an apple")
    assert '"trunk"' in asked[-1]["messages"][1]["content"] and "add an apple" in asked[-1]["messages"][1]["content"]
    assert len(json.loads((root / "oak-tree.json").read_text())["parts"]) == 3


def test_no_model_running_means_a_ready_made_design_or_studio(hive, monkeypatch):
    monkeypatch.setattr(brain_mod, "is_connected", lambda: False)
    events = _run(models3d, "make a low poly robot")
    assert "ready-made" in events[-1]["text"] and (hive["cells"] / "maker" / "work" / "models" / "robot.glb").exists()
    events = _run(models3d, "make a 3d model of a dragon")
    assert "Mnx Studio" in events[-1]["text"] and not (hive["cells"] / "maker" / "work" / "models" / "dragon.glb").exists()
    assert "/studio.html" in _run(models3d, "open studio")[-1]["text"]


def test_games_use_the_bees_models(hive, monkeypatch):
    monkeypatch.setattr(brain_mod, "is_connected", lambda: False)
    _run(models3d, "make a low poly car")  # → ~/models/car
    events = _run(games, "make a racing game with my car model")
    files = hive["deployed"][-1][2]
    assert files["models/car.glb"][:4] == b"glTF" and 'const PLAYER_MODEL = "models/car.glb";' in files["game.js"]
    assert "Put your 3D model in it: car" in events[-1]["text"]
    assert (hive["cells"] / "maker" / "work" / "games" / "racing-game" / "models" / "car.glb").exists()
    assert [r["kind"] for r in sandbox.of("u_maker0000001")] == ["game", "model"]
    plain = games.template_files()
    assert "const PLAYER_MODEL = null;" in plain["game.js"] and not any(p.startswith("models/") for p in plain)


def test_the_model_writing_a_game_is_told_about_the_models(hive, monkeypatch, tmp_path):
    files = games.with_models(games.template_files(), {"car": b"glTFxxxx"})
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})
    sent = []
    out = "=== FILE: index.html ===\n<html></html>\n=== END ===\n=== FILE: game.js ===\nwindow.__game={ready:true}\n=== END ==="
    b.transport = httpx.MockTransport(lambda r: sent.append(json.loads(r.content)) or
                                      httpx.Response(200, json={"choices": [{"message": {"content": out}}]}))
    monkeypatch.setattr(brain_mod, "brain", b)
    new = asyncio.run(games.write_game(Task("t", "c", "x", None), "a racing game", files))
    prompt = sent[0]["messages"][1]["content"]
    assert "models/car.glb" in prompt and "GLTFLoader" in prompt and "glTFxxxx" not in prompt
    assert new["models/car.glb"] == b"glTFxxxx" and "style.css" in new


@pytest.fixture
def client(tmp_path, monkeypatch, hive):
    from fastapi.testclient import TestClient
    import gateway.api as gw
    monkeypatch.setattr(auth, "USER_TOKENS_FILE", tmp_path / "user_tokens.txt")
    (tmp_path / "user_tokens.txt").write_text("user-token-maker\nuser-token-other\n")
    r = hive["registry"]
    for bee in r.bees:
        bee["owner"] = auth.user_of("user-token-maker")
    monkeypatch.setattr(gw, "registry", r)
    monkeypatch.setattr(gw, "BEES", r.bees)
    return TestClient(gw.app)


ME = {"Authorization": "Bearer user-token-maker"}
OTHER = {"Authorization": "Bearer user-token-other"}


def test_studio_saves_opens_and_exports_only_your_own(client, hive, monkeypatch):
    spec = {"name": "chair", "parts": [{"name": "seat", "shape": "box", "size": [1, 0.1, 1], "pos": [0, 0.5, 0], "color": "#884422"}]}
    assert client.get("/api/studio/bees", headers=ME).json() == {"bees": [{"id": "maker", "name": "Maker"}]}
    r = client.put("/api/studio/maker/models/My Chair", headers=ME, json=spec)
    assert r.status_code == 200 and r.json()["saved"] == "~/models/my-chair.glb"
    assert client.get("/api/studio/maker/models", headers=ME).json() == {"models": ["my-chair"]}
    assert client.get("/api/studio/maker/models/my-chair", headers=ME).json()["parts"][0]["name"] == "seat"
    glb = client.get("/api/studio/maker/models/my-chair/glb", headers=ME)
    assert glb.headers["content-type"] == "model/gltf-binary" and read_glb(glb.content)[0]["nodes"][0]["name"] == "seat"
    assert client.put("/api/studio/maker/models/x", headers=ME, json={"parts": []}).status_code == 400
    for method, path in (("get", "/api/studio/maker/models"), ("get", "/api/studio/maker/models/my-chair"),
                         ("put", "/api/studio/maker/models/evil"), ("post", "/api/studio/maker/ask"),
                         ("post", "/api/studio/maker/models/my-chair/view")):
        kw = {"json": spec if method == "put" else {"text": "x"}} if method != "get" else {}
        assert getattr(client, method)(path, headers=OTHER, **kw).status_code == 404, path
    assert client.get("/api/studio/bees", headers=OTHER).json() == {"bees": []}
    assert client.get("/api/studio/maker/models", headers={"Authorization": "Bearer wrong"}).status_code == 401

    v = client.post("/api/studio/maker/models/my-chair/view", headers=ME)
    assert v.status_code == 200 and v.json()["url"].startswith("/d/") and hive["deployed"][-1][3] is True
    assert client.get("/api/sandbox", headers=ME).json()["items"][0]["studio"] == "/studio.html?bee=maker&model=my-chair"
    assert client.get("/api/sandbox", headers=OTHER).json() == {"items": []}


def test_studio_asks_the_bee_to_change_the_design(client, monkeypatch, tmp_path):
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})
    sent = []
    out = {"name": "chair", "parts": [{"shape": "box"}, {"shape": "cylinder", "name": "leg"}]}
    b.transport = httpx.MockTransport(lambda r: sent.append(json.loads(r.content)) or
                                      httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(out)}}]}))
    monkeypatch.setattr(brain_mod, "brain", b)
    r = client.post("/api/studio/maker/ask", headers=ME, json={"text": "add a leg", "spec": {"parts": [{"shape": "box"}]}})
    assert r.status_code == 200 and [p["shape"] for p in r.json()["parts"]] == ["box", "cylinder"]
    assert "add a leg" in sent[0]["messages"][1]["content"] and '"box"' in sent[0]["messages"][1]["content"]


def test_the_studio_and_sandbox_pages_are_served(client):
    for page, needle in (("/studio.html", "js/studio.js"), ("/sandbox.html", "openRunner")):
        r = client.get(page)
        assert r.status_code == 200 and needle in r.text


def _sse(chunks):
    body = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


HOWTO = ["Creating an open-world game in 3D is a complex task that requires a deep understanding of game design. ", "Step 1: ..."]


@pytest.mark.parametrize("effort,replies,asks,final", [
    ("med", [HOWTO, ["```html\n<canvas></canvas>\n```\nMade: a 3D city."]], 2, "Made: a 3D city."),
    ("maxxxx", [["As an AI, I'm unable to create video games."], ["I cannot create that."], ["Here is the game: ..."]], 3, "Here is the game"),
    ("med", [["As an AI, I'm unable to create video games."], ["I cannot do that."]], 2, "wouldn't do that one"),
])
def test_how_tos_and_refusals_are_asked_again(tmp_path, monkeypatch, effort, replies, asks, final):
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})
    sent = []
    b.transport = httpx.MockTransport(lambda r: sent.append(json.loads(r.content)) or _sse(replies.pop(0)))
    monkeypatch.setattr(brain_mod, "brain", b)
    events = []

    async def emit(ev):
        events.append(ev)
    t = Task("t1", "c1", "Create a open world game in 3d", emit, effort=effort)
    asyncio.run(respond.respond(t, brain_mod.system_prompt(effort), web=False))
    shown = json.dumps([e for e in events if e["type"] in ("answer", "answer_delta")])
    assert len(sent) == asks and "complex task" not in shown and "unable" not in shown
    assert final in events[-1]["text"]


def test_a_how_to_question_is_still_answered_as_a_how_to(tmp_path, monkeypatch):
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434"})
    sent = []
    b.transport = httpx.MockTransport(lambda r: sent.append(1) or _sse(["Here is a step-by-step guide: 1. Install Godot."]))
    monkeypatch.setattr(brain_mod, "brain", b)

    async def emit(ev):
        pass
    t = Task("t1", "c1", "how do I make a game in Godot?", emit)
    assert asyncio.run(respond.respond(t, brain_mod.system_prompt(), web=False)).startswith("Here is a step-by-step guide")
    assert len(sent) == 1
