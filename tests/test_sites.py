"""Bees build full-stack sites with Chroma: what counts as asking for one, reading the model's files, deploying, and
keeping the files in the Bee's (Vortex OS) computer."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import sites  # noqa: E402


@pytest.mark.parametrize("text,kind,name", [
    ("build a website for my bakery where people can order cakes", "build", ""),
    ("create a landing page for my app with a waitlist", "build", ""),
    ("create a full stack web app called Tasky for team todos", "build", "tasky"),
    ("update my bakery site: add a contact page", "update", "bakery"),
    ("list my sites", "list", None),
])
def test_site_requests(text, kind, name):
    got = sites.parse(text)
    assert got["kind"] == kind and (name is None or got["name"] == name)


@pytest.mark.parametrize("text", ["make me a todo app in react", "how do I build a website", "build a neural network in numpy",
                                  "teach me python", "add KitKat in blinkit"])
def test_not_site_requests(text):
    assert sites.parse(text) is None


def test_names_from_the_request():
    assert sites.name_for("build a website for my bakery where people can order cakes") == "bakery"
    assert sites.name_for("make a portfolio site for my photos") == "photos"
    assert sites.slug("Tharun's Café!!") == "tharun-s-caf"


MODEL_OUTPUT = """Here you go.
=== FILE: index.html ===
<!doctype html><html><head><link rel="stylesheet" href="style.css"></head><body><h1>Cake Corner</h1>
<script type="module" src="app.js"></script></body></html>
=== END ===
=== FILE: style.css ===
body { font-family: system-ui; }
=== END ===
=== FILE: app.js ===
import { chroma } from '/assets/js/chroma-sdk.js';
const app = chroma();
=== END ===
=== FILE: api/total.js ===
export default async function (req, { db }) { return { json: { total: 1 } }; }
=== END ===
=== FILE: ../../etc/passwd ===
nope
=== END ===
=== COLLECTIONS ===
orders: read=private write=create
menu: read=public write=private
=== END ===
"""


def test_reading_the_models_files():
    files, cols = sites.parse_files(MODEL_OUTPUT)
    assert list(files) == ["index.html", "style.css", "app.js", "api/total.js"]  # nothing outside the site
    assert cols == [{"name": "orders", "read": "private", "write": "create"}, {"name": "menu", "read": "public", "write": "private"}]
    files, cols = sites.parse_files("Sure!\n<!doctype html><html><body>Hi</body></html>\nEnjoy")
    assert list(files) == ["index.html"] and cols == []


class FakeChroma:
    """Answers like Chroma's API: accounts, tokens, projects (names shared by everyone), deployments, collections."""

    def __init__(self):
        self.users, self.tokens, self.projects, self.collections, self.deployed = {}, {}, {"bakery": "someone-else"}, [], {}

    def handler(self, req: httpx.Request) -> httpx.Response:
        path, body = req.url.path, (json.loads(req.content) if req.content else {})
        who = self.tokens.get(req.headers.get("authorization", "").removeprefix("Bearer "))
        if path == "/api/health":
            return httpx.Response(200, json={"ok": True})
        if path == "/api/auth/signup":
            self.users[body["username"]] = body["password"]
            return httpx.Response(201, json={"user": {"username": body["username"]}})
        if path == "/api/auth/token":
            assert self.users.get(body["username"]) == body["password"]
            tok = "chr_" + body["username"]
            self.tokens[tok] = body["username"]
            return httpx.Response(201, json={"token": tok})
        assert who, "every other call carries the user's token"
        if path == "/api/projects" and req.method == "GET":
            return httpx.Response(200, json={"projects": [{"name": n, "url": f"/p/{n}/", "deployments": 1} for n, o in self.projects.items() if o == who]})
        if path == "/api/projects" and req.method == "POST":
            if body["name"] in self.projects:
                return httpx.Response(409, json={"error": "that project name is taken"})
            self.projects[body["name"]] = who
            return httpx.Response(201, json={"project": {"name": body["name"]}})
        name = path.split("/")[3]
        assert self.projects.get(name) == who
        if path.endswith("/deployments"):
            self.deployed[name] = {f["path"]: base64.b64decode(f["data"]).decode() for f in body["files"]}
            fns = [p[4:-3] for p in self.deployed[name] if p.startswith("api/")]
            return httpx.Response(201, json={"production": f"/p/{name}/", "deployment": {"id": "dpl_1", "status": "READY", "url": "/d/dpl_1/",
                                                                                        "functions": fns, "log": []}})
        if path.endswith("/db"):
            self.collections.append((name, body))
            return httpx.Response(201, json={"collection": body})
        return httpx.Response(404, json={"error": "no"})


def test_deploying_for_a_user(tmp_path):
    fake = FakeChroma()
    c = sites.Chroma(tmp_path / "chroma")
    c.transport = httpx.MockTransport(fake.handler)
    files, cols = sites.parse_files(MODEL_OUTPUT)
    d = asyncio.run(c.deploy("u_baker0000001", "bakery", files, cols))
    assert d["name"] == "bakery-2" and d["url"] == "/p/bakery-2/" and d["functions"] == ["total"]  # "bakery" belongs to someone else
    assert fake.deployed["bakery-2"]["api/total.js"].startswith("export default") and len(fake.collections) == 2
    acc = json.loads((tmp_path / "chroma" / "accounts.json").read_text())
    assert acc["u_baker0000001"]["username"] == "mnx-ubaker0000001" and oct((tmp_path / "chroma" / "accounts.json").stat().st_mode)[-3:] == "600"
    again = asyncio.run(c.deploy("u_baker0000001", "bakery-2", {"index.html": "<h1>v2</h1>"}, []))  # an update: same project
    assert again["name"] == "bakery-2" and fake.deployed["bakery-2"] == {"index.html": "<h1>v2</h1>"}
    assert len(fake.users) == 1  # one account per Hive user, kept
    other = asyncio.run(c.projects("u_other0000001"))
    assert other == []  # nobody sees another user's sites


def test_the_files_go_into_the_bees_computer(tmp_path, monkeypatch):
    from hive import cells as cells_mod
    monkeypatch.setattr(cells_mod, "CELLS", tmp_path / "cells")
    monkeypatch.setattr(cells_mod.cells, "cells", {})
    monkeypatch.setattr(cells_mod.cells, "mode", "local")
    b = {"id": "webby", "name": "Webby", "computer": {"size": "free", "where": "here", "os": "vortex"}}
    files, _ = sites.parse_files(MODEL_OUTPUT)
    where = asyncio.run(sites.save_to_bee(b, "bakery-2", files, "bakery-2"))
    root = tmp_path / "cells" / "webby" / "work" / "sites" / "bakery-2"
    assert where == "~/sites/bakery-2" and (root / "api" / "total.js").exists()
    assert json.loads((root / "chroma.json").read_text())["name"] == "bakery-2"
    assert asyncio.run(sites.read_from_bee(b, "bakery-2")) == files  # what an update starts from


def test_the_hive_passes_sites_through_decoded_and_sandboxed(monkeypatch):
    import gzip
    from fastapi.testclient import TestClient
    import gateway.api as gw
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path, req.headers.get("authorization")))
        if req.url.path == "/api/health":
            return httpx.Response(200, json={"ok": True})
        body = gzip.compress(b"export const hi = '\xe2\x82\xb9650';")
        return httpx.Response(200, content=body, headers={"content-encoding": "gzip", "content-type": "text/javascript",
                                                          "content-security-policy": "sandbox allow-scripts"})
    monkeypatch.setattr(sites.chroma, "transport", httpx.MockTransport(handler))
    client = TestClient(gw.app)
    r = client.get("/p/bakery/app.js", headers={"Authorization": "Bearer the-users-hive-token"})
    assert r.status_code == 200 and r.text == "export const hi = '₹650';" and r.headers["content-security-policy"] == "sandbox allow-scripts"
    assert seen[-1] == ("GET", "/p/bakery/app.js", None)  # the Hive's token never goes to the site server
    assert client.get("/vendor/xterm/xterm.js").status_code in (200, 404) and seen[-1][1] == "/p/bakery/app.js"  # the Hive's own files stay its own


def test_site_names_from_what_it_is():
    assert sites.name_for("create a landing page for my app with a waitlist") == "landing-waitlist"
