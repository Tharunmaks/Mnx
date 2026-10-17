"""Bees that make Bees: the registry's limits and family rules, computers, keys, the Bee-key API and the chat phrases."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import bee_chat, bees, cells as cells_mod  # noqa: E402
from hive.bees import BeeError, Registry  # noqa: E402

U = "u_test0000001"  # a user on the Free plan (the owner is always on Pro)


@pytest.fixture
def reg(tmp_path):
    r = Registry(tmp_path / "hive.json", tmp_path / "bee_keys.json")
    calls = {"created": [], "deleted": [], "computer": []}

    async def created(b):
        calls["created"].append(b["id"])

    async def deleted(b):
        calls["deleted"].append(b["id"])

    async def computer(b, old):
        calls["computer"].append((b["id"], old, dict(b["computer"])))

    r.on_created, r.on_deleted, r.on_computer = created, deleted, computer
    r.calls = calls
    return r


def run(coro):
    return asyncio.run(coro)


def test_no_premade_bees_and_saved_bees_load_back(reg, tmp_path):
    assert reg.bees == []
    run(reg.create("Scout", "watches prices", owner=U))
    again = Registry(tmp_path / "hive.json", tmp_path / "bee_keys.json")
    s = again.get("scout")
    assert s["created_by"] == "you" and s["skill"] == "watches prices" and s["owner"] == U and s["bot"]["tone"] == "friendly"


def test_old_built_in_bees_are_dropped_and_old_bees_become_the_owners(tmp_path):
    state = tmp_path / "hive.json"
    state.write_text(json.dumps({"bees": [
        {"id": "chat", "name": "Chat", "skill": "Talks", "builtin": True, "computer": {"size": "standard"}},
        {"id": "phone", "name": "Phone", "skill": "Taps", "builtin": True},
        {"id": "scout", "name": "Scout", "skill": "watches", "builtin": False, "created_by": "you",
         "computer": {"size": "xl", "cpus": 8, "memory_gb": 32, "where": "here"}}]}))
    r = Registry(state, tmp_path / "keys.json")
    assert [b["id"] for b in r.bees] == ["scout"]
    s = r.get("scout")
    assert s["owner"] == "owner" and s["computer"]["size"] == "pro" and s["computer"]["cpus"] == 4  # no more XL: the plan's computer
    state.write_text(json.dumps({"bees": [{"id": "old", "name": "Old", "skill": "x", "created_by": "you",
                                            "computer": {"size": "standard", "cpus": 2, "memory_gb": 8, "where": "here"}}]}))
    assert Registry(state, tmp_path / "keys.json").get("old")["computer"]["size"] == "pro"  # the owner's old Standard Bee: Pro now
    assert isinstance(s["bot"], dict)


def test_create_and_family(reg):
    scout = run(reg.create("Scout", "watches prices", owner=U))
    alert = run(reg.create("Price Alert", "sends alerts", created_by="scout"))
    assert scout["computer"] == {"size": "free", "cpus": 2, "memory_gb": 8, "disk_gb": 40, "gpu": False, "where": "here", "os": "vortex"}
    assert alert["id"] == "price_alert" and alert["created_by"] == "scout" and alert["owner"] == U  # a child is its parent's owner's
    assert [c["id"] for c in reg.children("scout")] == ["price_alert"]
    assert reg.depth("price_alert") == 2 and reg.calls["created"] == ["scout", "price_alert"]
    view = reg.view(alert)
    assert view["parent_name"] == "Scout" and view["computer_text"] == "Free · 2 CPUs, 8 GB RAM, 40 GB disk · Vortex OS"


def test_each_user_has_their_own_bees(reg):
    run(reg.create("Scout", "watches prices", owner=U))
    other = run(reg.create("Scout", "another user's scout", owner="u_other000001"))
    assert other["id"] == "scout_2"  # same name for another user is fine; ids stay unique
    assert [b["id"] for b in reg.of(U)] == ["scout"] and [b["id"] for b in reg.of("u_other000001")] == ["scout_2"]
    assert reg.mine("scout", U) and reg.mine("scout", "u_other000001") is None and reg.mine("scout", "owner")
    assert reg.find("scout", U)["id"] == "scout" and reg.find("scout", "u_other000001")["id"] == "scout_2"
    assert run(reg.create("Browser", "reads the web", owner=U))["id"] == "browser_2"  # never one of the Hive's own ids


def test_limits(reg, monkeypatch):
    with pytest.raises(BeeError):
        run(reg.create("Scout", "", owner=U))
    run(reg.create("Scout", "watches", owner=U))
    with pytest.raises(BeeError, match="already have"):
        run(reg.create("scout", "again", owner=U))
    with pytest.raises(BeeError, match="doesn't exist"):
        run(reg.create("Orphan", "x", created_by="nobody"))
    parent = "scout"
    for i in range(bees.MAX_DEPTH - 1):
        parent = run(reg.create(f"Gen{i}", "x", created_by=parent))["id"]
    with pytest.raises(BeeError, match="generations"):
        run(reg.create("TooDeep", "x", created_by=parent))
    run(reg.create("Fifth", "x", owner=U))  # Free has room for 5 Bees, children included
    with pytest.raises(BeeError, match="room for 5"):
        run(reg.create("Sixth", "x", owner=U))
    monkeypatch.setattr(bees, "MAX_CHILDREN", 1)
    run(reg.create("Boss", "x", owner="owner"))
    run(reg.create("A1", "x", created_by="boss"))
    with pytest.raises(BeeError, match="children"):
        run(reg.create("A2", "x", created_by="boss"))


def test_delete_hands_children_to_the_grandparent(reg):
    run(reg.create("Scout", "watches", owner=U))
    run(reg.create("Kid", "helps", created_by="scout"))
    run(reg.create("Grandkid", "helps more", created_by="kid"))
    key = reg.key_for("kid")
    adopted = run(reg.delete("kid"))
    assert [b["id"] for b in adopted] == ["grandkid"] and reg.get("grandkid")["created_by"] == "scout"
    assert reg.by_key(key) is None and reg.calls["deleted"] == ["kid"]


def test_keys_are_secret_per_bee_and_stored_owner_only(reg, tmp_path):
    run(reg.create("Scout", "watches", owner=U))
    run(reg.create("Mail", "reads mail", owner=U))
    k1, k2 = reg.key_for("scout"), reg.key_for("mail")
    assert k1 != k2 and len(k1) >= 24 and reg.by_key(k1)["id"] == "scout" and reg.by_key("nope") is None and reg.by_key("") is None
    assert oct((tmp_path / "bee_keys.json").stat().st_mode)[-3:] == "600"
    assert "key" not in reg.view(reg.get("scout")) and k1 not in json.dumps(reg.view(reg.get("scout")))


def test_the_plan_decides_the_computer(reg, monkeypatch, tmp_path):
    from hive.plan import plan
    run(reg.create("Scout", "watches", owner=U))
    assert reg.get("scout")["computer"]["size"] == "free"
    plan.grant(U, 30, {"id": "pay_1"})
    changed = run(reg.apply_plan(U))
    c = reg.get("scout")["computer"]
    assert [b["id"] for b in changed] == ["scout"] and c["size"] == "pro" and c["cpus"] == 4 and c["memory_gb"] == 16
    assert c["disk_gb"] == 100 and c["gpu"]
    (bee_id, old, new), = reg.calls["computer"]
    assert old["size"] == "free" and new["size"] == "pro"
    plan.revoke(U)
    run(reg.apply_plan(U))
    assert reg.get("scout")["computer"]["size"] == "free" and not reg.get("scout")["computer"]["gpu"]
    assert run(reg.apply_plan(U)) == []  # nothing left to change


def test_a_failed_move_keeps_the_old_computer(reg):
    run(reg.create("Scout", "watches", owner=U))

    async def boom(b, old):
        raise RuntimeError("server unreachable")

    reg.on_computer = boom
    with pytest.raises(RuntimeError):
        run(reg.move("scout", "gone-server"))
    assert reg.get("scout")["computer"]["where"] == "here"
    with pytest.raises(BeeError):
        run(reg.move("scout", "../etc"))


def test_find(reg):
    run(reg.create("Mail Helper", "reads mail", owner=U))
    assert reg.find("mail helper", U)["id"] == "mail_helper" and reg.find("Mail Helper Bee", U)["id"] == "mail_helper"
    assert reg.find("mail", U)["id"] == "mail_helper" and reg.find("coding", U) is None
    assert reg.find("", U) is None and reg.find("mail helper", "u_someone_else") is None


# ---------------------------------------------------------------- cells: limits and remote paths
def test_docker_limits_follow_the_computer(monkeypatch, tmp_path):
    monkeypatch.setattr(cells_mod, "CELLS", tmp_path)
    monkeypatch.setattr(cells_mod, "machine", lambda: {"cpus": 4, "memory_gb": 8.0})
    cell = cells_mod.Cell("scout", "Scout", "docker", {"size": "xl", "cpus": 8, "memory_gb": 32, "gpu": True})
    monkeypatch.setattr(cells_mod, "GPU_DOCKER", True)
    flags = cell._limits()
    assert flags[flags.index("--cpus") + 1] == "4" and flags[flags.index("--memory") + 1] == "7.20g" and "--gpus" in flags
    monkeypatch.setattr(cells_mod, "GPU_DOCKER", False)  # no GPU here: the container starts, its GPU comes from the pool
    assert "--gpus" not in cell._limits()
    cell.computer = {"size": "max", "cpus": None, "memory_gb": None, "gpu": False}
    assert cell._limits() == []
    assert (tmp_path / "scout" / "work" / ".mnx" / "mnx-bee").read_text().startswith("#!/usr/bin/env python3")


def test_remote_paths_stay_inside_the_cell():
    assert cells_mod.rpath("~/mnx-runs/cells/scout", "a/b.txt") == '"$HOME"/' + "mnx-runs/cells/scout/a/b.txt"
    assert cells_mod.rpath("/workspace/cells/scout", "../../etc/passwd") == "/workspace/cells/scout/etc/passwd"
    assert cells_mod.rpath("/w/c", "x y; rm -rf /") == "'/w/c/x y; rm -rf '"  # quoted: one odd file name, not a command


def test_remote_cells_run_over_ssh(monkeypatch, tmp_path):
    monkeypatch.setattr(cells_mod, "CELLS", tmp_path)
    seen = []

    async def fake_run_in(argv, timeout=60, stdin=None):
        seen.append((argv, stdin))
        return 0, "MNX_UP\n"

    monkeypatch.setattr(cells_mod, "_run_in", fake_run_in)
    remote = {"ssh": ["ssh", "-p", "22", "u@h", "--"], "dir": "~/runs/cells/scout", "name": "GPU box"}
    cell = cells_mod.Cell("scout", "Scout", "local", {"size": "max"}, remote, {"MNX_BEE_KEY": "k", "MNX_HIVE": "https://hive"})
    assert cell.mode == "remote" and cell.view()["server"] == "GPU box"
    code, out = run(cell.exec("echo hi"))
    up_argv, up_stdin = seen[0]
    assert up_argv[:4] == ["ssh", "-p", "22", "u@h"] and b"export MNX_BEE_KEY=k" in up_stdin and b"MNX_HIVE=https://hive" in up_stdin
    assert seen[1][0][-1].startswith('cd "$HOME"/runs/cells/scout && . .mnx/env && sh -lc')
    tty = cell.shell_argv()
    assert tty[:2] == ["ssh", "-tt"] and "bash -l" in tty[-1]


# ---------------------------------------------------------------- the Bee-key API
def test_bee_api_lets_a_bee_make_only_its_own_children(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import gateway.api as gw
    r = Registry(tmp_path / "hive.json", tmp_path / "bee_keys.json")

    async def nothing(*a):
        return None

    r.on_created = r.on_deleted = r.on_computer = nothing
    monkeypatch.setattr(gw, "registry", r)
    client = TestClient(gw.app)
    run(r.create("Scout", "watches prices", owner=U))
    key = r.key_for("scout")
    assert client.get("/bee-api/me").status_code == 401
    assert client.get("/bee-api/me", headers={"X-Bee-Key": "wrong"}).status_code == 401
    me = client.get("/bee-api/me", headers={"X-Bee-Key": key}).json()
    assert me["id"] == "scout" and "key" not in me
    made = client.post("/bee-api/children", headers={"X-Bee-Key": key}, json={"name": "Price Alert", "skill": "sends alerts"})
    assert made.status_code == 200 and made.json()["created_by"] == "scout" and made.json()["computer"]["size"] == "free"
    assert r.get("price_alert")["owner"] == U
    kids = client.get("/bee-api/children", headers={"X-Bee-Key": key}).json()
    assert [k["id"] for k in kids] == ["price_alert"]
    clash = client.post("/bee-api/children", headers={"X-Bee-Key": key}, json={"name": "Scout", "skill": "x"})
    assert clash.status_code == 400
    # the Bee key opens nothing else
    assert client.get("/api/bees", headers={"X-Bee-Key": key}).status_code == 401


# ---------------------------------------------------------------- chat phrases
@pytest.mark.parametrize("text,want", [
    ("list my bees", {"kind": "list"}),
    ("delete the scout bee", {"kind": "delete", "bee": "scout"}),
    ("give the coding bee a large computer", {"kind": "computer"}),  # not a choice any more: it explains the plan
    ("what computer do my bees get?", {"kind": "computer"}),
])
def test_bee_phrases(text, want):
    got = bee_chat.parse(text, U)
    assert got is not None
    for k, v in want.items():
        assert got[k] == v, (k, got)


@pytest.mark.parametrize("text", ["create a 10M model for stories", "run my stories model on the lpu", "train my model", "hello"])
def test_not_bee_phrases(text):
    assert bee_chat.parse(text, U) is None


@pytest.mark.parametrize("text,want", [
    ("create a bee called Scout that watches prices", {"name": "Scout", "description": "watches prices", "parent": None}),
    ("tell the coding bee to create a bee called Tester that runs the tests", {"name": "Tester", "parent": "coding"}),
    ("make a new bee named Mail Helper to summarise my emails every morning", {"name": "Mail Helper", "description": "summarise my emails every morning"}),
    ("create a bee that translates Tamil", {"name": "", "description": "translates Tamil"}),
    ("make a bot called Maya that is a friendly Tamil tutor", {"name": "Maya", "description": "a friendly Tamil tutor"}),
])
def test_making_a_bee_in_the_chat(text, want):
    from hive import bots
    got = bots.parse(text)
    assert got is not None
    for k, v in want.items():
        assert got[k] == v, (k, got)


def test_the_helper_reads_its_key_file_when_the_shell_starts_clean(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(cells_mod, "CELLS", tmp_path)
    cells_mod.Cell("scout", "Scout", "local", None, None, {"MNX_HIVE": "http://127.0.0.1:9", "MNX_BEE_KEY": "k'1 x"})
    mnx = tmp_path / "scout" / "work" / ".mnx"
    assert oct((mnx / "env").stat().st_mode)[-3:] == "600"
    # a clean environment (no MNX_*): the helper finds the Hive from the file and tries to reach it
    r = subprocess.run([sys.executable, str(mnx / "mnx-bee"), "me"], env={"PATH": os.environ["PATH"]}, capture_output=True, text=True, timeout=30)
    assert "can't reach the Hive" not in (r.stdout + r.stderr) and r.returncode != 0  # port 9: refused, but it knew where to go
