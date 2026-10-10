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
from hive.bees import BeeError, Registry, normalize_computer  # noqa: E402


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


def test_built_ins_are_there_and_saved_bees_load_back(reg, tmp_path):
    assert reg.get("coding")["builtin"] and reg.get("coding")["computer"]["size"] == "small"
    run(reg.create("Scout", "watches prices"))
    again = Registry(tmp_path / "hive.json", tmp_path / "bee_keys.json")
    assert again.get("scout")["created_by"] == "you" and again.get("scout")["skill"] == "watches prices"


def test_create_and_family(reg):
    scout = run(reg.create("Scout", "watches prices", computer={"size": "large"}))
    alert = run(reg.create("Price Alert", "sends alerts", created_by="scout"))
    assert scout["computer"]["cpus"] == 4 and scout["computer"]["memory_gb"] == 16
    assert alert["id"] == "price_alert" and alert["created_by"] == "scout"
    assert alert["computer"]["size"] == "small"  # children start small …
    assert [c["id"] for c in reg.children("scout")] == ["price_alert"]
    assert reg.depth("price_alert") == 2 and reg.calls["created"] == ["scout", "price_alert"]
    view = reg.view(alert)
    assert view["parent_name"] == "Scout" and "Small" in view["computer_text"]


def test_children_start_on_their_parents_machine(reg):
    run(reg.create("Big", "does heavy work", computer={"size": "xl", "where": "gpu-box"}))
    kid = run(reg.create("Kid", "helps", created_by="big"))
    assert kid["computer"]["where"] == "gpu-box" and kid["computer"]["size"] == "small"


def test_built_in_bees_can_make_bees(reg):
    t = run(reg.create("Tester", "runs the tests", created_by="coding"))
    assert t["created_by"] == "coding" and reg.view(t)["parent_name"] == "Coding"


def test_limits(reg, monkeypatch):
    with pytest.raises(BeeError):
        run(reg.create("Scout", ""))
    run(reg.create("Scout", "watches"))
    with pytest.raises(BeeError, match="already exists"):
        run(reg.create("scout", "again"))
    with pytest.raises(BeeError, match="doesn't exist"):
        run(reg.create("Orphan", "x", created_by="nobody"))
    parent = "scout"
    for i in range(bees.MAX_DEPTH - 1):
        parent = run(reg.create(f"Gen{i}", "x", created_by=parent))["id"]
    with pytest.raises(BeeError, match="generations"):
        run(reg.create("TooDeep", "x", created_by=parent))
    monkeypatch.setattr(bees, "MAX_CHILDREN", 2)
    run(reg.create("A1", "x", created_by="coding"))
    run(reg.create("A2", "x", created_by="coding"))
    with pytest.raises(BeeError, match="children"):
        run(reg.create("A3", "x", created_by="coding"))
    monkeypatch.setattr(bees, "MAX_BEES", len(reg.bees))
    with pytest.raises(BeeError, match="limit"):
        run(reg.create("Overflow", "x"))


def test_delete_hands_children_to_the_grandparent(reg):
    run(reg.create("Scout", "watches"))
    run(reg.create("Kid", "helps", created_by="scout"))
    run(reg.create("Grandkid", "helps more", created_by="kid"))
    key = reg.key_for("kid")
    adopted = run(reg.delete("kid"))
    assert [b["id"] for b in adopted] == ["grandkid"] and reg.get("grandkid")["created_by"] == "scout"
    assert reg.by_key(key) is None and reg.calls["deleted"] == ["kid"]
    with pytest.raises(BeeError, match="Built-in"):
        run(reg.delete("coding"))


def test_keys_are_secret_per_bee_and_stored_owner_only(reg, tmp_path):
    run(reg.create("Scout", "watches"))
    k1, k2 = reg.key_for("scout"), reg.key_for("coding")
    assert k1 != k2 and len(k1) >= 24 and reg.by_key(k1)["id"] == "scout" and reg.by_key("nope") is None and reg.by_key("") is None
    assert oct((tmp_path / "bee_keys.json").stat().st_mode)[-3:] == "600"
    assert "key" not in reg.view(reg.get("scout")) and k1 not in json.dumps(reg.view(reg.get("scout")))


def test_computers(reg):
    assert normalize_computer({"size": "max"})["cpus"] is None
    assert normalize_computer({"cpus": 3})["size"] == "custom"
    with pytest.raises(BeeError):
        normalize_computer({"size": "galactic"})
    with pytest.raises(BeeError):
        normalize_computer({"where": "../etc"})
    run(reg.create("Scout", "watches"))
    b = run(reg.set_computer("scout", {"size": "xl", "gpu": True}))
    assert b["computer"]["cpus"] == 8 and b["computer"]["gpu"]
    (bee_id, old, new), = reg.calls["computer"]
    assert bee_id == "scout" and old["size"] == "small" and new["size"] == "xl"


def test_a_failed_move_keeps_the_old_computer(reg):
    run(reg.create("Scout", "watches"))

    async def boom(b, old):
        raise RuntimeError("server unreachable")

    reg.on_computer = boom
    with pytest.raises(RuntimeError):
        run(reg.set_computer("scout", {"where": "gone-server"}))
    assert reg.get("scout")["computer"]["where"] == "here"


def test_find(reg):
    run(reg.create("Mail Helper", "reads mail"))
    assert reg.find("mail helper")["id"] == "mail_helper" and reg.find("the coding bee") is None
    assert reg.find("coding")["id"] == "coding" and reg.find("Coding Bee")["id"] == "coding"
    assert reg.find("") is None


# ---------------------------------------------------------------- cells: limits and remote paths
def test_docker_limits_follow_the_computer(monkeypatch, tmp_path):
    monkeypatch.setattr(cells_mod, "CELLS", tmp_path)
    monkeypatch.setattr(cells_mod, "machine", lambda: {"cpus": 4, "memory_gb": 8.0})
    cell = cells_mod.Cell("scout", "Scout", "docker", {"size": "xl", "cpus": 8, "memory_gb": 32, "gpu": True})
    flags = cell._limits()
    assert flags[flags.index("--cpus") + 1] == "4" and flags[flags.index("--memory") + 1] == "7.20g" and "--gpus" in flags
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
    run(r.create("Scout", "watches prices"))
    key = r.key_for("scout")
    assert client.get("/bee-api/me").status_code == 401
    assert client.get("/bee-api/me", headers={"X-Bee-Key": "wrong"}).status_code == 401
    me = client.get("/bee-api/me", headers={"X-Bee-Key": key}).json()
    assert me["id"] == "scout" and "key" not in me
    made = client.post("/bee-api/children", headers={"X-Bee-Key": key}, json={"name": "Price Alert", "skill": "sends alerts", "size": "medium"})
    assert made.status_code == 200 and made.json()["created_by"] == "scout" and made.json()["computer"]["size"] == "medium"
    kids = client.get("/bee-api/children", headers={"X-Bee-Key": key}).json()
    assert [k["id"] for k in kids] == ["price_alert"]
    clash = client.post("/bee-api/children", headers={"X-Bee-Key": key}, json={"name": "Scout", "skill": "x"})
    assert clash.status_code == 400
    # the Bee key opens nothing else
    assert client.get("/api/bees", headers={"X-Bee-Key": key}).status_code == 401


# ---------------------------------------------------------------- chat phrases
@pytest.mark.parametrize("text,want", [
    ("create a bee called Scout that watches prices", {"kind": "create", "name": "Scout", "skill": "watches prices", "parent": None}),
    ("tell the coding bee to create a bee called Tester that runs the tests", {"kind": "create", "name": "Tester", "parent": "coding"}),
    ("make a new bee named Mail Helper to summarise my emails every morning", {"name": "Mail Helper", "skill": "summarise my emails every morning"}),
    ("create a bee that translates Tamil", {"kind": "create", "name": "", "skill": "translates Tamil"}),
    ("make a large bee with a gpu called Trainer that trains models", {"size": "large", "gpu": True, "name": "Trainer", "skill": "trains models"}),
    ("give the coding bee a large computer", {"kind": "computer", "bee": "coding", "size": "large"}),
    ("upgrade the coding bee to xl", {"kind": "computer", "bee": "coding", "size": "xl"}),
    ("give the eval bee the most powerful computer", {"kind": "computer", "bee": "eval", "size": "max", "powerful": True}),
    ("list my bees", {"kind": "list"}),
    ("delete the scout bee", {"kind": "delete", "bee": "scout"}),
])
def test_bee_phrases(text, want):
    got = bee_chat.parse(text)
    assert got is not None
    for k, v in want.items():
        assert got[k] == v, (k, got)


@pytest.mark.parametrize("text", ["create a 10M model for stories", "run my stories model on the lpu", "train my model", "hello"])
def test_not_bee_phrases(text):
    assert bee_chat.parse(text) is None


def test_cloud_and_gpu_bees_default_to_the_whole_machine(reg, tmp_path):
    assert reg.get("cloud")["computer"]["size"] == "max" and reg.get("gpu")["computer"]["size"] == "max"
    assert reg.get("cloud")["computer"]["cpus"] is None and reg.get("coding")["computer"]["size"] == "small"
    assert "whole machine" in reg.view(reg.get("gpu"))["computer_text"]


def test_saved_built_ins_follow_the_default_until_you_choose(tmp_path):
    state, keys = tmp_path / "hive.json", tmp_path / "bee_keys.json"
    # a Hive saved before the default existed: every built-in is stored as Small
    old = Registry(state, keys)
    for b in old.bees:
        b["computer"] = bees.default_computer()
    old.save()
    again = Registry(state, keys)
    assert again.get("cloud")["computer"]["size"] == "max" and again.get("coding")["computer"]["size"] == "small"
    # you choose Small for Cloud on purpose: that sticks
    run(again.set_computer("cloud", {"size": "small"}))
    assert Registry(state, keys).get("cloud")["computer"]["size"] == "small"
    # and a size you chose for another built-in is kept
    run(again.set_computer("coding", {"size": "xl"}))
    assert Registry(state, keys).get("coding")["computer"]["size"] == "xl"


def test_the_helper_reads_its_key_file_when_the_shell_starts_clean(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(cells_mod, "CELLS", tmp_path)
    cells_mod.Cell("scout", "Scout", "local", None, None, {"MNX_HIVE": "http://127.0.0.1:9", "MNX_BEE_KEY": "k'1 x"})
    mnx = tmp_path / "scout" / "work" / ".mnx"
    assert oct((mnx / "env").stat().st_mode)[-3:] == "600"
    # a clean environment (no MNX_*): the helper finds the Hive from the file and tries to reach it
    r = subprocess.run([sys.executable, str(mnx / "mnx-bee"), "me"], env={"PATH": os.environ["PATH"]}, capture_output=True, text=True, timeout=30)
    assert "can't reach the Hive" not in (r.stdout + r.stderr) and r.returncode != 0  # port 9: refused, but it knew where to go
