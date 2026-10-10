"""Vortex OS on the Bees' computers: accounts, Docker hardening, tunnels, the OS choice, and a real desktop."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import bees, cells as cells_mod, vortex  # noqa: E402
from hive.bees import BeeError, normalize_computer  # noqa: E402

VORTEX_CHECKOUT = Path(os.getenv("MNX_TEST_VORTEX_DIR", "/home/user/tharunmaks/vortex-os-"))


@pytest.mark.parametrize("bee_id,user", [("scout", "scout"), ("price_alert", "price-alert"), ("ai", "bee-ai"), ("9lives", "bee-9lives"),
                                         ("x" * 40, ("bee-" + "x" * 40)[:32])])
def test_vortex_user_names_follow_its_rules(bee_id, user):
    got = vortex.username_for(bee_id)
    assert got == user and vortex.USER_RE.match(got)


def test_passwords_pass_vortex_strength_rules():
    for name in ("scout", "a" * 3, "bee-ai"):
        for _ in range(50):
            pw = vortex.new_password(name)
            classes = sum(bool(re.search(p, pw)) for p in (r"[a-z]", r"[A-Z]", r"[0-9]", r"[^A-Za-z0-9]"))
            assert len(pw) >= 10 and classes >= 3 and name not in pw.lower()


def test_credentials_are_private_and_ports_unique(tmp_path):
    c = vortex.Creds(tmp_path / "vortex.json")
    taken = {vortex.PORT_BASE + 1}
    a = c.get("scout", check_free=lambda p: p not in taken)
    b = c.get("coding", check_free=lambda p: p not in taken)
    assert a["port"] == vortex.PORT_BASE and b["port"] == vortex.PORT_BASE + 2  # skips a busy port and the other Bee's
    assert c.get("scout") == a  # stable
    assert oct((tmp_path / "vortex.json").stat().st_mode)[-3:] == "600"
    assert vortex.Creds(tmp_path / "vortex.json").data == c.data
    c.forget("scout")
    assert "scout" not in vortex.Creds(tmp_path / "vortex.json").data


def test_docker_computer_is_vortex_and_hardened(tmp_path):
    rec = {"user": "scout", "password": "Pw-123456789abcX", "port": 8425}
    argv = vortex.docker_run_args("mnx-cell-scout", "scout", "Scout", tmp_path / "work", tmp_path / "vortex", ["--cpus", "2"], rec, "img:1")
    s = " ".join(argv)
    for flag in ("--cap-drop ALL", "--read-only", "--security-opt no-new-privileges", "--pids-limit 512", "--cpus 2",
                 "--label mnx.cell=scout", "-p 127.0.0.1:8425:8420", "VORTEX_ADMIN_USER=scout", "VORTEX_HOMES_DIR=/homes"):
        assert flag in s, flag
    assert f"{(tmp_path / 'work').resolve()}:/homes/scout" in s and f"{(tmp_path / 'work').resolve()}:/work" in s
    assert argv[-1] == "img:1" and "sleep" not in argv  # Vortex is the container's main process


def test_tunnel_reuses_the_labs_ssh_options():
    ssh = ["ssh", "-i", "/k", "-p", "2222", "-o", "BatchMode=yes", "root@1.2.3.4", "--"]
    t = vortex.tunnel_argv(ssh, 8430, 8421, "127.0.0.1")
    assert t[:7] == ["ssh", "-i", "/k", "-p", "2222", "-o", "BatchMode=yes"] and t[-1] == "root@1.2.3.4"
    assert "-N" in t and "127.0.0.1:8430:127.0.0.1:8421" in t and "--" not in t


def test_remote_script_prefers_docker_then_node_and_quotes_secrets():
    rec = {"user": "scout", "password": "a'b; rm -rf ~ X1", "port": 8421}
    sc = vortex.remote_script(rec, "Scout", "'/w/cells/scout'", "scout")
    assert sc.index("command -v docker") < sc.index("command -v node")
    assert "'a'\"'\"'b; rm -rf ~ X1'" in sc  # the password is one shell word, never code
    assert "MNX_VORTEX_NEEDS: Docker or Node.js" in sc and "127.0.0.1:8421:8420" in sc


def test_os_choice():
    assert normalize_computer(None)["os"] == "vortex"
    assert normalize_computer({"os": "plain"})["os"] == "plain"
    assert normalize_computer({"os": None}, {"size": "small", "cpus": 1, "memory_gb": 1, "gpu": False, "where": "here"})["os"] == "vortex"
    with pytest.raises(BeeError):
        normalize_computer({"os": "windows"})
    assert "Vortex OS" in bees.describe_computer(normalize_computer(None))


def test_a_plain_computer_has_no_desktop(tmp_path, monkeypatch):
    monkeypatch.setattr(cells_mod, "CELLS", tmp_path)
    cell = cells_mod.Cell("shell", "Shell", "local", {"size": "small", "os": "plain"})
    st = vortex.Desktops().status(cell)
    assert st["status"] == "off" and st["os"] == "plain"
    with pytest.raises(RuntimeError, match="plain Linux"):
        asyncio.run(vortex.Desktops().start(cell))


def test_failing_vortex_image_falls_back_to_a_plain_computer(tmp_path, monkeypatch):
    monkeypatch.setattr(cells_mod, "CELLS", tmp_path)
    calls = []

    async def fake_run(argv, timeout=60):
        calls.append(argv)
        return (1, "no such container") if argv[1] == "inspect" else (0, "ok")

    async def no_image():
        raise RuntimeError("build failed: no network")

    monkeypatch.setattr(cells_mod, "_run", fake_run)
    monkeypatch.setattr(vortex, "ensure_image", no_image)
    cell = cells_mod.Cell("scout", "Scout", "docker", normalize_computer(None))
    asyncio.run(cell.ensure_up())
    run_argv = next(a for a in calls if a[:2] == ["docker", "run"])
    assert run_argv[-3:] == [cells_mod.IMAGE, "sleep", "infinity"] and "build failed" in cell.os_error
    assert cell.view()["os_error"].startswith("Vortex OS isn't available")


# ---------------------------------------------------------------- a real Vortex OS desktop for a Bee
needs_vortex = pytest.mark.skipif(not (VORTEX_CHECKOUT / "node_modules" / "node-pty").exists() or not shutil.which("node"),
                                  reason="needs a Vortex OS checkout with its packages installed (MNX_TEST_VORTEX_DIR) and Node.js")


@needs_vortex
def test_a_bee_gets_a_real_vortex_desktop_on_its_own_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cells_mod, "CELLS", tmp_path / "cells")
    monkeypatch.setattr(vortex, "DIR", VORTEX_CHECKOUT)
    monkeypatch.setattr(vortex, "creds", vortex.Creds(tmp_path / "vortex.json"))
    monkeypatch.setattr(vortex, "PORT_BASE", 18421)
    cell = cells_mod.Cell("scout", "Scout", "local", normalize_computer(None))
    (cell.work / "prices.csv").write_text("item,price\ntea,10\n")
    d = vortex.Desktops()
    try:
        st = asyncio.run(d.start(cell))
        assert st["status"] == "running" and st["user"] == "scout"
        base = f"http://127.0.0.1:{st['port']}"
        assert json.loads(urllib.request.urlopen(base + "/api/hive", timeout=5).read())["name"] == "Scout Bee · MNX Hive"
        jar = CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        req = urllib.request.Request(base + "/api/auth/login", method="POST",
                                     data=json.dumps({"username": st["user"], "password": st["password"]}).encode(),
                                     headers={"Content-Type": "application/json", "Origin": base})
        me = json.loads(opener.open(req, timeout=10).read())
        assert me.get("user", me).get("role") == "queen" and not me.get("user", me).get("mustChangePassword")
        listing = json.loads(opener.open(base + "/api/fs/list?path=/", timeout=10).read())
        names = [e["name"] for e in (listing.get("entries") if isinstance(listing, dict) else listing)]
        assert "prices.csv" in names  # the desktop's home is the Bee's own work folder
        bad = urllib.request.Request(base + "/api/auth/login", method="POST", data=json.dumps({"username": "scout", "password": "wrong-Pass-1"}).encode(),
                                     headers={"Content-Type": "application/json", "Origin": base})
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad, timeout=10)
    finally:
        asyncio.run(d.stop(cell))
    assert not vortex.port_open(st["port"])
