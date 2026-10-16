"""Your Mnx models: Drive folder → parts → joined, checked files → running as the Hive's brain."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import brain as brain_mod, models as models_mod  # noqa: E402
from hive.models import ModelError, plan_joins, read_md5_lists  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def test_parts_join_in_number_order_and_gaps_are_caught():
    joins, whole = plan_joins(["a.gguf.part10", "a.gguf.part02", "README.txt", "a.gguf.part00", "a.gguf.part01",
                               *[f"a.gguf.part{i:02d}" for i in range(3, 10)]])
    assert joins["a.gguf"][:3] == ["a.gguf.part00", "a.gguf.part01", "a.gguf.part02"] and joins["a.gguf"][-1] == "a.gguf.part10"
    assert whole == ["README.txt"]
    with pytest.raises(ModelError, match="missing part 1"):
        plan_joins(["b.pt.part00", "b.pt.part02"])


def test_md5_lists_in_all_the_formats_your_folders_use(tmp_path):
    (tmp_path / "MD5SUMS_model.txt").write_text("8c2b9665d6a525eb37a4b614028da39e  model-00000.safetensors\n")
    (tmp_path / "manifest_core_only.pt.txt").write_text("a0987809d80c1c11e0461c6505dcf723 95000000 core_only.pt.part000\n")
    (tmp_path / "README.md").write_text("    fb7f2aa8129dc6fd3dc302d63c4a50f9  README.md\nnot a line\n")
    got = read_md5_lists(tmp_path)
    assert got == {"model-00000.safetensors": "8c2b9665d6a525eb37a4b614028da39e", "core_only.pt.part000": "a0987809d80c1c11e0461c6505dcf723",
                   "README.md": "fb7f2aa8129dc6fd3dc302d63c4a50f9"}


def test_drive_links():
    assert models_mod.folder_id("https://drive.google.com/drive/folders/1qZiLMCJWyMytw88ZIO5T2_lCkYLXRZHS?usp=sharing") == "1qZiLMCJWyMytw88ZIO5T2_lCkYLXRZHS"
    with pytest.raises(ModelError):
        models_mod.folder_id("https://example.com/x")


class FakeDrive:
    """A shared Drive folder: a GGUF split into 3 parts, a README, and a sub-folder like the 7B checkpoint."""

    def __init__(self, tmp: Path, corrupt: bool = False):
        self.blob = os.urandom(250_000)
        self.files = {}
        for i in range(3):
            self.files[f"id{i}"] = (f"tiny.gguf.part{i:02d}", self.blob[i * 100_000:(i + 1) * 100_000])
        self.files["idr"] = ("README.txt", b"Tiny model\n")
        side = b"x" * 1000
        self.files["ids0"] = ("table.bin_parts/table.bin.part000", side[:600])
        self.files["ids1"] = ("table.bin_parts/table.bin.part001", side[600:] if not corrupt else b"y" * 400)
        self.side_md5 = hashlib.md5(side).hexdigest()
        self.fetched = []

    def lister(self, fid):
        assert fid == "1qZiLMCJWyMytw88ZIO5T2_lCkYLXRZHS"
        return [(k, v[0]) for k, v in self.files.items()]

    def fetcher(self, file_id, dest, progress):
        self.fetched.append(file_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.files[file_id][1])


@pytest.fixture
def mm(tmp_path, monkeypatch):
    m = models_mod.Models(tmp_path / "models")
    b = brain_mod.Brain(tmp_path / "brain.json")
    monkeypatch.setattr(brain_mod, "brain", b)
    monkeypatch.delenv("MNX_BRAIN_URL", raising=False)
    return m, b


def _tiny(monkeypatch, drive):
    entry = {"id": "tiny", "name": "Tiny", "emoji": "🐜", "about": "test", "kind": "gguf", "run": "llama.cpp", "ram_gb": 1,
             "disk_gb": 0.001, "main": "tiny.gguf", "md5": {"tiny.gguf": hashlib.md5(drive.blob).hexdigest(), "table.bin": drive.side_md5},
             "sizes": {"tiny.gguf": len(drive.blob)}}
    monkeypatch.setattr(models_mod, "CATALOG", [entry])


async def _wait(m, mid):
    await asyncio.wait_for(m.jobs[mid], 10)


def test_download_joins_checks_and_cleans_up(mm, monkeypatch):
    m, _ = mm
    drive = FakeDrive(m.root)
    _tiny(monkeypatch, drive)
    m.lister, m.fetcher = drive.lister, drive.fetcher
    with pytest.raises(ModelError, match="folder link first"):
        run(m.download("tiny") or asyncio.sleep(0))
    m.set_source("tiny", "https://drive.google.com/drive/folders/1qZiLMCJWyMytw88ZIO5T2_lCkYLXRZHS")
    assert oct((m.root / "sources.json").stat().st_mode)[-3:] == "600"

    async def go():
        m.download("tiny")
        await _wait(m, "tiny")
    run(go())
    st = m.state["tiny"]
    assert st["status"] == "ready", st
    files = m.root / "tiny" / "files"
    assert (files / "tiny.gguf").read_bytes() == drive.blob and st["path"].endswith("tiny.gguf")
    assert (files / "table.bin").stat().st_size == 1000  # table.bin_parts/… joins next to the others
    assert not (m.root / "tiny" / "download").exists()  # parts removed once checked
    assert m.view()[0]["status"] == "ready" and "has_link" not in m.view()[0] and m.view(owner=True)[0]["has_link"]


def test_a_corrupt_part_fails_the_checksum(mm, monkeypatch):
    m, _ = mm
    drive = FakeDrive(m.root, corrupt=True)
    _tiny(monkeypatch, drive)
    m.lister, m.fetcher = drive.lister, drive.fetcher
    m.set_source("tiny", "1qZiLMCJWyMytw88ZIO5T2_lCkYLXRZHS")

    async def go():
        m.download("tiny")
        await _wait(m, "tiny")
    run(go())
    assert m.state["tiny"]["status"] == "failed" and "table.bin" in m.state["tiny"]["error"]


def test_a_rerun_skips_parts_already_fetched(mm, monkeypatch):
    m, _ = mm
    drive = FakeDrive(m.root)
    _tiny(monkeypatch, drive)
    calls = {"n": 0}

    def flaky(file_id, dest, progress):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ModelError("network dropped")
        drive.fetcher(file_id, dest, progress)
    m.lister, m.fetcher = drive.lister, flaky
    m.set_source("tiny", "1qZiLMCJWyMytw88ZIO5T2_lCkYLXRZHS")

    async def go():
        m.download("tiny")
        await _wait(m, "tiny")
    run(go())
    assert m.state["tiny"]["status"] == "failed"
    before = len(drive.fetched)
    m.fetcher = drive.fetcher
    run(go())
    assert m.state["tiny"]["status"] == "ready" and len(drive.fetched) - before == len(drive.files) - 2


FAKE_SERVER = r'''#!{python}
# Stands in for llama.cpp's llama-server: /health and /v1/chat/completions.
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1]); model = sys.argv[sys.argv.index("-m") + 1]
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": "from " + model.split("/")[-1] + ": " + body["messages"][-1]["content"]}}]}).encode()
        self.send_response(200); self.send_header("content-type", "application/json"); self.end_headers(); self.wfile.write(out)
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def test_start_runs_llama_server_and_makes_it_the_brain(mm, monkeypatch, tmp_path):
    m, b = mm
    drive = FakeDrive(m.root)
    _tiny(monkeypatch, drive)
    m.lister, m.fetcher = drive.lister, drive.fetcher
    m.set_source("tiny", "1qZiLMCJWyMytw88ZIO5T2_lCkYLXRZHS")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "llama-server"
    exe.write_text(FAKE_SERVER.replace("{python}", sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setattr(models_mod, "PORT", 18631)

    async def go():
        m.download("tiny")
        await _wait(m, "tiny")
        await m.start("tiny")
        assert b.url == "http://127.0.0.1:18631" and b.view()["model"] == "tiny"
        reply = await brain_mod.chat([{"role": "user", "content": "hello"}])
        v = m.view()[0]
        await m.stop("tiny")
        return reply, v
    reply, v = run(go())
    assert reply == "from tiny.gguf: hello" and v["status"] == "running" and v["in_use"]
    assert m.state["tiny"]["status"] == "ready" and not b.url  # stopping it unplugs the brain


def test_without_llama_cpp_it_says_how_to_get_it(mm, monkeypatch):
    m, _ = mm
    monkeypatch.setattr(models_mod.shutil, "which", lambda name: None)
    with pytest.raises(ModelError, match="pkg install llama-cpp"):
        m._server_argv("/x/tiny.gguf")


def test_other_formats_need_their_own_start_command(mm):
    m, _ = mm
    with pytest.raises(ModelError, match="start command"):
        run(m.start("mnx-lonjer"))
    m.set_command("mnx-lonjer", "./run_mnx1.sh", 8000)
    assert m.view(owner=True)[3]["command"] == "./run_mnx1.sh" and m.view()[3]["can_run"]


def test_catalog_matches_your_readmes():
    c = {x["id"]: x for x in models_mod.CATALOG}
    assert c["mnx-1.5-max-coder"]["md5"]["mnx-1.5-max-coder-q4_k_m.gguf"] == "b3e450e066fae5e18d2738344b3a6ef8"
    assert c["mnx-1.5-max-coder"]["sizes"]["mnx-1.5-max-coder-q4_k_m.gguf"] == 986048128
    assert c["mnx-1-lite"]["md5"]["mnx_lite1.pt"] == "9d0e6c88eaf867bc499524e6e7b697e6"
    assert c["mnx-1"]["md5"]["table.bin"] == "9639b782ab2649bd178fe3152b500f61"
    assert "drive.google.com" not in json.dumps(models_mod.CATALOG)  # links live in data/, never in the public repo


def test_models_api_is_owner_only_and_never_shows_links(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import gateway.api as gw
    from gateway import auth
    m = models_mod.Models(tmp_path / "models")
    monkeypatch.setattr(models_mod, "models", m)
    monkeypatch.setattr(auth, "USER_TOKENS_FILE", tmp_path / "u.txt")
    (tmp_path / "u.txt").write_text("user-tok\n")
    client = TestClient(gw.app)
    owner, user = {"Authorization": f"Bearer {auth.TOKEN}"}, {"Authorization": "Bearer user-tok"}
    link = "https://drive.google.com/drive/folders/1qZiLMCJWyMytw88ZIO5T2_lCkYLXRZHS"
    assert client.put("/api/models/mnx-1.5-max-coder/source", headers=user, json={"link": link}).status_code == 403
    assert client.put("/api/models/mnx-1.5-max-coder/source", headers=owner, json={"link": link}).status_code == 200
    assert client.post("/api/models/mnx-1.5-max-coder/download", headers=user).status_code == 403
    assert client.get("/api/models", headers=user).json() == []  # nothing set up yet: users see nothing to pick
    assert client.get("/api/models?setup=1", headers=user).json() == []  # ?setup is the owner's view only
    full = client.get("/api/models?setup=1", headers=owner).json()
    assert [x["name"] for x in full] == ["Mnx 1.5 Max Coder", "Mnx 1 Lite", "Mnx 1 (7B)", "Mnx lonjer"] and full[0]["has_link"]
    assert "emoji" not in full[0]


# ---------------------------------------------------------------- users just pick a model by name
def _two_models(monkeypatch, tmp_path, m, drive=None):
    """Two fake models (different fake servers) that the owner has already set up."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    exe = bindir / "llama-server"
    exe.write_text(FAKE_SERVER.replace("{python}", sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    ports = {"alpha": 18641, "beta": 18642}
    cat = []
    for i, (mid, name) in enumerate((("alpha", "Alpha Chat"), ("beta", "Beta Coder"))):
        f = tmp_path / f"{mid}.gguf"
        f.write_bytes(b"x")
        cat.append({"id": mid, "name": name, "about": "t", "kind": "gguf", "run": "llama.cpp", "ram_gb": 1, "disk_gb": 0.001,
                    "main": f.name, "md5": {}, "sizes": {}})
        m.state[mid] = {"status": "ready", "path": str(f)}
    monkeypatch.setattr(models_mod, "CATALOG", cat)
    monkeypatch.setattr(models_mod, "PORT", 18641)  # alpha 18641, beta 18642: one port per model
    return ports


def test_users_see_only_names_and_each_message_goes_to_the_model_picked(mm, monkeypatch, tmp_path):
    m, b = mm
    monkeypatch.setattr(models_mod, "models", m)
    _two_models(monkeypatch, tmp_path, m)
    assert m.choices() == [{"id": "alpha", "name": "Alpha Chat"}, {"id": "beta", "name": "Beta Coder"}]  # names only, no more
    assert brain_mod.is_connected()  # a model set up is a brain: no separate Brain setting needed

    async def ask(model):
        tok = models_mod.selected.set(model)
        try:
            return await brain_mod.chat([{"role": "user", "content": "hi"}])
        finally:
            models_mod.selected.reset(tok)

    async def go():
        a = await ask("alpha")        # starts alpha by itself: no download, no click on "start"
        c = await ask("beta")         # the user switches: beta starts, alpha stops
        again = await ask(None)       # nothing picked: whichever is running
        await m.stop_all()
        return a, c, again
    a, c, again = run(go())
    assert a == "from alpha.gguf: hi" and c == "from beta.gguf: hi" and again == "from beta.gguf: hi"
    assert not b.url  # picking models never touched the Settings brain


def test_an_answer_in_progress_isnt_cut_off_by_another_users_pick(mm, monkeypatch, tmp_path):
    m, _ = mm
    monkeypatch.setattr(models_mod, "models", m)
    _two_models(monkeypatch, tmp_path, m)
    order = []

    async def slow_alpha():
        async with m.using("alpha") as url:
            order.append("alpha-start")
            await asyncio.sleep(1.2)
            order.append("alpha-done")

    async def beta_wants_in():
        await asyncio.sleep(0.4)
        async with m.using("beta") as url:
            order.append("beta-start")

    async def go():
        await asyncio.gather(slow_alpha(), beta_wants_in())
        await m.stop_all()
    run(go())
    assert order == ["alpha-start", "alpha-done", "beta-start"]


def test_a_model_the_owner_hasnt_set_up_cant_be_used(mm, monkeypatch, tmp_path):
    m, _ = mm
    _two_models(monkeypatch, tmp_path, m)
    m.state.pop("beta")
    assert [c["id"] for c in m.choices()] == ["alpha"] and not m.usable("beta") and not m.usable("nope")
    with pytest.raises(ModelError, match="isn't available"):
        async def go():
            async with m.using("beta"):
                pass
        run(go())


# ---------------------------------------------------------------- the owner installs from the command line; users just pick
def test_one_command_installs_models_and_a_running_server_sees_them(tmp_path, monkeypatch, capsys):
    from hive import admin
    drive = FakeDrive(tmp_path)
    _tiny(monkeypatch, drive)
    cli = models_mod.Models(tmp_path / "shared")        # the command line
    server = models_mod.Models(tmp_path / "shared")     # the server that was already running
    cli.lister, cli.fetcher = drive.lister, drive.fetcher
    monkeypatch.setattr(models_mod, "models", cli)
    assert server.choices() == []                       # before: nothing for users to pick
    assert admin.main(["install-models", "tiny=https://drive.google.com/drive/folders/1qZiLMCJWyMytw88ZIO5T2_lCkYLXRZHS"]) == 0
    out = capsys.readouterr().out
    assert "ready: users can now pick it by name" in out and "checking MD5s" in out
    assert server.choices() == [{"id": "tiny", "name": "Tiny"}]  # no restart, no clicking: it just appears
    assert admin.main(["install-models", "tiny=not-a-link"]) == 1


def test_a_failed_install_says_so_and_nothing_appears(tmp_path, monkeypatch, capsys):
    from hive import admin
    drive = FakeDrive(tmp_path, corrupt=True)
    _tiny(monkeypatch, drive)
    cli = models_mod.Models(tmp_path / "shared")
    cli.lister, cli.fetcher = drive.lister, drive.fetcher
    monkeypatch.setattr(models_mod, "models", cli)
    assert admin.main(["install-models", "tiny=1qZiLMCJWyMytw88ZIO5T2_lCkYLXRZHS"]) == 1
    assert "Checksum mismatch" in capsys.readouterr().out and cli.choices() == []
