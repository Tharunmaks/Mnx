"""Bees you make yourself (Grok-style bots), the brain they talk with, the Pro plan, the Hive's GPU pool and its free cloud."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import bots, brain as brain_mod, cloud, gpu_pool, plan as plan_mod, plans  # noqa: E402
from hive.bees import Registry  # noqa: E402


def run(coro):
    return asyncio.run(coro)


U = "u_test0000001"  # a user (the owner is always on Pro)


# ---------------------------------------------------------------- the brain
class FakeModel:
    def __init__(self):
        self.requests = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        self.requests.append((str(req.url), req.headers.get("authorization"), body))
        system = body["messages"][0]["content"]
        last = body["messages"][-1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": f"[{len(body['messages'])} msgs] {last} | {system[:40]}"}}]})


@pytest.fixture
def model(tmp_path, monkeypatch):
    for k in ("MNX_BRAIN_URL", "MNX_BRAIN_KEY", "MNX_BRAIN_MODEL"):
        monkeypatch.delenv(k, raising=False)
    b = brain_mod.Brain(tmp_path / "brain.json")
    fake = FakeModel()
    b.transport = httpx.MockTransport(fake.handler)
    monkeypatch.setattr(brain_mod, "brain", b)
    return b, fake


def test_brain_speaks_openai_chat_to_any_server(model, tmp_path):
    b, fake = model
    assert not brain_mod.is_connected()
    with pytest.raises(brain_mod.BrainNotConnected):
        run(b.chat([{"role": "user", "content": "hi"}]))
    b.update({"url": "https://api.groq.com/openai", "key": "gsk-secret", "model": "llama-3.1-8b-instant"})
    assert brain_mod.is_connected() and "gsk-secret" not in json.dumps(b.view()) and b.view()["key_set"]
    assert oct((tmp_path / "brain.json").stat().st_mode)[-3:] == "600"
    out = run(brain_mod.ask("hello"))
    url, auth, body = fake.requests[-1]
    assert url == "https://api.groq.com/openai/v1/chat/completions" and auth == "Bearer gsk-secret"
    assert body["model"] == "llama-3.1-8b-instant" and out.startswith("[2 msgs] hello")
    for u, want in [("http://127.0.0.1:11434", "http://127.0.0.1:11434/v1/chat/completions"),
                    ("http://x:8080/v1", "http://x:8080/v1/chat/completions"),
                    ("https://generativelanguage.googleapis.com/v1beta/openai", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")]:
        b.update({"url": u})
        assert b.endpoint() == want
    with pytest.raises(ValueError):
        b.update({"url": "ftp://nope"})


# ---------------------------------------------------------------- bots
@pytest.fixture
def hive(tmp_path, monkeypatch):
    from hive import bees, modes, queen, bee_chat, calls, gpu_chat
    r = Registry(tmp_path / "hive.json", tmp_path / "keys.json")

    async def nothing(*a):
        return None
    r.on_created = r.on_deleted = r.on_computer = nothing
    monkeypatch.setattr(bees, "registry", r)
    for mod in (modes, queen, bee_chat, calls, plans, gpu_chat):
        monkeypatch.setattr(mod, "registry", r)
    monkeypatch.setattr(bots, "store", bots.Store(tmp_path / "bots"))
    monkeypatch.delenv("MNX_PLAN", raising=False)
    monkeypatch.setattr(plan_mod.plan, "path", tmp_path / "plan.json")
    return r


def test_make_a_bot_with_personality_knowledge_and_memory(hive):
    b = run(bots.create_bot("Maya", "a friendly Tamil tutor", {"emoji": "🌸", "personality": "Teach Tamil gently.", "tone": "caring",
                                                               "greeting": "Vanakkam!", "starters": ["Teach me numbers", "", "Quiz me"],
                                                               "visibility": "hive"},
                            knowledge={"numbers.md": "Onru means one.\n\nIrandu means two.\n\nMoondru means three."}, owner=U))
    assert b["id"] == "maya" and b["bot"]["starters"] == ["Teach me numbers", "Quiz me"] and b["tools"] == ["bot"]
    assert b["computer"]["size"] == "free" and b["owner"] == U  # Free plan: the Free computer, like every Bee of theirs
    hits = bots.store.search("maya", "what does irandu mean")
    assert hits[0]["file"] == "numbers.md" and "Irandu means two" in hits[0]["text"]
    bots.store.remember("maya", "I live in Madurai.")
    bots.store.remember("maya", "i live in madurai")  # the same fact once
    assert [m["text"] for m in bots.store.memories("maya")] == ["i live in madurai"]
    assert bots.store.forget("maya", "madurai") == 1 and bots.store.memories("maya") == []
    with pytest.raises(bots.BotError):
        bots.clean_profile({"tone": "evil"})
    with pytest.raises(bots.BotError):
        bots.store.add_file("maya", "virus.exe", b"MZ")
    prompt = bots.system_prompt(b, hits, ["likes filter coffee"], "Tharun")
    assert "Teach Tamil gently." in prompt and "gentle, patient and caring" in prompt and "likes filter coffee" in prompt
    assert "[numbers.md]" in prompt and "Tharun" in prompt
    exported = bots.export(b)
    assert exported["mnx_bot"] == 1 and exported["knowledge"]["numbers.md"].startswith("Onru")


def test_bee_phrases():
    assert bots.parse("make a bot called Maya that is a friendly Tamil tutor") == \
        {"name": "Maya", "description": "a friendly Tamil tutor", "parent": None}
    assert bots.parse("create my own companion") == {"name": "", "description": "", "parent": None}  # no premade ones: you shape it
    assert bots.parse("build me a custom gpt for interview practice") == {"name": "", "description": "interview practice", "parent": None}
    assert bots.parse("create a bee called Scout that watches prices") == {"name": "Scout", "description": "watches prices", "parent": None}
    assert bots.parse("train my model") is None
    assert not hasattr(bots, "TEMPLATES")


class Collect:
    def __init__(self):
        self.events = []

    async def __call__(self, ev):
        self.events.append(ev)

    def answer(self):
        return next(e["text"] for e in self.events if e["type"] == "answer")


def _task(text, bee, history=None):
    from hive.core import Task
    c = Collect()
    t = Task("t1", "c1", text, c, bee=bee, mode="peace", user=U)
    t.history, t.user_name = history or [], "Tharun"
    return t, c


def test_a_bot_without_a_brain_remembers_and_answers_from_its_knowledge(hive, model):
    from hive import queen
    run(bots.create_bot("Maya", "Tamil tutor", {"greeting": "Vanakkam!"}, knowledge={"n.md": "Onru means one. Irandu means two. Moondru means three."},
                        owner=U))
    t, c = _task("remember that I am learning for my trip", "maya")
    run(queen.handle(t))
    assert "I'll remember" in c.answer() and bots.store.memories("maya")[0]["text"] == "I am learning for my trip"
    t, c = _task("what does irandu mean?", "maya")
    run(queen.handle(t))
    assert c.answer() == "From n.md: Irandu means two." and any(e.get("badge") == "Knowledge" for e in c.events)
    t, c = _task("tell me a joke", "maya")
    run(queen.handle(t))
    assert c.answer().startswith("Vanakkam!") and "No model is running" in c.answer()
    assert {e["bee"] for e in c.events if e.get("bee")} == {"Maya"}


def test_a_bot_with_a_brain_talks_in_character_with_the_chat_so_far(hive, model):
    from hive import queen
    b, fake = model
    b.update({"url": "http://127.0.0.1:11434"})
    run(bots.create_bot("Maya", "Tamil tutor", {"personality": "You are Maya, a Tamil tutor."}, owner=U))
    bots.store.remember("maya", "I love filter coffee")
    t, c = _task("how do I say hello?", "maya", [{"role": "user", "text": "hi"}, {"role": "bot", "text": "Vanakkam!"}])
    run(queen.handle(t))
    body = fake.requests[-1][2]
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user", "assistant", "user"] and "You are Maya, a Tamil tutor." in body["messages"][0]["content"]
    assert "I love filter coffee" in body["messages"][0]["content"] and c.answer().startswith("[4 msgs] how do I say hello?")


def test_chat_only_bots_dont_run_commands(hive, model):
    from hive import queen
    run(bots.create_bot("Poet", "writes poems", {"computer": False}, owner=U))
    t, c = _task("$ rm -rf /", "poet")
    run(queen.handle(t))
    assert "chat-only" in c.answer()


def test_another_users_bee_never_answers_you(hive, model):
    from hive import queen
    run(bots.create_bot("Maya", "Tamil tutor", {}, owner="u_other000001"))
    t, c = _task("$ cat ~/.ssh/id_rsa", "maya")
    run(queen.handle(t))
    assert "isn't one of yours" in c.answer() and not any(e["type"] == "output" for e in c.events)


# ---------------------------------------------------------------- Pro (paid with Razorpay)
def test_pro_gives_16gb_100gb_and_a_16gb_gpu_and_ends_on_its_date(hive):
    run(hive.create("Scout", "watches", owner=U))
    plan_mod.plan.grant(U, 30, {"id": "pay_x"})
    changed = run(plans.apply(U))
    s = hive.get("scout")["computer"]
    assert (s["size"], s["memory_gb"], s["disk_gb"], s["gpu"]) == ("pro", 16, 100, True) and changed == ["Scout"]
    plan_mod.plan.revoke(U)
    run(plans.apply(U))
    assert hive.get("scout")["computer"]["size"] == "free" and not hive.get("scout")["computer"]["gpu"]


@pytest.mark.parametrize("text,want", [("subscribe me to pro", "up"), ("upgrade to the pro plan", "up"), ("get plus", "up"),
                                       ("upgrade", "up"), ("switch back to free", "down"), ("cancel my subscription", "down"),
                                       ("what plan am I on", "which"), ("upgrade the coding bee to pro", None), ("what is 2 plus 2", None)])
def test_plan_phrases(text, want):
    got = plans.parse(text)
    assert (got or {}).get("kind") == want


def test_upgrading_in_the_chat_points_to_the_pro_page(hive):
    from hive import queen
    t, c = _task("upgrade to pro", None)
    t.mode = "rush"
    run(queen.handle(t))
    assert "pro.html" in c.answer() and "499" in json.dumps(c.events)


# ---------------------------------------------------------------- the Hive's GPU (Kaggle, with a stand-in CLI)
FAKE_KAGGLE = r'''#!{python}
# Stands in for the kaggle CLI: push really runs the job script (with /kaggle/working moved to a temp folder).
import json, os, subprocess, sys
state = "{state}"
a = sys.argv[1:]
open(os.path.join(state, "calls.txt"), "a").write(" ".join(a) + "\n")
assert os.environ.get("KAGGLE_USERNAME") == "tharun" and (os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY"))
if a[:2] == ["kernels", "list"]:
    print("ref title"); sys.exit(0)
if a[:2] == ["kernels", "push"]:
    folder = a[3]
    meta = json.load(open(os.path.join(folder, "kernel-metadata.json")))
    assert meta["enable_gpu"] and meta["is_private"] and meta["code_file"] == "run.py"
    work = os.path.join(state, "working")
    os.makedirs(work, exist_ok=True)
    script = open(os.path.join(folder, "run.py")).read().replace("/kaggle/working", work)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=work)
    json.dump({"id": meta["id"], "log": out.stdout + out.stderr}, open(os.path.join(state, "job.json"), "w"))
    print("Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/code/" + meta["id"]); sys.exit(0)
if a[:2] == ["kernels", "status"]:
    print('%s has status "KernelWorkerStatus.COMPLETE"' % a[2]); sys.exit(0)
if a[:2] == ["kernels", "output"]:
    job = json.load(open(os.path.join(state, "job.json")))
    out = a[a.index("-p") + 1]
    work = os.path.join(state, "working")
    for root, _, files in os.walk(work):
        for f in files:
            src = os.path.join(root, f); dst = os.path.join(out, os.path.relpath(src, work))
            os.makedirs(os.path.dirname(dst), exist_ok=True); open(dst, "wb").write(open(src, "rb").read())
    json.dump([{"stream_name": "stdout", "data": job["log"]}], open(os.path.join(out, job["id"].split("/")[1] + ".log"), "w"))
    sys.exit(0)
sys.exit(2)
'''


@pytest.fixture
def gpu(tmp_path, monkeypatch, hive):
    fake = tmp_path / "kaggle"
    state = tmp_path / "kstate"
    state.mkdir()
    fake.write_text(FAKE_KAGGLE.replace("{python}", sys.executable).replace("{state}", str(state)))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(gpu_pool, "KAGGLE_BIN", str(fake))
    p = gpu_pool.GpuPool(tmp_path / "gpu")
    p.poll = 0.01
    secrets = {}

    class Lab:
        _secrets = secrets
        gpu_local = False
    monkeypatch.setattr(p, "_secrets", lambda: secrets)
    import hive.lab as lab_mod
    monkeypatch.setattr(lab_mod, "lab", Lab)
    return p, secrets, state


def test_the_gpu_needs_pro_and_the_hives_kaggle(gpu, hive):
    p, secrets, _ = gpu
    run(hive.create("Scout", "watches", owner=U))
    with pytest.raises(gpu_pool.GpuError, match="Pro"):
        run(p.submit("scout", "python a.py", {}))
    with pytest.raises(gpu_pool.GpuError, match="No such Bee"):
        run(p.submit("ghost", "python a.py", {}))
    plan_mod.plan.grant(U, 30, {"id": "pay_g"})
    with pytest.raises(gpu_pool.GpuError, match="owner adds it"):
        run(p.submit("scout", "python a.py", {}))
    secrets.update(KAGGLE_USERNAME="tharun", KAGGLE_API_TOKEN="KGAT_x")
    assert p.kaggle_ready() and p.source() == "kaggle"
    with pytest.raises(gpu_pool.GpuError, match="Bad file name"):
        run(p.submit("scout", "ls", {"../etc/passwd": b"x"}))


def test_a_gpu_job_runs_the_bees_files_and_brings_the_outputs_back(gpu, hive):
    p, secrets, state = gpu
    run(hive.create("Scout", "watches"))  # the owner's Bee: the owner is on Pro
    secrets.update(KAGGLE_USERNAME="tharun", KAGGLE_API_TOKEN="KGAT_x")
    delivered = {}

    async def deliver(bee, rel, data):
        delivered[rel] = data
    p.deliver = deliver
    train = textwrap.dedent("""
        import csv
        rows = list(csv.reader(open("data.csv")))
        open("model.txt", "w").write(f"trained on {len(rows)} rows")
        print("loss 0.12")
    """).encode()

    async def scenario():
        job = await p.submit("scout", "python train.py", {"train.py": train, "data.csv": b"a,b\n1,2\n3,4\n"})
        assert job["status"] == "running" and job["kernel"].startswith("tharun/mnx-scout-")
        await asyncio.wait_for(p._watch[job["id"]], 10)
        return p.jobs[job["id"]]
    job = run(scenario())
    assert job["status"] == "complete" and job["exit"] == 0 and "loss 0.12" in job["log"]
    assert delivered[f"gpu-out/{job['id']}/model.txt"] == b"trained on 3 rows"
    assert "model.txt" in job["outputs"] and p.output_path(job["id"], "model.txt").read_text() == "trained on 3 rows"
    assert p.hours_this_week() >= 0 and json.loads((p.root / "jobs.json").read_text())[job["id"]]["status"] == "complete"
    calls = (state / "calls.txt").read_text()
    assert "kernels push" in calls and "kernels status" in calls and "kernels output" in calls


def test_a_failing_command_is_reported(gpu, hive):
    p, secrets, _ = gpu
    run(hive.create("Scout", "watches"))  # the owner's Bee: the owner is on Pro
    secrets.update(KAGGLE_USERNAME="tharun", KAGGLE_KEY="0" * 32)

    async def scenario():
        job = await p.submit("scout", "python -c 'raise SystemExit(3)'", {})
        await asyncio.wait_for(p._watch[job["id"]], 10)
        return p.jobs[job["id"]]
    job = run(scenario())
    assert job["status"] == "error" and job["exit"] == 3 and "exited with 3" in job["error"]


def test_weekly_free_hours_are_respected(gpu, hive):
    p, secrets, _ = gpu
    run(hive.create("Scout", "watches"))  # the owner's Bee: the owner is on Pro
    secrets.update(KAGGLE_USERNAME="tharun", KAGGLE_API_TOKEN="KGAT_x")
    import time
    now = time.time()
    p.jobs["old"] = {"id": "old", "source": "kaggle", "started": now - 31 * 3600, "ended": now - 60, "created": now - 31 * 3600}
    assert p.hours_this_week(now) > 30
    with pytest.raises(gpu_pool.GpuError, match="hours this week"):
        run(p.submit("scout", "ls", {}))


@pytest.mark.parametrize("text,command,files", [
    ("run train.py on the gpu", "python train.py", ["train.py"]),
    ("run train.py data.csv on the free gpu", "python train.py data.csv", ["train.py", "data.csv"]),
    ("run nvidia-smi on the gpu", "nvidia-smi", []),
])
def test_gpu_phrases(text, command, files):
    assert gpu_pool.parse(text) == {"kind": "run", "command": command, "files": files}


# ---------------------------------------------------------------- the Hive's free cloud (Oracle Always Free etc.)
def test_plus_bees_go_to_the_hives_free_cloud_when_it_has_room(hive, monkeypatch):
    targets = {"oracle": {"id": "oracle", "name": "Hive cloud · 1.2.3.4", "pool": True, "status": "ok", "info": {"ram_mb": "24000"}}}

    class Lab:
        pass
    Lab.targets = targets
    import hive.lab as lab_mod
    monkeypatch.setattr(lab_mod, "lab", Lab)
    assert cloud.slots(targets["oracle"]) == 2  # 24 GB × 2 overcommit / 16 GB
    a = run(hive.create("Alpha", "x"))  # the owner's Bees: Pro computers
    b = run(hive.create("Beta", "y"))
    c = run(hive.create("Gamma", "z"))
    assert a["computer"]["where"] == "oracle" and b["computer"]["where"] == "oracle" and c["computer"]["where"] == "here"
    assert [x["bees"] for x in cloud.pool_view()] == [["Alpha", "Beta"]]


# ---------------------------------------------------------------- through the gateway
@pytest.fixture
def gw(hive, model, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import gateway.api as gw_mod
    from gateway import auth
    from hive import cells as cells_mod
    monkeypatch.setattr(gw_mod, "registry", hive)
    monkeypatch.setattr(gw_mod, "BEES", hive.bees)
    monkeypatch.setattr(cells_mod, "CELLS", tmp_path / "cells")
    monkeypatch.setattr(cells_mod.cells, "cells", {})
    return TestClient(gw_mod.app), {"Authorization": f"Bearer {auth.TOKEN}"}


def test_bots_api_make_teach_share_and_import(gw):
    client, auth = gw
    assert client.post("/api/bots", json={"name": "Maya"}).status_code == 401
    made = client.post("/api/bots", headers=auth, json={"name": "Maya", "tagline": "Tamil tutor",
                                                         "profile": {"emoji": "🌸", "personality": "Teach Tamil.", "starters": ["Quiz me"]}})
    assert made.status_code == 200 and made.json()["knowledge"] == [] and "owner" not in made.json()
    up = client.post("/api/bots/maya/knowledge?name=words.md", headers=auth, content="Nandri means thank you.".encode())
    assert up.status_code == 200 and up.json()["name"] == "words.md"
    assert client.post("/api/bots/maya/knowledge?name=a.exe", headers=auth, content=b"MZ").status_code == 400
    edited = client.put("/api/bots/maya", headers=auth, json={"tagline": "Tamil and Hindi tutor", "profile": {"tone": "playful"}}).json()
    assert edited["skill"] == "Tamil and Hindi tutor" and edited["bot"]["tone"] == "playful"
    listed = client.get("/api/bots", headers=auth).json()
    assert [b["id"] for b in listed] == ["maya"] and listed[0]["knowledge"][0]["name"] == "words.md"
    exported = client.get("/api/bots/maya/export", headers=auth).json()
    imported = client.post("/api/bots/import", headers=auth, json=exported).json()
    assert imported["name"] == "Maya 2" and imported["knowledge"][0]["name"] == "words.md"
    assert client.post("/api/bots/import", headers=auth, json={"hello": 1}).status_code == 400
    remix = client.post("/api/bots", headers=auth, json={"name": "Mini Maya", "remix_of": "maya", "profile": {}}).json()
    assert remix["knowledge"][0]["name"] == "words.md"
    assert client.get("/api/bots/templates", headers=auth).json()["templates"] == []  # no premade Bees
    assert client.delete("/api/bees/maya", headers=auth).status_code == 200
    assert client.get("/api/bots/maya/memory", headers=auth).status_code == 404


def test_brain_and_free_cloud_api(gw, monkeypatch):
    client, auth = gw
    v = client.put("/api/brain", headers=auth, json={"url": "http://127.0.0.1:11434", "key": "sk-x", "model": "llama3.2"}).json()
    assert v["connected"] and v["key_set"] and "sk-x" not in json.dumps(v)
    assert client.post("/api/brain/test", headers=auth).json()["ok"]
    assert client.put("/api/brain", headers=auth, json={"url": "nope"}).status_code == 400
    free = client.get("/api/cloud/free", headers=auth).json()
    assert {"gpu", "pool", "hive_key", "plan"} <= set(free) and free["plan"]["plan"] == "pro"  # the owner is always on Pro
    assert client.put("/api/plan", headers=auth, json={"plan": "pro"}).status_code in (404, 405)  # Pro is paid for, not switched on
    assert client.get("/api/bees", headers=auth).json() == []  # no premade Bees
