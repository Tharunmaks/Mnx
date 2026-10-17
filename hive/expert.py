"""Expert Stream and Expert Craft, from github.com/Tharunmaks/Stream-.

Expert Stream   the models' engine. es_serve speaks the OpenAI API like llama.cpp's server, and adds prefix reuse,
                a prompt cache on disk, faster prompt reading, Turbo speculative decoding and expert streaming for
                mixture-of-experts models. When it's built, the Hive serves your Mnx models with it (hive/models.py;
                MNX_ENGINE=llama keeps llama.cpp). The owner builds it once:  python3 -m hive.admin install-expertstream
                (or says "set up expert stream" in the chat).

Expert Craft    train a model with up to 70 billion parameters on a small machine: a dense model in RAM and millions of
                page-sized experts in a sparse file. A Bee runs it on its own computer:
                  "expert craft plan 70B for 2 hours"
                  "train a 70B model with expert craft called stories for 3 hours"   (asks first: it rewrites flash pages)
                  "expert craft status stories"     "ask my expert craft model stories: Once upon a time"
                Its log is the job "expert-craft …" on the Bee's computer.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from .core import Task

DATA = Path(__file__).resolve().parent.parent / "data"
HOME = DATA / "expertstream"
SRC = HOME / "src"
ES_SERVE = HOME / "es_serve"
REPO = os.getenv("MNX_STREAM_REPO", "https://github.com/Tharunmaks/Stream-")
CRAFT_REL = Path("craft/skills/expert-craft/scripts")
IN_BEE = ".mnx/expert-craft"  # where a Bee's computer keeps the trainer (its home folder)
EC = f"python3 {IN_BEE}/expertcraft.py"
NUMPY = "python3 -c 'import numpy' 2>/dev/null || python3 -m pip install --user -q numpy || pip install -q numpy"


class ExpertError(RuntimeError):
    pass


# ---------------------------------------------------------------- the source and the engine
def es_serve_path() -> str | None:
    return shutil.which("es_serve") or (str(ES_SERVE) if ES_SERVE.exists() and os.access(ES_SERVE, os.X_OK) else None)


def fetch_source() -> Path:
    HOME.mkdir(parents=True, exist_ok=True)
    if (SRC / ".git").exists():
        subprocess.run(["git", "-C", str(SRC), "pull", "-q", "--ff-only"], capture_output=True, timeout=600)
    else:
        r = subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, str(SRC)], capture_output=True, text=True, timeout=900)
        if r.returncode:
            raise ExpertError(f"Couldn't download Stream-: {(r.stderr or r.stdout)[-300:]}")
    return SRC


def build_server() -> str:
    """Download (or update) the source and build es_serve with the machine's C compiler."""
    fetch_source()
    cc = shutil.which("clang") or shutil.which("gcc")
    if not cc or not shutil.which("make"):
        raise ExpertError("Building Expert Stream needs a C compiler and make (Termux: pkg install clang make)")
    r = subprocess.run(["make", "-s", "-C", str(SRC), f"CC={cc}", "es_serve"], capture_output=True, text=True, timeout=1800)
    if r.returncode or not (SRC / "es_serve").exists():
        raise ExpertError(f"es_serve didn't build: {(r.stderr or r.stdout)[-400:]}")
    shutil.copy2(SRC / "es_serve", ES_SERVE)
    ES_SERVE.chmod(0o755)
    return str(ES_SERVE)


def craft_scripts() -> Path:
    if not (SRC / CRAFT_REL / "expertcraft.py").exists():
        fetch_source()
    path = SRC / CRAFT_REL
    if not (path / "expertcraft.py").exists():
        raise ExpertError("The Expert Craft trainer isn't in the Stream- repository any more")
    return path


# ---------------------------------------------------------------- what was asked
CRAFT = re.compile(r"\bexpert[\s-]*craft\b|\bflash\s+experts\b", re.I)
STREAM_SETUP = re.compile(r"\b(?:set\s*up|install|build|enable|use|switch\s+to|turn\s+on)\b.*\bexpert[\s-]*stream\b", re.I)
PARAMS = re.compile(r"(\d+(?:\.\d+)?)\s*(b|bn|billion|m|million|k|thousand)\b", re.I)
MINUTES = re.compile(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?)\b", re.I)
NAMED = re.compile(r"\b(?:called|named)\s+[\"“']?(?P<n>[A-Za-z0-9][\w-]{0,30})", re.I)


def _params(text: str) -> float:
    m = PARAMS.search(text)
    if not m:
        return 70e9
    mult = {"b": 1e9, "bn": 1e9, "billion": 1e9, "m": 1e6, "million": 1e6, "k": 1e3, "thousand": 1e3}[m.group(2).lower()]
    return min(float(m.group(1)) * mult, 70e9)


def _minutes(text: str) -> float:
    m = MINUTES.search(text)
    if not m:
        return 60.0
    n = float(m.group(1))
    return max(1.0, min(n * 60 if m.group(2).lower().startswith("h") else n, 24 * 60))


def _name(text: str, fallback: str = "craft") -> str:
    m = NAMED.search(text)
    if m:
        return re.sub(r"[^a-z0-9_-]", "", m.group("n").lower())[:30] or fallback
    m = re.search(r"\b(?:status|model)\s+(?:of\s+)?(?:my\s+)?(?:expert\s*craft\s+)?(?:model\s+)?(?P<n>[a-z0-9][\w-]{0,30})\b", text, re.I)
    if m and m.group("n").lower() not in ("is", "on", "for", "with", "the", "my", "model"):
        return m.group("n").lower()
    return fallback


def parse(text: str) -> dict | None:
    t = " ".join((text or "").split())
    if STREAM_SETUP.search(t):
        return {"kind": "stream"}
    if not CRAFT.search(t):
        return None
    if re.search(r"\b(?:status|progress|how\s+is|how's)\b", t, re.I):
        return {"kind": "status", "name": _name(t)}
    m = re.search(r"\b(?:ask|talk\s+to|chat\s+with|sample|try)\b.*?(?::|\bwith\b|\bprompt\b)\s*[\"“]?(?P<p>.+?)[\"”]?$", t, re.I)
    if m and not re.search(r"\btrain\b", t, re.I):
        return {"kind": "sample", "name": _name(t), "prompt": m.group("p")[:500]}
    data = None
    dm = re.search(r"\bon\s+(?:my\s+)?(?:file\s+)?(?P<f>[\w./~-]+\.(?:txt|md|jsonl|csv))\b", t, re.I)
    if dm:
        data = dm.group("f")
    kind = "train" if re.search(r"\b(?:train|start|run|make|build)\b", t, re.I) and not re.search(r"\bplan\b", t, re.I) else "plan"
    return {"kind": kind, "params": _params(t), "minutes": _minutes(t), "name": _name(t), "data": data}


def fmt_params(n: float) -> str:
    return f"{n / 1e9:g}B" if n >= 1e9 else f"{n / 1e6:g}M" if n >= 1e6 else f"{n:g}"


# ---------------------------------------------------------------- in the chat
def _pick_bee(task: Task) -> dict | None:
    from .bees import registry
    if task.bee:
        return registry.mine(task.bee, task.user)
    mine = registry.of(task.user)
    words = re.compile(r"\b(?:train\w*|model\w*|craft\w*|ai|ml|research\w*)\b", re.I)
    return next((b for b in mine if words.search(b["skill"]) or words.search(b["name"])), mine[0] if mine else None)


async def _give_trainer(cell) -> None:
    """Put the trainer into the Bee's computer (once; updated when the Hive's copy changes)."""
    if cell.mode == "remote":
        q = shlex.quote(REPO)
        code, out = await cell.exec(f"[ -f {IN_BEE}/expertcraft.py ] || (git clone -q --depth 1 {q} .mnx/stream-src && "
                                    f"mkdir -p {IN_BEE} && cp -r .mnx/stream-src/{CRAFT_REL}/. {IN_BEE}/)", 900)
        if code:
            raise ExpertError(f"Couldn't put Expert Craft on that computer: {out[-200:]}")
        return
    src = await asyncio.to_thread(craft_scripts)
    dst = cell.safe_path(IN_BEE)
    await asyncio.to_thread(shutil.copytree, src, dst, dirs_exist_ok=True)


async def handle(task: Task, intent: dict) -> None:
    if intent["kind"] == "stream":
        await _stream(task)
        return
    from .cells import cells
    from .cloud_chat import ask
    b = _pick_bee(task)
    if not b:
        await task.answer("Expert Craft trains on a Bee's computer. Make a Bee first (New Bee in Peace, e.g. “Trainer: trains my models”).")
        return
    bee = b["name"]
    cell = cells.get(b["id"], b["name"], b["computer"])
    s = await task.step("running", "Getting Expert Craft ready on its computer", bee=bee)
    try:
        await _give_trainer(cell)
    except (ExpertError, OSError) as exc:
        await task.finish_step(s, status="error", text=str(exc)[:200])
        await task.answer(str(exc))
        return
    await task.finish_step(s)
    name = intent.get("name") or "craft"

    if intent["kind"] == "status":
        code, out = await cell.exec(f"{EC} phone-status {shlex.quote(name)}", 120)
        try:
            st = json.loads(out[out.index("{"):])
        except ValueError:
            await task.answer(f"No Expert Craft model called {name} on {bee}'s computer yet." if code else out[-800:])
            return
        await task.emit("result", bee=bee, service="Expert Craft", title=f"{name}", badge="Training",
                        details=[[k.replace("_", " "), str(v)[:80]] for k, v in list(st.items())[:14] if not isinstance(v, (dict, list))])
        await task.answer(f"That's how {name} is doing on {bee}'s computer. Its full log is the job “expert-craft {name}” there.")
        return

    if intent["kind"] == "sample":
        s = await task.step("thinking", f"Asking {name}", bee=bee)
        code, out = await cell.exec(f"{NUMPY} >/dev/null 2>&1; {EC} phone-sample {shlex.quote(name)} --prompt {shlex.quote(intent['prompt'])}", 300)
        await task.finish_step(s, status="done" if code == 0 else "error")
        text = out[out.find(intent["prompt"]):] if intent["prompt"] in out else out  # without what the login shell printed first
        await task.answer(text.strip()[-2000:] if code == 0 else f"It didn't answer: {out.strip()[-300:]}")
        return

    params, minutes = intent["params"], intent["minutes"]
    s = await task.step("running", f"Measuring {bee}'s computer and planning {fmt_params(params)} for {minutes:g} minutes", bee=bee)
    code, plan = await cell.exec(f"{NUMPY} >/dev/null 2>&1; {EC} phone-plan {params:g} --minutes {minutes:g}", 900)
    if code:
        await task.finish_step(s, status="error", text=plan.strip()[-200:])
        await task.answer(f"The plan didn't run: {plan.strip()[-400:]}")
        return
    await task.finish_step(s)
    honest = [ln.strip(" -") for ln in plan.split("Honest summary:", 1)[-1].splitlines() if ln.strip().startswith("-")][:5]
    if intent["kind"] == "plan":
        await task.emit("result", bee=bee, service="Expert Craft", title=f"Plan: {fmt_params(params)} for {minutes:g} minutes", badge="Plan",
                        details=[["", h] for h in honest] or [["Plan", plan.strip()[-600:]]])
        await task.answer(f"Say “train a {fmt_params(params)} model with expert craft called {name} for {minutes:g} minutes” to start it.")
        return
    wear = max(50, round(minutes / 60 * 200))
    reply = await ask(task, "confirm", title=f"Train {name} ({fmt_params(params)}) for {minutes:g} minutes?", service="Expert Craft",
                      ask_anyway=True, details=[["", h] for h in honest] + [["Flash writes", f"capped at {wear} GB"],
                                                                             ["Data", intent.get("data") or "TinyStories (downloaded)"]],
                      text=f"It trains on {bee}'s computer in the background, rewriting pages of its disk as it learns.",
                      actions=[{"id": "go", "label": "Start training"}, {"id": "cancel", "label": "Not now", "style": "danger"}])
    if reply["action"] != "go":
        await task.answer("Okay, nothing started.")
        return
    data = f"--data {shlex.quote(intent['data'])}" if intent.get("data") else "--dataset tinystories"
    cmd = (f"{NUMPY}; {EC} phone-train {shlex.quote(name)} {data} --params {params:g} --minutes {minutes:g} "
           f"--max-wear-gb {wear} --min-free-gb 2")
    job = cell.add_job(f"expert-craft {name}", cmd, "@manual")
    asyncio.create_task(cell.run_job(job["id"]))
    await task.emit("result", bee=bee, service="Expert Craft", title=f"{name} is training", badge="Started",
                    details=[["Size", fmt_params(params)], ["For", f"{minutes:g} minutes"], ["Log", f"job “expert-craft {name}” on its computer"],
                             ["Its computer", f"/cell.html?bee={b['id']}"]])
    await task.answer(f"{name} is training on {bee}'s computer. Say “expert craft status {name}” to check, and "
                      f"“ask my expert craft model {name}: Once upon a time” when it's done.")


async def _stream(task: Task) -> None:
    from . import users
    from .models import models
    if not users.is_owner(task.user):
        await task.answer("Expert Stream is the engine the Hive's owner sets up for the models; you get it automatically.")
        return
    have = es_serve_path()
    if have:
        await task.answer(f"Expert Stream is already the engine ({have}). Models started from now on use it.")
        return
    s = await task.step("running", "Downloading and building Expert Stream (es_serve)", bee="Queen")
    try:
        path = await asyncio.to_thread(build_server)
    except (ExpertError, OSError, subprocess.TimeoutExpired) as exc:
        await task.finish_step(s, status="error", text=str(exc)[:200])
        await task.answer(f"It didn't build: {exc}")
        return
    await task.finish_step(s, text=path)
    await models.stop_all()  # the next answer starts the model again, on Expert Stream
    await task.answer("Expert Stream is built and is now the engine for your models: prefix reuse, a prompt cache on disk and "
                      "Turbo decoding make answers start faster. The next message starts the model with it.")
