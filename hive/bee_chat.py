"""Bees in the chat: create Bees, have a Bee create Bees, and give each one the computer it needs.

  "create a bee called Scout that watches prices"
  "tell the Coding bee to create a bee called Tester that runs the tests"
  "make a large bee with a gpu called Trainer that trains models"
  "give the Scout bee a large computer" · "give Scout a gpu" · "move Scout to my RunPod server"
  "give Scout the most powerful computer"   → this server's max, a connected GPU server, or rent one
  "list my bees" · "delete the Scout bee"
"""

from __future__ import annotations

import re

from .bees import SIZE_WORDS, SIZES, BeeError, describe_computer, registry
from .core import Task

BEE = "Queen"
CREATE = re.compile(r"\b(?:create|make|add|build|spawn|hatch|start)\b(?P<adj>(?:\s+(?:a|an|another|one|new|child|baby|"
                    r"small|tiny|medium|large|big|huge|xl|powerful|gpu))*)\s+bees?\b", re.I)
PARENT = re.compile(r"\b(?:tell|ask|have|get|let)\s+(?:the\s+|my\s+)?(?P<p>[\w -]{1,30}?)(?:\s+bee)?\s+to\b", re.I)
NAME = re.compile(r"\b(?:called|named|name it|named as)\s+[\"“']?(?P<n>[A-Za-z][\w -]{0,23}?)[\"”']?"
                  r"(?=\s+(?:that|which|who|to|for|with|on|and|using)\b|[,.!?]|$)", re.I)
SKILL = re.compile(r"(?:^|\s)(?:that|which|who|to|for)\s+(?P<s>.+)$", re.I)
SIZE_RE = re.compile(r"\b(?P<w>very powerful|most powerful|extra large|maximum|biggest|strongest|powerful|small|tiny|medium|normal|"
                     r"large|big|huge|xl|max|full)\s+(?:computer|machine|server|bee|one|size|box|cell)\b"
                     r"|\b(?:to|size)\s+(?P<w2>small|medium|large|xl|max)\b", re.I)
GPU_RE = re.compile(r"\b(?:with|has|have|a|give \w+(?: a)?|gpu)\s*(?:an?\s+)?gpus?\b|\bgpu\s+(?:computer|machine|server|bee)\b", re.I)
WHERE_RE = re.compile(r"\b(?:on|to|onto)\s+(?:my\s+|the\s+)?(?P<w>[\w .-]{1,40}?)\s+(?:server|machine|pod|box)\b", re.I)
COMPUTER = re.compile(r"\b(?:give|set|upgrade|move|put|resize|make|switch|change)\b.*\b(?:computer|machine|server|gpu|cpus?|ram|memory|"
                      r"small|medium|large|xl|max|powerful|bigger|pod|runpod)\b", re.I)
LIST = re.compile(r"\b(?:list|show|see)\b.*\bbees\b|\bmy bees\b|\bhow many bees\b|\bwhich bees\b|\bbee (?:family|tree)\b", re.I)
DELETE = re.compile(r"\b(?:delete|remove|kill|destroy)\s+(?:the\s+|my\s+)?(?P<b>[\w -]{1,30}?)\s+bee\b", re.I)
STOP = {"a", "an", "the", "my", "and", "of", "for", "to", "that", "which", "who", "every", "all", "it", "its", "me", "i"}


def _size(text: str) -> str | None:
    m = SIZE_RE.search(text)
    if not m:
        return None
    return SIZE_WORDS.get((m.group("w") or m.group("w2") or "").lower())


def _where(text: str) -> str | None:
    """A connected server named in the text ("on my runpod server")."""
    from .lab import lab
    m = WHERE_RE.search(text)
    words = (m.group("w") if m else text).lower()
    if m and words in ("this", "this same", "the hive", "hive", "local", "home"):
        return "here"
    for t in lab.targets.values():
        name = t["name"].lower()
        if (m and (words in name or name in words)) or re.search(rf"\b{re.escape(t['id'])}\b", text.lower()):
            return t["id"]
    if re.search(r"\brunpod\b", text, re.I):
        for t in lab.targets.values():
            if "runpod" in t["name"].lower():
                return t["id"]
    return None


def _bee_named(text: str) -> dict | None:
    for m in re.finditer(r"\b(?:the\s+)?([\w-]+(?:\s[\w-]+)?)\s+bee(?:'s)?\b", text, re.I):
        for cand in (m.group(1), m.group(1).split()[-1]):
            b = registry.find(cand)
            if b:
                return b
    for w in re.findall(r"[\w-]+", text):
        if w.lower() not in STOP and len(w) > 2:
            b = registry.find(w)
            if b:
                return b
    return None


def parse(text: str) -> dict | None:
    t = " ".join(text.strip().split())
    m = CREATE.search(t)
    if m:
        parent = None
        pm = PARENT.search(t[:m.start()] + " ")
        if pm:
            parent = pm.group("p").strip()
        rest = t[m.end():]
        nm = NAME.search(rest)
        name = nm.group("n").strip() if nm else ""
        sm = SKILL.search(rest[nm.end():] if nm else rest)
        skill = (sm.group("s") if sm else "").strip(" .!?")
        skill = re.sub(r"\s*\b(?:with|using|on)\s+(?:a\s+|an\s+|the\s+|my\s+)?(?:[\w.-]+\s+){0,2}(?:gpus?|computer|machine|server|pod)\b", "", skill, flags=re.I).strip(" ,.")
        adj = (m.group("adj") or "").lower()
        size = _size(t) or next((SIZE_WORDS[w] for w in adj.split() if w in SIZE_WORDS), None)
        gpu = bool(GPU_RE.search(t)) or " gpu" in adj
        return {"kind": "create", "parent": parent, "name": name, "skill": skill, "size": size, "gpu": gpu, "where": _where(t) if WHERE_RE.search(t) else None}
    dm = DELETE.search(t)
    if dm:
        return {"kind": "delete", "bee": dm.group("b").strip()}
    if LIST.search(t):
        return {"kind": "list"}
    if COMPUTER.search(t) and (re.search(r"\bbee", t, re.I) or _bee_named(t)):
        b = _bee_named(t)
        if b:
            return {"kind": "computer", "bee": b["id"], "size": _size(t) or (
                        "max" if re.search(r"\b(?:most powerful|very powerful|powerful|strongest|biggest|maximum)\b", t, re.I) else None),
                    "gpu": True if re.search(r"\bgpus?\b", t, re.I) else None, "where": _where(t),
                    "where_text": (WHERE_RE.search(t).group("w") if WHERE_RE.search(t) else None),
                    "powerful": bool(re.search(r"\b(?:powerful|strongest|biggest|maximum|max|gpu)\b", t, re.I))}
    return None


def _auto_name(skill: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z][\w-]*", skill) if w.lower() not in STOP][:2]
    return " ".join(w.capitalize() for w in words)[:24] or "Helper"


def _server_options() -> list[dict]:
    from .lab import lab
    from .cells import machine
    m = machine()
    opts = [{"value": "here", "label": f"This server · {m['cpus']} CPUs, {m['memory_gb']:g} GB" + (" · GPU" if lab.gpu_local else "")}]
    for t in lab.targets.values():
        info = t.get("info") or {}
        gpus = ", ".join(info.get("gpus") or []) or "no GPU"
        ram = round(int(info.get("ram_mb") or 0) / 1024)
        opts.append({"value": t["id"], "label": f"{t['name']} · {info.get('cpus') or '?'} CPUs, {ram or '?'} GB · {gpus}"
                     + ("" if t.get("status") == "ok" else f" ({t.get('status')})")})
    if lab._secrets.get("RUNPOD_API_KEY"):
        opts.append({"value": "rent", "label": "Rent a GPU machine on RunPod (costs money; you confirm the price)"})
    return opts


SIZE_OPTS = [{"value": k, "label": f"{v['label']}" + ("" if k == "max" else f" · {v['cpus']} CPU{'s' if v['cpus'] > 1 else ''}, {v['memory_gb']} GB")}
             for k, v in SIZES.items()]


async def _rent(task: Task, parent: str) -> str | None:
    from .swarm import rent_gpu
    return await rent_gpu(task, parent, {"params": 1e9})


async def handle(task: Task, text: str) -> None:
    from .cloud_chat import ask
    intent = parse(text) or {}
    kind = intent.get("kind")
    bee = await task.emit("bee_created", bee=BEE, text="is looking after the Hive's Bees")
    if kind == "list":
        rows = []

        def walk(parent: str | None, depth: int) -> None:
            for b in registry.bees:
                owner = b.get("created_by")
                if (parent is None and owner in (None, "you")) or (parent and owner == parent):
                    v = registry.view(b)
                    who = "built in" if b.get("builtin") else f"made by {v['parent_name']}"
                    rows.append([("  " * depth) + ("└ " if depth else "") + b["name"], f"{b['skill']} · {who} · {v['computer_text']}"])
                    walk(b["id"], depth + 1)
        walk(None, 0)
        await task.emit("result", parent=bee, service="Hive", title=f"{len(registry.bees)} Bees", badge="Bees", details=rows)
        await task.answer(f"The Hive has {len(registry.bees)} Bees; {sum(1 for b in registry.bees if not b.get('builtin'))} were made by you or by other Bees. "
                          "Open the Hive page to see each one's computer.")
        return
    if kind == "delete":
        b = registry.find(intent["bee"])
        if not b:
            await task.answer(f"I don't have a Bee called {intent['bee']}.")
            return
        if b.get("builtin"):
            await task.answer(f"{b['name']} is a built-in Bee; it can't be removed.")
            return
        kids = registry.children(b["id"])
        reply = await ask(task, "confirm", parent=bee, title=f"Remove the {b['name']} Bee?", service="Hive", ask_anyway=True,
                          details=[["Computer", registry.view(b)["computer_text"]], ["Children", ", ".join(k["name"] for k in kids) or "none"]],
                          text="Its computer is deleted: container, files, browser storage and jobs. Its children stay and move up to its parent.",
                          actions=[{"id": "go", "label": "Remove", "style": "danger"}, {"id": "cancel", "label": "Keep it"}])
        if reply["action"] != "go":
            await task.answer("Okay, kept.")
            return
        await registry.delete(b["id"])
        await task.answer(f"Removed the {b['name']} Bee." + (f" {', '.join(k['name'] for k in kids)} now belong to its parent." if kids else ""))
        return
    if kind == "create":
        parent_id = "you"
        if intent.get("parent"):
            p = registry.find(intent["parent"])
            if not p:
                await task.answer(f"I don't have a Bee called {intent['parent']} to ask. Say “list my bees” to see them.")
                return
            parent_id = p["id"]
        skill = intent.get("skill") or ""
        name = intent.get("name") or (_auto_name(skill) if skill else "")
        computer = {"size": intent.get("size") or "small", "gpu": bool(intent.get("gpu"))}
        where = intent.get("where") or (registry.get(parent_id)["computer"].get("where", "here") if parent_id != "you" else "here")
        opts = [o for o in _server_options() if o["value"] != "rent"]
        if where not in [o["value"] for o in opts]:
            where = "here"
        maker = "you" if parent_id == "you" else registry.get(parent_id)["name"]
        reply = await ask(task, "clarify", parent=bee, title=f"New Bee{'' if parent_id == 'you' else ' from ' + maker}", fields=[
            {"name": "name", "label": "Name", "value": name, "placeholder": "e.g. Scout"},
            {"name": "skill", "label": "What it does (one line)", "value": skill, "placeholder": "e.g. watches prices and tells me when they drop"},
            {"name": "size", "label": "Its computer", "input": "select", "value": computer["size"], "options": SIZE_OPTS},
            {"name": "where", "label": "Runs on", "input": "select", "value": where, "options": opts}],
            text=f"It gets its own computer: a terminal, files, a browser and 24/7 jobs. {'It belongs to ' + maker + '.' if parent_id != 'you' else ''}",
            actions=[{"id": "create", "label": "Create Bee"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
        if reply["action"] != "create":
            await task.answer("Okay, no new Bee.")
            return
        v = reply.get("values") or {}
        name = (v.get("name") or name).strip()
        skill = (v.get("skill") or skill).strip()
        computer = {"size": v.get("size") or computer["size"], "gpu": computer["gpu"], "where": v.get("where") or where}
        if not name or not skill:
            await task.answer("A Bee needs a name and a one-line skill. Try again with both, e.g. “create a bee called Scout that watches prices”.")
            return
        try:
            b = await registry.create(name, skill, parent_id, computer=computer)
        except BeeError as exc:
            await task.answer(f"I couldn't create it: {exc}.")
            return
        view = registry.view(b)
        await task.emit("result", parent=bee, service="Hive", title=f"{b['name']} Bee is ready", badge="New Bee",
                        details=[["Skill", b["skill"]], ["Made by", view["parent_name"]], ["Computer", view["computer_text"]],
                                 ["Its Cell", f"/cell.html?bee={b['id']}"]],
                        text=f"{b['name']} can make its own Bees from its terminal: python3 .mnx/mnx-bee new \"Name\" \"what it does\"")
        await task.answer(f"{b['name']} Bee is ready with its own computer ({view['computer_text']}). Open it from the Hive page to use its terminal, files and jobs.")
        return
    if kind == "computer":
        b = registry.get(intent["bee"])
        change: dict = {}
        if intent.get("size"):
            change["size"] = intent["size"]
        if intent.get("gpu") is not None:
            change["gpu"] = intent["gpu"]
        if intent.get("where"):
            change["where"] = intent["where"]
        if intent.get("powerful") and "where" not in change:
            # the strongest place there is: ask, showing what each server has
            opts = _server_options()
            reply = await ask(task, "clarify", parent=bee, title=f"A powerful computer for {b['name']}", ask_anyway=True, fields=[
                {"name": "where", "label": "Where", "input": "select", "value": opts[-1]["value"] if len(opts) > 1 else "here", "options": opts},
                {"name": "size", "label": "Size", "input": "select", "value": "max", "options": SIZE_OPTS}],
                text="The most powerful computer is the biggest machine you have: a connected GPU server, or one you rent. "
                     "On this server, Max gives the Bee the whole machine.",
                actions=[{"id": "go", "label": "Set it"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
            if reply["action"] != "go":
                await task.answer("Okay, nothing changed.")
                return
            v = reply.get("values") or {}
            change["size"] = v.get("size") or "max"
            where = v.get("where") or "here"
            if where == "rent":
                tid = await _rent(task, bee)
                if not tid:
                    await task.answer("Nothing was rented, so the Bee's computer is unchanged.")
                    return
                where = tid
            change["where"] = where
        if not change and intent.get("where_text"):
            await task.answer(f"I don't have a server called {intent['where_text']} connected. Connect it in Cloud Bee · Lab → Servers "
                              "(or say “give the … bee the most powerful computer” to rent one), then ask again.")
            return
        from .lab import lab
        if change.get("gpu") and change.get("where", b["computer"].get("where", "here")) == "here" and not lab.gpu_local:
            change.pop("gpu")
            if not change:
                await task.answer(f"This server has no GPU Docker can use, so {b['name']} can't get one here. "
                                  f"Say “give the {b['name']} bee the most powerful computer” to pick a GPU server or rent one.")
                return
        if not change:
            await task.answer(f"What should {b['name']}'s computer be? Say a size (small, medium, large, xl, max), “with a gpu”, or a server (“on my RunPod server”).")
            return
        s = await task.step("running", f"Setting up {b['name']}'s computer", bee=BEE, parent=bee)
        try:
            b = await registry.set_computer(b["id"], change)
        except (BeeError, RuntimeError) as exc:
            await task.finish_step(s, status="error", text=str(exc)[:200])
            await task.answer(f"I couldn't change it: {exc}")
            return
        text = registry.view(b)["computer_text"]
        await task.finish_step(s, text=text)
        moved = "where" in change
        await task.answer(f"{b['name']}'s computer is now {text}." + (" Files stay where they were; copy what it needs from its old folder." if moved else
                          " Its files and jobs are kept; the computer restarted with the new size."))
        return
    await task.answer("Say “create a bee called … that …”, “give the … bee a large computer” or “list my bees”.")


def describe(b: dict) -> str:
    return describe_computer(b["computer"])
