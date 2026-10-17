"""Your Bees in the chat. Making one is "make a bee called … that …" (hive/bots.py); here:

  "list my bees" · "show my bees"           your Bees, who made each, and their computers
  "delete the Scout bee"                     removes it and its computer (asks first)
  "open Scout's desktop"                     starts its Vortex OS desktop
  "what computer do my bees get"             the plan decides it: Free or Pro

Nobody chooses a Bee's computer: every Bee gets its owner's plan's computer (hive/plan.py).
"""

from __future__ import annotations

import re

from .bees import describe_computer, registry
from .core import Task

BEE = "Queen"
LIST = re.compile(r"\b(?:list|show|see)\b.*\bbees\b|\bmy bees\b|\bhow many bees\b|\bwhich bees\b|\bbee (?:family|tree)\b", re.I)
DELETE = re.compile(r"\b(?:delete|remove|kill|destroy)\s+(?:the\s+|my\s+)?(?P<b>[\w -]{1,30}?)\s+bee\b", re.I)
DESKTOP = re.compile(r"\b(?:open|start|show|launch|give me)\b.*\bdesktop\b|\bdesktop\b.*\b(?:open|start|link)\b", re.I)
# "give the coding bee a large computer", "what computer do my bees get": about a Bee's computer, in any order
COMPUTER_ASK = re.compile(r"\b(?:give|set|upgrade|move|put|resize|make|switch|change|what|which|how\s+(?:much|many|big))\b", re.I)
COMPUTER_WORD = re.compile(r"\b(?:computers?|machines?|servers?|gpus?|cpus?|ram|memory|disk|storage|vram)\b", re.I)
BEE_WORD = re.compile(r"\bbees?\b", re.I)
STOP = {"a", "an", "the", "my", "and", "of", "for", "to", "that", "which", "who", "every", "all", "it", "its", "me", "i"}


def _bee_named(text: str, owner: str) -> dict | None:
    for m in re.finditer(r"\b(?:the\s+)?([\w-]+(?:\s[\w-]+)?)(?:\s+bee|'s)\b", text, re.I):
        for cand in (m.group(1), m.group(1).split()[-1]):
            b = registry.find(cand, owner)
            if b:
                return b
    for w in re.findall(r"[\w-]+", text):
        if w.lower() not in STOP and len(w) > 2:
            b = registry.find(w, owner)
            if b:
                return b
    return None


def parse(text: str, owner: str | None = None) -> dict | None:
    from .users import who
    owner = owner or who()
    t = " ".join(text.strip().split())
    if DESKTOP.search(t):
        b = _bee_named(t, owner)
        if b:
            return {"kind": "desktop", "bee": b["id"]}
    dm = DELETE.search(t)
    if dm:
        return {"kind": "delete", "bee": dm.group("b").strip()}
    if COMPUTER_ASK.search(t) and COMPUTER_WORD.search(t) and BEE_WORD.search(t):
        return {"kind": "computer"}
    if LIST.search(t):
        return {"kind": "list"}
    return None


async def handle(task: Task, text: str) -> None:
    from .cloud_chat import ask
    from .plan import PLANS, plan
    intent = parse(text, task.user) or {}
    kind = intent.get("kind")
    mine = registry.of(task.user)
    if kind == "list":
        if not mine:
            await task.answer("You don't have any Bees yet. Say “make a bee called … that …”, or press New Bee in Peace.")
            return
        rows = []

        def walk(parent: str, depth: int) -> None:
            for b in mine:
                if b.get("created_by", "you") == parent:
                    rows.append([("  " * depth) + ("└ " if depth else "") + b["name"], f"{b['skill']} · {describe_computer(b['computer'])}"])
                    walk(b["id"], depth + 1)
        walk("you", 0)
        await task.emit("result", service="Bees", title=f"{len(mine)} Bee{'s' if len(mine) != 1 else ''}", badge="Bees", details=rows)
        await task.answer(f"You have {len(mine)} Bee{'s' if len(mine) != 1 else ''} (your plan has room for {plan.max_bees(task.user)}). "
                          "Each has its own chat in Peace.")
        return
    if kind == "computer":
        v = PLANS[plan.of(task.user)]
        await task.emit("result", service="Bees", title=f"Every Bee gets the {v['label']} computer", badge="Computer",
                        details=[[p["label"], p["computer"]] for p in PLANS.values()])
        await task.answer(f"Computers aren't picked one by one: on {v['label']} every Bee gets {v['computer']}."
                          + ("" if plan.of(task.user) == "pro" else " Get Pro (Upgrade in the menu) for 16 GB RAM, 100 GB disk and a 16 GB GPU."))
        return
    if kind == "desktop":
        from .cells import cells
        from .vortex import desktops
        b = registry.mine(intent["bee"], task.user)
        if not b:
            await task.answer("That isn't one of your Bees.")
            return
        s = await task.step("running", f"Starting {b['name']}'s Vortex OS desktop", bee=b["name"])
        try:
            cell = cells.get(b["id"], b["name"], b["computer"])
            st = await desktops.start(cell)
        except Exception as exc:
            await task.finish_step(s, status="error", text=str(exc)[:200])
            await task.answer(f"The desktop didn't start: {exc}")
            return
        await task.finish_step(s, text=f"Running on port {st['port']}")
        await task.emit("result", service="Vortex OS", title=f"{b['name']}'s desktop is running", badge="Desktop",
                        details=[["Bee name", st["user"]], ["Password", "on the Desktop tab of its computer (not shown in the chat)"],
                                 ["Home folder", "this Bee's work folder"], ["Its computer", f"/cell.html?bee={b['id']}#desktop"]])
        await task.answer(f"{b['name']}'s Vortex OS desktop is running. Open its computer's Desktop tab and press Open desktop; sign in as {st['user']}.")
        return
    if kind == "delete":
        b = registry.find(intent["bee"], task.user)
        if not b:
            await task.answer(f"You don't have a Bee called {intent['bee']}.")
            return
        kids = registry.children(b["id"])
        reply = await ask(task, "confirm", title=f"Remove {b['name']}?", service="Bees", ask_anyway=True,
                          details=[["Responsibility", b["skill"]], ["Children", ", ".join(k["name"] for k in kids) or "none"]],
                          text="Its chat stays on your device, but its computer, files, memory and knowledge are deleted. Its children stay and move up.",
                          actions=[{"id": "go", "label": "Remove", "style": "danger"}, {"id": "cancel", "label": "Keep it"}])
        if reply["action"] != "go":
            await task.answer("Okay, kept.")
            return
        await registry.delete(b["id"])
        await task.answer(f"Removed {b['name']}." + (f" {', '.join(k['name'] for k in kids)} moved up." if kids else ""))
        return
    await task.answer("Say “make a bee called … that …”, “list my bees” or “delete the … bee”.")


def describe(b: dict) -> str:
    return describe_computer(b["computer"])
