"""Plans in the chat: "subscribe me to plus", "switch to free", "what plan am I on". Both plans are free.

Changing the plan gives every Bee the plan's computer (Standard or Plus), except Bees whose computer you chose
yourself and the Bees that get the whole machine (Cloud, GPU).
"""

from __future__ import annotations

import re

from .bees import SIZES, registry
from .core import Task
from .plan import PLANS, plan

SUBSCRIBE = re.compile(r"\b(?:subscribe|sign\s+me\s+up|upgrade|join|get|switch|move|go)\b.*\bplus\b|\bplus\s+(?:plan|subscription)\b", re.I)
DOWN = re.compile(r"\b(?:unsubscribe|cancel\s+(?:my\s+)?(?:plus|subscription)|(?:switch|go|move)\s+(?:back\s+)?to\s+(?:the\s+)?free(?:\s+plan)?)\b", re.I)
WHICH = re.compile(r"\b(?:which|what)\s+plan\b|\bmy\s+(?:plan|subscription)\b", re.I)


def parse(text: str) -> dict | None:
    t = text or ""
    if re.search(r"\bbees?\b|\bbot\b", t, re.I):  # "upgrade the coding bee to plus" is about one Bee's computer
        return None
    if DOWN.search(t):
        return {"kind": "set", "plan": "free"}
    if SUBSCRIBE.search(t):
        return {"kind": "set", "plan": "plus"}
    if WHICH.search(t):
        return {"kind": "which"}
    return None


async def apply(new_plan: str) -> list[str]:
    """Give every Bee the plan's computer (unless you chose its computer, or it has the whole machine)."""
    size = PLANS[new_plan]["size"]
    changed = []
    for b in list(registry.bees):
        c = b["computer"]
        if b.get("computer_set") or c.get("size") in ("max", "custom") or c.get("size") == size or c.get("cloud"):
            continue
        z = SIZES[size]
        change = {"size": size, "gpu": bool(z.get("gpu_vram_gb"))}
        if size == "plus" and c.get("where", "here") == "here":
            from .cloud import pool_place
            spot = pool_place()  # the Hive's own cloud, when it has room
            if spot:
                change["where"] = spot
        await registry.set_computer(b["id"], change, by_you=False)
        changed.append(b["name"])
    return changed


async def handle(task: Task, intent: dict) -> None:
    if intent["kind"] == "which":
        v = plan.view()
        await task.emit("result", bee="Queen", service="Plan", title=f"You're on {v['label']} ({v['price']})", badge="Plan",
                        details=[[p["label"], f"{p['price']} · {p['computer']}"] for p in v["plans"]])
        await task.answer(f"You're on {v['label']}. " + ("Say “subscribe me to plus” for 16 GB RAM, 100 GB disk and a 16 GB GPU, free."
                                                         if v["plan"] == "free" else "Every Bee has the Plus computer."))
        return
    want = intent["plan"]
    if plan.name == want:
        await task.answer(f"You're already on {PLANS[want]['label']}.")
        return
    try:
        plan.set(want)
    except ValueError as exc:
        await task.answer(str(exc))
        return
    s = await task.step("running", f"Giving every Bee the {PLANS[want]['label']} computer", bee="Queen")
    try:
        changed = await apply(want)
    except Exception as exc:
        await task.finish_step(s, status="error", text=str(exc)[:200])
        await task.answer(f"You're on {PLANS[want]['label']} now, but some computers didn't change: {exc}")
        return
    await task.finish_step(s, text=f"{len(changed)} Bees upgraded" if want == "plus" else f"{len(changed)} Bees back on Standard")
    from .gpu_pool import pool
    gpu_note = ("" if want == "free" else
                " The GPU is the Hive's: " + ("this machine's own GPU." if pool.local_ready() else
                                              "a free 16 GB Kaggle GPU." if pool.kaggle_ready() else
                                              "set up the free GPU once on the Free cloud page (Kaggle, 2 minutes)."))
    await task.emit("result", bee="Queen", service="Plan", title=f"{PLANS[want]['label']} · {PLANS[want]['price']}", badge="Plan",
                    details=[["Every Bee gets", PLANS[want]["computer"]], ["Changed", ", ".join(changed) or "nothing to change"]])
    await task.answer(f"Done: you're on {PLANS[want]['label']}, free." + gpu_note)
