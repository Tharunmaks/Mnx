"""Two ways to work with the Bees.

Rush   every Bee works together on one page. The Queen hands the message to all of them at once and the page
       shows each Bee's live work side by side.
         "everyone run pip install numpy"   the same command on every Bee's own computer, in parallel
         "roll call" · "everyone report"    each Bee reports its status, computer, jobs and disk
         anything the Hive understands (train, create a model, create bees …) runs with all the Bees it needs
Peace  a separate chat with each Bee, like chatting with one bot. The Bee answers as itself, in its own chat.
         "$ ls -la" · "run `df -h`"         runs on that Bee's own computer
         "who are you" · "status"           what it does, its computer, jobs and children
         "create a bee called …"            a child of this Bee
         everything else the Hive understands, answered by this Bee

Without the Mnx brain, both modes use fixed phrases (no AI). With the brain, Rush lets the Queen split any goal
between the Bees and Peace lets each Bee talk in its own voice.
"""

from __future__ import annotations

import asyncio
import json
import re

from . import brain
from .bees import registry
from .core import Task

MODES = ("rush", "peace")
SHELL = re.compile(r"^\s*(?:\$\s*(?P<a>.+)|run\s*:\s*(?P<b>.+)|run\s+`(?P<c>[^`]+)`\s*)$", re.I | re.S)
WHO = re.compile(r"^\s*(?:who are you|what (?:can|do) you do|what are you|status|your status|how are you|help|about you|"
                 r"what(?:'s| is) your (?:computer|job|skill))\s*[?.!]*\s*$", re.I)
EVERYONE = r"(?:every(?:one|body|\s+bee)|all(?:\s+(?:the|my))?\s+bees|each\s+bee|all\s+of\s+you|rush)"
FAN_RUN = re.compile(rf"^\s*{EVERYONE}\s*[:,]?\s*(?:please\s+)?(?:run|execute|exec|\$)\s*:?\s*(?P<cmd>.+)$", re.I | re.S)
ROLL = re.compile(rf"\broll\s*call\b|^\s*{EVERYONE}\s*[:,]?\s*(?:report|status|check in|sound off)\b|"
                  rf"\bwhat(?:'s| is| are)\s+{EVERYONE}\s+doing\b", re.I)
OUT_MAX = 3000
FAN_LIMIT = 8  # Bees running at once in a fan-out


def shell_command(text: str) -> str | None:
    m = SHELL.match(text or "")
    return (m.group("a") or m.group("b") or m.group("c") or "").strip() if m else None


def _tail(out: str, n: int = OUT_MAX) -> str:
    out = (out or "").rstrip()
    return out if len(out) <= n else "…" + out[-n:]


async def _exec(b: dict, command: str, timeout: float = 120) -> tuple[int, str]:
    from .cells import cells
    cell = cells.get(b["id"], b["name"], b["computer"])
    return await cell.exec(command, timeout)


# ---------------------------------------------------------------- Peace: one Bee, its own chat
async def peace(task: Task) -> bool:
    """A message in a Bee's own chat. Returns False to let the Hive's shared skills answer it (as this Bee)."""
    b = registry.get(task.bee or "")
    if not b:
        await task.answer("That Bee isn't in the Hive any more.")
        return True
    name = b["name"]
    cmd = shell_command(task.text)
    if cmd:
        s = await task.step("running", f"$ {cmd[:120]}", bee=name)
        try:
            code, out = await _exec(b, cmd)
        except Exception as exc:
            await task.finish_step(s, status="error", text=str(exc)[:200])
            await task.answer(f"My computer didn't run it: {exc}")
            return True
        await task.finish_step(s, status="done" if code == 0 else "error", text=f"exit {code}")
        await task.emit("output", bee=name, command=cmd, code=code, text=_tail(out) or "(no output)")
        await task.answer(f"Ran it on my computer: exit {code}." if code == 0 else f"It exited with {code}; the output is above.")
        return True
    if WHO.match(task.text):
        await task.emit("result", bee=name, **_who(b))
        await task.answer(f"I'm the {name} Bee: {b['skill']}. Send “$ command” to run something on my computer, "
                          "or ask me anything the Hive can do.")
        return True
    return False


def _who(b: dict) -> dict:
    from .cells import cells
    v = registry.view(b)
    cell = cells.cells.get(b["id"])
    kids = registry.children(b["id"])
    details = [["Skill", b["skill"]], ["Computer", v["computer_text"]],
               ["Made by", "built in" if b.get("builtin") else v["parent_name"] or "you"],
               ["Children", ", ".join(k["name"] for k in kids) or "none yet"]]
    if cell:
        cv = cell.view()
        details += [["Jobs", f"{cv['jobs']} ({cv['running']} running)"], ["Its Cell", f"/cell.html?bee={b['id']}"]]
    return {"service": "Hive", "title": b["name"], "badge": "Bee", "details": details}


async def peace_fallback(task: Task) -> None:
    """Nothing in the Hive understood it: the Bee answers with its brain, or says what it can do."""
    b = registry.get(task.bee or "") or {"name": "Bee", "skill": ""}
    if brain.is_connected():
        await task.answer(await brain.ask(f"You are the {b['name']} Bee in the Mnx Hive. Your job: {b['skill']}. "
                                          f"Answer as yourself, briefly.\n\nUser: {task.text}", task.effort))
        return
    await task.answer(f"I'm the {b['name']} Bee ({b['skill']}). My brain (the Mnx model) isn't connected yet, so I can't chat "
                      "freely. What I can do now: “$ command” runs it on my computer, “status” tells you about me, "
                      "“create a bee called … that …” makes a child of mine, “call me” rings you with my voice, "
                      "and everything else the Hive understands (training, models, GPUs, bees).")


# ---------------------------------------------------------------- Rush: every Bee together
def rush_kind(text: str) -> str | None:
    if FAN_RUN.match(text or ""):
        return "run"
    if ROLL.search(text or ""):
        return "roll"
    return None


async def rush(task: Task) -> bool:
    """All-Bees commands. Returns False when it's an ordinary request (then the Hive's flows run, with every Bee they need)."""
    kind = rush_kind(task.text)
    if not kind:
        return False
    bs = list(registry.bees)
    await task.emit("rush", bee="Queen", text=f"Rush: {len(bs)} Bees on it", bees=[b["name"] for b in bs])
    if kind == "roll":
        await _fan(task, bs, _report)
        await task.answer(f"All {len(bs)} Bees reported in. Each one's card is on the board above.")
        return True
    cmd = FAN_RUN.match(task.text).group("cmd").strip().strip("`")
    from .cloud_chat import ask
    reply = await ask(task, "confirm", title=f"Run on all {len(bs)} Bees' computers?", service="Queen",
                      details=[["Command", cmd[:200]], ["Bees", ", ".join(b["name"] for b in bs)]],
                      text="Each Bee runs it on its own computer at the same time (8 at once). Output comes back to the board.",
                      actions=[{"id": "go", "label": "Run everywhere"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
    if reply["action"] != "go":
        await task.answer("Okay, nothing ran.")
        return True
    results = await _fan(task, bs, lambda t, b: _run_on(t, b, cmd))
    ok = sum(1 for r in results if r and r[0] == 0)
    await task.emit("result", bee="Queen", service="Rush", title=f"{ok} of {len(bs)} Bees ran it", badge="Rush",
                    details=[[b["name"], f"exit {r[0]} · {(r[1].strip().splitlines() or [''])[-1][:80]}" if r else "didn't run"]
                             for b, r in zip(bs, results)])
    await task.answer(f"{ok} of {len(bs)} Bees ran “{cmd[:60]}” successfully." + ("" if ok == len(bs) else " The others' errors are on their cards."))
    return True


async def _fan(task: Task, bs: list[dict], work) -> list:
    gate = asyncio.Semaphore(FAN_LIMIT)

    async def one(b):
        async with gate:
            try:
                return await work(task, b)
            except Exception:
                return None
    return await asyncio.gather(*(one(b) for b in bs))


async def _run_on(task: Task, b: dict, cmd: str) -> tuple[int, str] | None:
    s = await task.step("running", f"$ {cmd[:80]}", bee=b["name"])
    try:
        code, out = await _exec(b, cmd)
    except Exception as exc:
        await task.finish_step(s, status="error", text=str(exc)[:160])
        return None
    await task.finish_step(s, status="done" if code == 0 else "error", text=f"exit {code}")
    await task.emit("output", bee=b["name"], command=cmd, code=code, text=_tail(out, 1200) or "(no output)", parent=s)
    return code, out


async def _report(task: Task, b: dict) -> bool:
    s = await task.step("running", "Reporting in", bee=b["name"])
    card = _who(b)
    await task.finish_step(s, text=registry.view(b)["computer_text"])
    await task.emit("result", bee=b["name"], parent=s, **card)
    return True


async def rush_plan(task: Task) -> bool:
    """With the brain: split the goal between the Bees and work on every part at once."""
    if not brain.is_connected():
        return False
    bs = {b["id"]: b for b in registry.bees}
    roster = "\n".join(f"- {b['id']}: {b['skill']}" for b in bs.values())
    raw = await brain.ask("Split this goal between these Bees. Reply only with JSON {\"bee id\": \"its part\"}.\n"
                          f"Bees:\n{roster}\n\nGoal: {task.text}", task.effort)
    try:
        parts = {k: str(v) for k, v in json.loads(raw[raw.index("{"):raw.rindex("}") + 1]).items() if k in bs}
    except ValueError:
        return False
    if not parts:
        return False
    await task.emit("rush", bee="Queen", text=f"Rush: {len(parts)} Bees on it", bees=[bs[k]["name"] for k in parts])

    async def work(t, b):
        s = await t.step("running", parts[b["id"]][:160], bee=b["name"])
        out = await brain.ask(f"You are the {b['name']} Bee: {b['skill']}. Do your part and report briefly: {parts[b['id']]}", t.effort)
        await t.finish_step(s, text=out[:300])
        return out
    outs = await _fan(task, [bs[k] for k in parts], work)
    merged = await brain.ask("Merge these reports into one answer for the user:\n\n" +
                             "\n\n".join(f"{bs[k]['name']}: {o}" for k, o in zip(parts, outs) if o), task.effort)
    await task.answer(merged)
    return True
