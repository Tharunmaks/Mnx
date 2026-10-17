"""Bees make games by themselves: they write it, play-test it, fix it and hand you a private link to play.

  "create an open world game in 3d"      "make a racing game"      "create a 3d game like GTA 6"
  "use three.js for the game and make it yourself and run it"   (it takes the game from earlier in the chat)

How a Bee does it (no steps for you):
  1. writes the game with three.js (it runs in any browser, phones too), starting from a working 3D open-world
     city (hive/templates/game_city) and changing it into what you asked for
  2. saves it in its own Vortex OS computer, ~/games/<name>/, where you can open and edit it
  3. deploys it privately with Chroma: a preview at an unguessable /d/<id>/ link, never a public page
  4. plays it in its own browser: catches the game's errors, takes a screenshot, and checks it really runs
  5. fixes what broke (up to two rounds); if it still won't run, you get the working starter game instead
A game "like GTA" is an original game in that style: its own city, names and art, nothing copied from the real one.
Unity can't run on the Hive's computers (it needs a desktop with a GPU, a Unity account and ~10 GB), so the Bees use
three.js; ask for Unity code and the model writes it for your own Unity.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from . import brain, sites
from .core import Task

TEMPLATE = Path(__file__).resolve().parent / "templates" / "game_city"
GAME = re.compile(r"\b(?:make|create|build|design|code|develop|generate|program|craft)\b[^.?!]*\b(?:game|gaming\s+world|open[\s-]world)\b"
                  r"|\bgame\b[^.?!]*\b(?:make|create|build|run)\s+it\b|\b(?:make|create|build)\s+(?:it|the\s+game)\s+yourself\b", re.I)
# Games people name as the style they want: the Bee makes an original game in that style, nothing copied.
STYLES = {"gta": "GTA", "grand theft auto": "GTA", "minecraft": "Minecraft", "pokemon": "Pokémon", "pokémon": "Pokémon",
          "mario": "Mario", "fortnite": "Fortnite", "pubg": "PUBG", "free fire": "Free Fire", "call of duty": "Call of Duty",
          "valorant": "Valorant", "among us": "Among Us", "subway surfers": "Subway Surfers", "temple run": "Temple Run",
          "roblox": "Roblox", "zelda": "Zelda", "need for speed": "Need for Speed", "forza": "Forza", "fifa": "FIFA",
          "clash of clans": "Clash of Clans", "angry birds": "Angry Birds", "candy crush": "Candy Crush", "pac-man": "Pac-Man",
          "pacman": "Pac-Man", "tetris": "Tetris", "sonic": "Sonic", "red dead": "Red Dead", "cyberpunk": "Cyberpunk",
          "watch dogs": "Watch Dogs", "spider-man": "Spider-Man", "spiderman": "Spider-Man", "batman": "Batman"}
UNITY = re.compile(r"\bunity\b|\bunreal\b", re.I)
FIX_ROUNDS = 2

SPEC = """You make complete, fun browser games with three.js. You get a working 3D game to start from; change and extend it
into the game the user asked for. Write every file of the game in this exact format, nothing before or after:

=== FILE: index.html ===
(full content)
=== END ===
=== FILE: game.js ===
(full content)
=== END ===

Rules
- Keep the importmap in index.html exactly as it is ("three" → /vendor/three/three.module.js, "three/addons/" →
  /vendor/three/addons/). No other libraries, no npm, no build step, no files you don't write.
- The game must run as written: every variable defined, no placeholders, no "TODO". Keep window.__game = {..., ready: true}
  at the end of game.js.
- Keep keyboard and touch controls (phones), a HUD with the goal and score, and a way to win or progress.
- If the user names a famous game (GTA, Minecraft …), make an ORIGINAL game in that style: your own title, city or world,
  characters, missions and art made from shapes and colours. Never copy names, logos, characters, maps, music or text from it.
- Keep it light enough for a phone: instanced meshes for many objects, few lights, modest shadows.
"""


def parse(text: str) -> dict | None:
    t = " ".join((text or "").split())
    if not GAME.search(t):
        return None
    from .cloud_chat import CODE_ASK
    code = CODE_ASK.search(t)
    if code and not re.search(r"three\.?js|yourself|run it", t, re.I):  # "make a snake game in python": code for the model to write
        return None
    if re.search(r"\b(?:game\s+(?:theory|plan|show|night)|video\s+game\s+(?:ideas?|names?))\b", t, re.I):
        return None
    low = t.lower()
    styles = sorted({v for k, v in STYLES.items() if re.search(rf"\b{re.escape(k)}\b", low)})
    return {"goal": t, "styles": styles, "unity": bool(UNITY.search(t))}


def template_files() -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(TEMPLATE.iterdir()) if p.is_file()}


def _brief(task: Task, intent: dict) -> str:
    """What to make: this message, plus what was said about the game earlier in the chat ("make it yourself")."""
    earlier = [h["text"] for h in (getattr(task, "history", None) or []) if h.get("role") == "user" and re.search(r"\bgame\b", h.get("text", ""), re.I)]
    goal = " ".join(earlier[-2:] + [intent["goal"]]) if earlier and len(intent["goal"]) < 120 else intent["goal"]
    if intent["styles"]:
        goal += (f"\n(The user named {', '.join(intent['styles'])}: make an original game in that style with your own title, "
                 "world, characters and missions; copy nothing from it.)")
    return goal


def name_for(goal: str, styles: list[str]) -> str:
    words = re.sub(r"\b(?:create|make|build|design|code|develop|generate|me|a|an|the|my|game|games|in|3d|2d|like|not|use|for|and|it|"
                   r"yourself|run|three\.?js|with|of|please)\b", " ", goal.lower())
    for st in styles:
        words = re.sub(re.escape(st.lower()), " ", words)
    words = re.sub(r"\b\d+\b", " ", words)
    name = sites.slug(" ".join(words.split()[:3]))
    return (name or "open-world") + "-game" if not name.endswith("game") else name


async def write_game(task: Task, goal: str, current: dict[str, str], errors: list[str] | None = None) -> dict[str, str]:
    listing = "\n".join(f"=== FILE: {p} ===\n{v}\n=== END ===" for p, v in current.items())
    if errors:
        ask = (f"The game was played in a browser and broke with these errors:\n" + "\n".join(f"- {e}" for e in errors[:8])
               + f"\n\nFix them. The game should still be: {goal}\nWrite every file again, fixed.")
    else:
        ask = f"Make this game: {goal}\nStart from this working game and change it into that:"
    raw = await brain.brain.chat([{"role": "system", "content": SPEC}, {"role": "user", "content": f"{ask}\n\n{listing}"}],
                                 task.effort, max_tokens=12000, temperature=0.5)
    files, _ = sites.parse_files(raw)
    files = {p: v for p, v in files.items() if p.endswith((".html", ".js", ".css", ".json"))}
    if "index.html" not in files or not any(p.endswith(".js") for p in files):
        raise sites.SiteError("the model didn't write a whole game")
    return {**{p: v for p, v in current.items() if p.endswith(".css") and p not in files}, **files}


async def play(url: str) -> dict:
    """Open the game in a browser for a few seconds: its errors, whether it says it's ready, and a screenshot."""
    import base64
    from .browser import BrowserError, BrowserSession
    port = sites.chroma.port
    b = BrowserSession(allow={f"127.0.0.1:{port}", f"localhost:{port}"})
    try:
        await b.do("goto", url=sites.chroma.url + url)
        await asyncio.sleep(4)
        ready = (await b.do("eval", js="Boolean(window.__game && window.__game.ready)")).get("value")
        errs = (await b.do("errors")).get("errors") or []
        shot = (await b.do("screenshot")).get("image", "")
        return {"ok": bool(ready) and not errs, "ready": bool(ready), "errors": errs[:10],
                "shot": base64.b64decode(shot.split(",", 1)[1]) if "," in shot else b""}
    except BrowserError as exc:
        if "couldn't start" in str(exc):
            return {"ok": None, "errors": [], "shot": b"", "skipped": "this machine has no Chromium to play-test it"}
        return {"ok": False, "ready": False, "errors": [str(exc)[:300]], "shot": b""}
    finally:
        await b.shutdown()


def _pick_bee(task: Task) -> dict | None:
    from .bees import registry
    if task.bee:
        return registry.mine(task.bee, task.user)
    mine = registry.of(task.user)
    words = re.compile(r"\b(?:game\w*|code\w*|develop\w*|build\w*|design\w*|web\w*|app\w*)\b", re.I)
    return next((b for b in mine if words.search(b["skill"]) or words.search(b["name"])), mine[0] if mine else None)


async def handle(task: Task, intent: dict) -> None:
    from .apps import _shot_id
    b = _pick_bee(task)
    if not b:
        await task.answer("A Bee makes the game on its own computer. Make one first (New Bee in Peace, e.g. “Gamer: makes my games”), then ask again.")
        return
    bee = b["name"]
    goal = _brief(task, intent)
    name = name_for(intent["goal"], intent["styles"])
    files = template_files()
    note = []
    if intent["unity"]:
        note.append("Unity can't run on the Hive's computers (it needs a desktop with a graphics card, a Unity account and about "
                    "10 GB), so I made it with three.js, which plays in any browser, phones included.")
    if intent["styles"]:
        note.append(f"It's an original game in the style of {', '.join(intent['styles'])}: its own world, names and art, nothing copied.")

    if brain.is_connected():
        s = await task.step("writing", "Designing and writing the game (three.js)", bee=bee)
        try:
            files = await write_game(task, goal, files)
            await task.finish_step(s, text=f"{len(files)} files")
        except (sites.SiteError, brain.BrainNotConnected) as exc:
            await task.finish_step(s, status="error", text=str(exc)[:160])
            note.append("The model's own version didn't come out whole, so this is the starter open-world game; ask for changes and I'll make them.")
    else:
        note.append("No model is running on this Hive, so this is the starter open-world game as it is.")

    result: dict = {}
    d: dict = {}
    for attempt in range(FIX_ROUNDS + 1):
        s = await task.step("running", "Deploying it privately" if attempt == 0 else f"Deploying fix {attempt}", bee=bee)
        try:
            await sites.chroma.ensure()
            d = await sites.chroma.deploy(task.user, name, files, [], private=True)
        except Exception as exc:  # the sites server: say what went wrong, keep the files
            await task.finish_step(s, status="error", text=str(exc)[:200])
            folder = await sites.save_to_bee(b, name, files, name, folder="games")
            await task.answer(f"The game is written and saved in {bee}'s computer at {folder}, but it didn't deploy: {exc}")
            return
        await task.finish_step(s, text="private link ready")
        s = await task.step("running", "Playing it to check it works", bee=bee)
        result = await play(d["url"])
        if result.get("skipped"):
            await task.finish_step(s, status="done", text=f"Not play-tested: {result['skipped']}")
            break
        if result["ok"]:
            await task.finish_step(s, text="It runs: no errors")
            break
        problems = result["errors"] or ["the game never started (window.__game.ready stayed false)"]
        await task.finish_step(s, status="error", text=f"{len(problems)} problem{'s' if len(problems) != 1 else ''}: {problems[0][:120]}")
        if attempt == FIX_ROUNDS or not brain.is_connected():
            files = template_files()  # still broken: the starter game, which is known to run
            note.append("My changes kept breaking, so you got the starter open-world game; tell me what to change and I'll try again.")
            d = await sites.chroma.deploy(task.user, name, files, [], private=True)
            result = await play(d["url"])
            break
        s = await task.step("writing", "Fixing what broke", bee=bee)
        try:
            files = await write_game(task, goal, files, problems)
            await task.finish_step(s)
        except (sites.SiteError, brain.BrainNotConnected) as exc:
            await task.finish_step(s, status="error", text=str(exc)[:160])
            files = template_files()

    folder = await sites.save_to_bee(b, name, files, d.get("name", name), folder="games")
    sid = _shot_id(task.user, result.get("shot") or b"", "image/jpeg")
    if sid:
        await task.emit("screen", bee=bee, title=f"{name} running in {bee}'s browser", shot=sid, live=d["url"])
    tested = "play-tested in a browser: no errors" if result.get("ok") else ("not play-tested (no Chromium here)" if result.get("skipped")
                                                                            else "it had problems in the test")
    await task.emit("result", bee=bee, service="Game", title=f"{name} is ready to play", badge="Private",
                    details=[["Play", d["url"]], ["Who can open it", "only people you give the link to: it isn't listed or public"],
                             ["Controls", "WASD or arrows to drive, Space to brake, C camera, N night, R reset; touch buttons on phones"],
                             ["Tested", tested], ["In Vortex OS", f"{bee}'s computer, {folder}: open its desktop to edit the code"]])
    await task.answer(f"**[Play {name}]({d['url']})**: {bee} made it, played it and it {'runs' if result.get('ok') else 'is ready'}. "
                      + " ".join(note) + " Say what to change (more cars, a different city, a story, police chases …) and I'll update it.")
