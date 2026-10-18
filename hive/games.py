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
READY_WAIT = 25   # seconds a game gets to say window.__game.ready

KIT = ("gfx.js",)   # the graphics kit: the model uses it, never rewrites it, and isn't sent its code (only its API, below)

BASE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>GAME TITLE</title>
<style>html,body{margin:0;height:100%;overflow:hidden;background:#000}canvas{display:block;width:100%;height:100%}</style>
<script type="importmap">
{ "imports": { "three": "/vendor/three/three.module.js", "three/addons/": "/vendor/three/addons/" } }
</script>
<script type="module" src="game.js"></script>
</head>
<body><canvas id="game"></canvas></body>
</html>"""

SPEC = """You make complete, fun browser games with three.js, written from scratch for exactly what the user asked. A train game
is about trains (tracks, stations, signals, passengers), a racing game about racing, a space game about space: never turn the
request into a different game. You can do this: you write the code and the Hive runs
it, so never refuse, never explain how, never list steps: just write the game. Write every file in this exact format, nothing
before or after (if the format is hard, one ```js block with the whole game.js is enough; the Hive adds index.html):

=== FILE: index.html ===
(full content)
=== END ===
=== FILE: game.js ===
(full content)
=== END ===

Rules
- Start index.html from this page and keep its importmap exactly (add your HUD and buttons to the body):
""" + BASE_HTML + """
- No other libraries, no npm, no build step, no files you don't write. Add a style.css file only if you need one.
- The game must run as written: every variable defined, no placeholders, no "TODO", no "..." in the code.
- The first line of game.js is a comment listing the controls: // CONTROLS: <keys and taps, in one line>
- Keyboard controls and on-screen touch buttons (phones), a HUD with the goal and score, and a way to win or progress.
- End game.js with: window.__game = { ready: true };
- If the user names a famous game (GTA, Minecraft …), make an ORIGINAL game in that style: your own title, world, characters
  and art made from shapes and colours. Never copy names, logos, characters, maps, music or text from it.
- Keep it light enough for a phone: instanced meshes for many objects, few lights, modest shadows.

A graphics kit, gfx.js, is next to game.js (never write it, import from it: `import { createStage } from "./gfx.js";`). Use it so
the game looks great:
  const stage = createStage(document.getElementById("game"), { time: 15 })  → { renderer, scene, camera, setTime(hours 0-24),
        setRain(bool), night (0-1), follow(vec3: keeps the shadows around the player), update(dt), render(dt, speed01),
        glow(material, dayI, nightI) (lights up at night) }. Each frame: stage.update(dt); stage.follow(target); stage.render(dt).
        It already has the sky, sun, moon, stars, fog, shadows, bloom and tone mapping: don't add your own sky or renderer.
  Also: Particles(scene, max, {additive}) .spawn(pos, vel, {life,size,grow,color,alpha,gravity}) .update(dt);
        makeWater(size, y) (call water.userData.update(dt)); asphaltTextures(), concreteTextures(), facadeTextures(seed);
        buildCar(stage, {kind, color}) only if the game has cars; makeAudio(); rng(seed); lerp.
Prefer MeshStandardMaterial / MeshPhysicalMaterial (real light and reflections), not Basic or Lambert.
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


def template_files(models: dict[str, bytes] | None = None) -> dict[str, str | bytes]:
    """The starter city game (hive/templates/game_city): an example of the kit in use. Never handed to anyone as their game."""
    files: dict[str, str | bytes] = {p.name: p.read_text() for p in sorted(TEMPLATE.iterdir()) if p.is_file()}
    return with_models(files, models or {})


def kit_files(models: dict[str, bytes] | None = None) -> dict[str, str | bytes]:
    """What every new game starts with: the graphics kit, and the user's own 3D models if they named any. No game code."""
    files: dict[str, str | bytes] = {k: (TEMPLATE / k).read_text() for k in KIT}
    return {**files, **{f"models/{n}.glb": glb for n, glb in (models or {}).items()}}


def with_models(files: dict, models: dict[str, bytes]) -> dict:
    """Put the Bee's .glb models in the game; the starter game drives the vehicle (or the only one) instead of its box car."""
    files = {**files, **{f"models/{n}.glb": glb for n, glb in models.items()}}
    ride = next((n for n in models if re.search(r"car|truck|bike|kart|vehicle|bus|jeep|taxi|van|ship|plane|tank", n)),
                next(iter(models)) if len(models) == 1 else None)
    js = files.get("game.js")
    if ride and isinstance(js, str) and "const PLAYER_MODEL = null;" in js:
        files["game.js"] = js.replace("const PLAYER_MODEL = null;", f'const PLAYER_MODEL = "models/{ride}.glb";')
    return files


async def models_for(b: dict, goal: str) -> dict[str, bytes]:
    """The Bee's 3D models the request names ("with my sports-car model"), or all of them for "my models" (four at most)."""
    from . import models3d
    names = await models3d.names(b)
    if not names or not re.search(r"\bmodels?\b|\.glb\b|\bstudio\b", goal, re.I):
        return {}
    low = sites.slug(goal)
    picked = [n for n in names if n in low or n.replace("-", " ") in goal.lower()]
    if not picked and re.search(r"\bmy\s+(?:3d\s+)?models\b|\ball\s+my\b", goal, re.I):
        picked = names
    out = {}
    for n in picked[:4]:
        glb = await models3d.glb_of(b, n)
        if glb:
            out[n] = glb
    return out


def _brief(task: Task, intent: dict) -> str:
    """What to make: this message, plus what was said about the game earlier in the chat ("make it yourself")."""
    earlier = [h["text"] for h in (getattr(task, "history", None) or []) if h.get("role") == "user" and re.search(r"\bgame\b", h.get("text", ""), re.I)]
    goal = " ".join(earlier[-2:] + [intent["goal"]]) if earlier and len(intent["goal"]) < 120 else intent["goal"]
    if intent["styles"]:
        goal += (f"\n(The user named {', '.join(intent['styles'])}: make an original game in that style with your own title, "
                 "world, characters and missions; copy nothing from it.)")
    return goal


def name_for(goal: str, styles: list[str]) -> str:
    goal = re.sub(r"\b(?:with|using)\s+(?:my|the|these|those)\b.*$", "", goal, flags=re.I)  # "… with my car model"
    words = re.sub(r"\b(?:create|make|build|design|code|develop|generate|me|a|an|the|my|game|games|in|3d|2d|like|not|use|for|and|it|"
                   r"yourself|run|three\.?js|with|of|please)\b", " ", goal.lower())
    for st in styles:
        words = re.sub(re.escape(st.lower()), " ", words)
    words = re.sub(r"\b\d+\b", " ", words)
    name = sites.slug(" ".join(words.split()[:3]))
    return (name or "open-world") + "-game" if not name.endswith("game") else name


async def write_game(task: Task, goal: str, current: dict, errors: list[str] | None = None) -> dict:
    glbs = [p for p, v in current.items() if isinstance(v, bytes) and p.endswith(".glb")]
    models = ("\n\n3D models the user made for this game (glTF, y up, front faces +z, real size in metres); load them with "
              "GLTFLoader from 'three/addons/loaders/GLTFLoader.js' and use them: " + ", ".join(glbs)) if glbs else ""
    own = {p: v for p, v in current.items() if isinstance(v, str) and p not in KIT}
    if errors and own:
        listing = "\n".join(f"=== FILE: {p} ===\n{v}\n=== END ===" for p, v in own.items())
        ask = (f"The game below was played in a browser and broke with these errors:\n" + "\n".join(f"- {e}" for e in errors[:8])
               + f"\n\nFix them. The game should still be: {goal}\nWrite every file again, fixed.{models}\n\n{listing}")
    else:
        ask = f"Make this game, from scratch: {goal}{models}"
    msgs = [{"role": "system", "content": SPEC}, {"role": "user", "content": ask}]
    raw = await brain.brain.chat(msgs, task.effort, max_tokens=7000, temperature=0.5)
    files = salvage(raw)
    if not files:   # a small model refused, rambled or lost the format: ask once more, for one file only
        s = await task.step("writing", "Asking the model again, for just the game code")
        raw = await brain.brain.chat(msgs + [{"role": "assistant", "content": (raw or "")[:1500]}, {"role": "user", "content": AGAIN}],
                                     task.effort, max_tokens=7000, temperature=0.3)
        files = salvage(raw)
        await task.finish_step(s, "done" if files else "failed")
    if not files:
        raise sites.SiteError("the model didn't write the game's code (it answered without a game.js)")
    keep = {p: v for p, v in current.items() if (isinstance(v, bytes) or p in KIT) and p not in files}
    return {**keep, **files}


AGAIN = """You can make this game: you write code, and the Hive runs it. Don't explain, don't refuse, don't describe steps.
Reply with only one ```js code block: the complete game.js for the game I asked for (three.js from "three", the kit from
"./gfx.js"), first line // CONTROLS: ..., last line window.__game = { ready: true };"""

FENCE = re.compile(r"```[ \t]*(?P<lang>[\w.+-]*)[^\n]*\n(?P<body>.*?)(?:```|\Z)", re.S)


def salvage(raw: str) -> dict:
    """The game files in whatever shape a model wrote them: the === FILE === format, fenced code blocks, or one whole page.
    Only game.js is a must; the Hive supplies the standard index.html when the model leaves it out. {} when there's no game."""
    from . import respond
    files, _ = sites.parse_files(raw)
    files = {p: v for p, v in files.items() if p.endswith((".html", ".js", ".css", ".json")) and p not in KIT}
    if "game.js" not in files:
        for m in FENCE.finditer(raw or ""):
            body, lang = m.group("body").strip(), m.group("lang").lower()
            if lang in ("html", "htm") or body.lower().startswith(("<!doctype", "<html")):
                files.setdefault("index.html", body)
            elif lang in ("js", "javascript", "mjs") or (not lang and re.search(r"\bimport\b.+\bfrom\b|THREE\.", body)):
                if len(body) > len(files.get("game.js", "")):   # the longest script is the game, not a one-line snippet
                    files["game.js"] = body
            elif lang == "css":
                files.setdefault("style.css", body)
    page = files.get("index.html", "")
    inline = re.search(r"<script\b[^>]*>\s*\S[\s\S]{200,}?</script>", page) if "game.js" not in files else None
    if "game.js" not in files and not inline:
        return {}
    if "game.js" in files and respond.REFUSAL.search(files["game.js"][:300]):
        return {}
    if not page or ("game.js" in files and "game.js" not in page):
        files["index.html"] = BASE_HTML   # the model's page doesn't load its game: use the standard one
    js = files.get("game.js")
    if js and "window.__game" not in js:
        files["game.js"] = js.rstrip() + "\nwindow.__game = { ready: true };\n"   # runs only if everything above it did
    return files


def _why(exc: Exception) -> str:
    """The model's error, in words."""
    msg = str(exc)
    if "longer than the context" in msg or "context length" in msg or "exceeds the context" in msg:
        return ("the model's memory (its context window) is too small for a whole game. Restart the Hive so the model starts "
                "with a bigger window (MNX_CTX, 16384 by default), or pick a bigger model")
    return msg[:200]


def _controls(files: dict) -> str:
    first = str(files.get("game.js") or "").lstrip().splitlines()[:1]
    m = re.match(r"//\s*CONTROLS:\s*(.+)", first[0]) if first else None
    return m.group(1).strip()[:200] if m else "shown in the game"


async def play(url: str) -> dict:
    """Open the game in a browser for a few seconds: its errors, whether it says it's ready, and a screenshot."""
    import base64
    from .browser import BrowserError, BrowserSession
    port = sites.chroma.port
    b = BrowserSession(allow={f"127.0.0.1:{port}", f"localhost:{port}"})
    try:
        await b.do("goto", url=sites.chroma.url + url)
        ready = False
        for _ in range(READY_WAIT * 2):          # a rich scene takes a while to build on a slow machine: wait until it says it's ready
            await asyncio.sleep(0.5)
            ready = bool((await b.do("eval", js="Boolean(window.__game && window.__game.ready)")).get("value"))
            if ready:
                await asyncio.sleep(1.5)         # let a few frames draw before the picture is taken
                break
        errs = (await b.do("errors")).get("errors") or []
        shot = (await b.do("screenshot")).get("image", "")
        return {"ok": bool(ready) and not errs, "ready": bool(ready), "errors": errs[:10],
                "shot": base64.b64decode(shot.split(",", 1)[1]) if "," in shot else b""}
    except BrowserError as exc:
        if any(w in str(exc) for w in ("couldn't start", "Playwright isn't installed", "can't run a Bee's website browser")):
            return {"ok": None, "errors": [], "shot": b"", "skipped": "this machine has no browser to play-test it"}
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
    if not brain.is_connected():
        await task.answer("A Bee writes every game from scratch with an AI model, and no model is running on this Hive yet. Pick one in the "
                          "model menu (Mnx 1.5 Max Coder if it's installed) or set one up in Settings, then ask again.")
        return
    goal = _brief(task, intent)
    name = name_for(intent["goal"], intent["styles"])
    models = await models_for(b, goal)
    files = kit_files(models)
    note, did = [], []
    if models:
        did.append(f"Put your 3D model{'s' if len(models) > 1 else ''} in it: " + ", ".join(models))
    if intent["unity"]:
        note.append("Unity can't run on the Hive's computers (it needs a desktop with a graphics card, a Unity account and about "
                    "10 GB), so I made it with three.js, which plays in any browser, phones included.")
    if intent["styles"]:
        note.append(f"It's an original game in the style of {', '.join(intent['styles'])}: its own world, names and art, nothing copied.")

    s = await task.step("writing", f"Writing {name} from scratch (three.js)", bee=bee)
    try:
        files = await write_game(task, goal, files)
    except (sites.SiteError, brain.BrainNotConnected) as exc:
        await task.finish_step(s, status="error", text=_why(exc)[:160])
        await task.answer(f"I couldn't make the game: {_why(exc)}. Nothing was made; ask again and I'll write it from scratch.")
        return
    except Exception as exc:  # the model's server said no (too long a prompt, overloaded …)
        await task.finish_step(s, status="error", text=_why(exc)[:160])
        await task.answer(f"I couldn't make the game: {_why(exc)}. Nothing was made.")
        return
    await task.finish_step(s, text=f"{len(files)} files")
    did.append(f"Wrote {name} from scratch in three.js, for what you asked")

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
        if attempt:
            did.append(f"Fixed what broke ({attempt} round{'s' if attempt > 1 else ''}) and put it up again")
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
        if attempt == FIX_ROUNDS:
            note.append(f"It still has a problem after {FIX_ROUNDS} fixes ({problems[0][:120]}); tell me what you see and I'll fix it.")
            break
        s = await task.step("writing", "Fixing what broke", bee=bee)
        try:
            files = await write_game(task, goal, files, problems)
            await task.finish_step(s)
        except Exception as exc:  # keep the last version rather than swapping in anything else
            await task.finish_step(s, status="error", text=_why(exc)[:160])
            note.append(f"The fix didn't come out ({_why(exc)}), so this is my last version.")
            break

    folder = await sites.save_to_bee(b, name, files, d.get("name", name), folder="games")
    did.append(f"Saved it in {bee}'s Vortex OS computer at {folder}")
    did.append("Put it on a private link (not listed or public) and in your Sandbox")
    if result.get("ok"):
        did.append("Played it in its own browser: it runs with no errors")
    from . import sandbox
    sandbox.record(task.user, "game", name, bee, d["url"])
    sid = _shot_id(task.user, result.get("shot") or b"", "image/jpeg")
    if sid:
        await task.emit("screen", bee=bee, title=f"{name} running in {bee}'s browser", shot=sid, live=d["url"])
    tested = ("play-tested in a browser: no errors" if result.get("ok") else
              f"not play-tested ({result['skipped']})" if result.get("skipped") else "it had problems in the test")
    await task.emit("result", bee=bee, service="Game", title=f"{name} is ready to play", badge="Private",
                    details=[["Play", d["url"]], ["Who can open it", "only people you give the link to: it isn't listed or public"],
                             ["Controls", _controls(files)],
                             ["Tested", tested], ["In Vortex OS", f"{bee}'s computer, {folder}: open its desktop to edit the code"]])
    lines = "\n".join(f"- {x}" for x in did)
    await task.answer(f"**[Play {name}]({d['url']})**: {bee} wrote it from scratch and it {'runs' if result.get('ok') else 'is ready'}.\n\n"
                      f"What I did:\n{lines}\n\n" + (" ".join(note) + " " if note else "") + "Tell me what to change and I'll update it.")
