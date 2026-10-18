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

SPEC = """You write browser games in JavaScript with three.js. You can do this: the Hive runs whatever you write, so never refuse,
never explain, never list steps. Write exactly the game asked for (a train game is about trains, a racing game about racing:
never turn the request into a different game). If it names a famous game, make an ORIGINAL game in that style with your own
names and art from shapes and colours; copy nothing.

Reply with ONE ```js code block: the complete game.js, nothing else (the Hive supplies the HTML page and the canvas
with id "game"). Keep it short and working: about 120 lines, every variable defined, no placeholders.
First line: // CONTROLS: <keys and taps in one line>. Last line: window.__game = { ready: true };

Start like this and build the game inside it:
import * as THREE from "three";
import { createStage } from "./gfx.js";
const stage = createStage(document.getElementById("game"), { time: 15 });   // sky, sun, fog, shadows, bloom: done for you
// stage.scene, stage.camera, stage.update(dt), stage.follow(vec3), stage.render(dt), stage.setTime(0-24), stage.setRain(bool)
// also from "./gfx.js": Particles, makeWater, asphaltTextures, concreteTextures, buildCar (cars only), makeAudio, rng, lerp
// loop: requestAnimationFrame(f); dt = clock.getDelta(); move things; stage.update(dt); stage.render(dt)

Use MeshStandardMaterial. Keyboard controls plus on-screen buttons for phones, a score on screen, and a goal to reach.
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


async def write_game(task: Task, goal: str, current: dict, errors: list[str] | None = None, step: str | None = None,
                     change: str | None = None) -> dict:
    glbs = [p for p, v in current.items() if isinstance(v, bytes) and p.endswith(".glb")]
    models = ("\n\n3D models the user made for this game (glTF, y up, front faces +z, real size in metres); load them with "
              "GLTFLoader from 'three/addons/loaders/GLTFLoader.js' and use them: " + ", ".join(glbs)) if glbs else ""
    own = {p: v for p, v in current.items() if isinstance(v, str) and p not in KIT}
    if change and own:
        listing = "\n".join(f"=== FILE: {p} ===\n{v}\n=== END ===" for p, v in own.items())
        ask = (f"Here is the game, as it is now. Change it like this: {change}\nKeep everything else exactly as it is and keep it working; "
               f"it is still: {goal}\nWrite every file again with the change made.{models}\n\n{listing}")
    elif errors and own:
        listing = "\n".join(f"=== FILE: {p} ===\n{v}\n=== END ===" for p, v in own.items())
        ask = (f"The game below was played in a browser and broke with these errors:\n" + "\n".join(f"- {e}" for e in errors[:8])
               + f"\n\nFix them. The game should still be: {goal}\nWrite every file again, fixed.{models}\n\n{listing}")
    else:
        ask = f"Make this game, from scratch: {goal}{models}"
    msgs = [{"role": "system", "content": SPEC}, {"role": "user", "content": ask}]
    raw = await _stream(task, msgs, step, temperature=0.5)
    files = salvage(raw)
    if not files:   # a small model refused, rambled or lost the format: ask once more, for one file only
        s = await task.step("writing", "Asking the model again, for just the game code")
        raw = await _stream(task, msgs + [{"role": "assistant", "content": (raw or "")[:1500]}, {"role": "user", "content": AGAIN}],
                            s, temperature=0.3)
        files = salvage(raw)
        await task.finish_step(s, "done" if files else "failed")
    if not files:
        raise sites.SiteError("the model didn't write the game's code (it answered without a game.js)")
    keep = {p: v for p, v in current.items() if (isinstance(v, bytes) or p in KIT) and p not in files}
    return {**keep, **files}


async def _stream(task: Task, msgs: list[dict], step: str | None, temperature: float) -> str:
    """The model writes the game piece by piece (a slow model on a phone takes minutes, and a streamed answer doesn't time out
    while it's still writing). The step shows how far it has got and how long since its last words, and a model that stops
    in the middle of the code is told to carry on from there."""
    import time
    seen = {"lines": 0, "chars": 0, "at": time.monotonic()}
    t0 = time.monotonic()

    async def show() -> None:
        now = time.monotonic()
        quiet = now - seen["at"]
        text = (f"Writing the game: {seen['lines']} lines so far ({seen['chars'] / 3 / max(now - t0, 1):.1f} tokens a second)"
                + (f", thinking for {int(quiet)} s" if quiet > 6 else ""))
        if step:
            await task.emit("step", id=step, text=text)

    async def progress(piece: str) -> None:
        seen["lines"] += piece.count("\n")
        seen["chars"] += len(piece)
        seen["at"] = time.monotonic()
        await show()

    async def beat() -> None:   # a slow phone can be quiet for a minute: keep the counter moving
        while True:
            await asyncio.sleep(5)
            await show()
    ticker = asyncio.create_task(beat())
    try:
        raw = await brain.brain.chat(msgs, task.effort, max_tokens=3500, temperature=temperature, on_text=progress)
        for _ in range(CONTINUES):
            if not unfinished(raw):
                break
            more = await brain.brain.chat(
                msgs + [{"role": "assistant", "content": raw}, {"role": "user", "content": CONTINUE}],
                task.effort, max_tokens=3000, temperature=temperature, on_text=progress)
            more = re.sub(r"^\s*```[\w.+-]*\n", "", more)
            if not more.strip():
                break
            raw = raw.rstrip() + "\n" + more
        return raw
    finally:
        ticker.cancel()


CONTINUES = 3
CONTINUE = ("You stopped before the game was finished. Carry on from exactly where you stopped: write only the rest of the "
            "code (no repeating, no explanation), then end with window.__game = { ready: true };")


def unfinished(raw: str) -> bool:
    """True when the code stops in the middle: brackets still open, or it never says it's ready and never draws anything."""
    m = None
    for m in FENCE.finditer(raw or ""):
        pass
    code = m.group("body") if m else (raw or "")
    if "=== FILE" in (raw or "") and "=== END ===" in (raw or "") and raw.rstrip().endswith("=== END ==="):
        return False
    if "game.js" in (raw or "") and "=== FILE" in raw:
        return "=== END ===" not in raw.rsplit("=== FILE", 1)[-1]
    code = re.sub(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`", "", code, flags=re.S)
    depth = sum(code.count(c) for c in "({[") - sum(code.count(c) for c in ")}]")
    if depth > 0:
        return True
    looks_like_code = re.search(r"\b(?:const|let|function|import)\b", code)   # not a refusal or an apology
    return bool(looks_like_code and "__game" not in (raw or "") and not re.search(r"requestAnimationFrame|setAnimationLoop|\.render\(", code))


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


OVERLAY = """<script>(function(){function show(m){var d=document.getElementById("__err");if(!d){d=document.createElement("pre");
d.id="__err";d.style.cssText="position:fixed;left:0;right:0;bottom:0;max-height:45%;overflow:auto;margin:0;padding:10px;z-index:99;background:#300d;color:#fcc;font:12px monospace;white-space:pre-wrap";
document.body.appendChild(d)}d.textContent+=m+"\\n"}addEventListener("error",function(e){show("Error: "+e.message+(e.filename?" ("+e.filename.split("/").pop()+":"+e.lineno+")":""))});
addEventListener("unhandledrejection",function(e){show("Error: "+(e.reason&&e.reason.message||e.reason))})})()</script>"""


def shown(files: dict) -> dict:
    """What's deployed: the game, with a panel that prints any error on the page (a game that breaks is never just blank)."""
    page = files.get("index.html")
    if isinstance(page, str) and "__err" not in page:
        files = {**files, "index.html": page.replace("<head>", "<head>\n" + OVERLAY, 1) if "<head>" in page else OVERLAY + page}
    return files


KIT_EXPORTS = re.compile(r"export\s+(?:async\s+)?(?:function|class|const|let|var)\s+(\w+)")


def autofix(files: dict) -> tuple[dict, list[str]]:
    """Repairs that need no model: a kit function used without being imported (`buildCar is not defined`), or THREE used
    without its import. Returns the files and what was repaired."""
    js = files.get("game.js")
    if not isinstance(js, str):
        return files, []
    kit = set(KIT_EXPORTS.findall((TEMPLATE / "gfx.js").read_text()))
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", js, flags=re.S)
    imported: set[str] = set()
    gfx_import = None
    for m in re.finditer(r"import\s*\{([^}]*)\}\s*from\s*[\"']([^\"']+)[\"'];?", code):
        names = {x.split(" as ")[-1].strip() for x in m.group(1).split(",") if x.strip()}
        imported |= names
        if m.group(2) == "./gfx.js":
            gfx_import = m
    imported |= set(re.findall(r"import\s+(?:\*\s+as\s+)?(\w+)\s+(?:,|from)", code))
    declared = set(re.findall(r"\b(?:const|let|var|function|class)\s+(\w+)", code)) | imported
    need = sorted(n for n in kit if n not in declared and re.search(rf"(?<![\w.$]){n}\s*\(", code))
    fixed = []
    if need:
        if gfx_import:
            old = gfx_import.group(0)
            names = [x.strip() for x in gfx_import.group(1).split(",") if x.strip()]
            js = js.replace(old, "import { " + ", ".join(names + need) + ' } from "./gfx.js";', 1)
        else:
            line = "import { " + ", ".join(need) + ' } from "./gfx.js";\n'
            lines = js.split("\n", 1)
            js = lines[0] + "\n" + line + (lines[1] if len(lines) > 1 else "") if lines[0].startswith("//") else line + js
        fixed.append("added the missing import" + ("s" if len(need) > 1 else "") + " " + ", ".join(need))
    if re.search(r"\bTHREE\.", code) and "THREE" not in declared:
        js = 'import * as THREE from "three";\n' + js
        fixed.append("added the missing import of THREE")
    return ({**files, "game.js": js} if fixed else files), fixed


def _repair(files: dict, did: list) -> dict:
    files, fixed = autofix(files)
    if fixed:
        did.append("Repaired the code before running it: " + "; ".join(fixed))
    return files


def check(files: dict) -> list[str]:
    """Problems found without running the game: broken syntax, names imported from the kit that it doesn't have, no draw loop."""
    import subprocess, tempfile
    js = files.get("game.js")
    if not isinstance(js, str):
        return []
    problems = []
    kit = (TEMPLATE / "gfx.js").read_text()
    have = set(re.findall(r"export\s+(?:async\s+)?(?:function|class|const|let|var)\s+(\w+)", kit))
    for m in re.finditer(r"import\s*\{([^}]*)\}\s*from\s*[\"']\./gfx\.js[\"']", js):
        for n in (x.split(" as ")[0].strip() for x in m.group(1).split(",")):
            if n and n not in have:
                problems.append(f"gfx.js has no export called {n} (it has: {', '.join(sorted(have))})")
    if not re.search(r"requestAnimationFrame|setAnimationLoop|\.render\(", js):
        problems.append("the game never draws: add a loop with requestAnimationFrame that calls stage.update(dt) and stage.render(dt)")
    if "createStage" not in js and "WebGLRenderer" not in js:
        problems.append("the game never creates a scene: use createStage(document.getElementById('game')) from './gfx.js'")
    try:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "game.mjs"
            f.write_text(js)
            r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True, timeout=20)
            if r.returncode:
                line = next((x for x in r.stderr.splitlines() if "Error" in x), r.stderr.strip()[:200])
                problems.append(f"syntax error in game.js: {line[:200]}")
    except (OSError, subprocess.SubprocessError):
        pass   # no node on this machine: the other checks still ran
    return problems


def _why(exc: Exception) -> str:
    """The model's error, in words."""
    import httpx
    msg = str(exc)
    if isinstance(exc, httpx.TimeoutException) or "timed out" in msg.lower():
        return ("the model went quiet for too long (no new words for 10 minutes), so I stopped waiting. A model on a phone "
                "is slow: make sure nothing else heavy is running, or set MNX_MODEL_TIMEOUT higher, or use a faster model")
    if isinstance(exc, httpx.ConnectError):
        return "I couldn't reach the model's server (is it still running?)"
    if not msg.strip():
        return f"the model's server failed ({type(exc).__name__})"
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


ERROR_PASTE = re.compile(r"\b(?:Uncaught\s+)?(?:Reference|Type|Syntax|Range|Eval)Error\b|\bis not (?:defined|a function|a constructor)\b|"
                         r"\bCannot read propert(?:y|ies)\b|\(game\.js:\d+\)|\bUnexpected (?:token|end of input|identifier)\b|"
                         r"\bFailed to (?:resolve module|fetch dynamically)", re.I)
EDIT_WORDS = re.compile(r"^\s*(?:please\s+)?(?:change|update|fix|edit|improve|make|add|remove|replace|put|use|increase|decrease|speed|slow)\b", re.I)


def parse_edit(text: str, history: list | None = None) -> dict | None:
    """An error pasted from the game, or "make it faster": a change to the game made earlier in this chat (or the newest one)."""
    t = " ".join((text or "").split())
    if not t or len(t) > 1500:
        return None
    named = re.search(r"\b(?:my|the)\s+([\w -]{2,40}?)\s+game\b", t, re.I)
    if ERROR_PASTE.search(t) and not re.match(r"\s*(?:what|why|how|explain|tell me|can you|define)\b", t, re.I):
        return {"change": f"it broke in the browser with this error: {t}. Fix the cause in the code.", "error": t, "name": None}
    recent = any(h.get("role") == "bot" and "wrote it from scratch" in (h.get("text") or "") for h in (history or [])[-6:])
    if recent and EDIT_WORDS.search(t) and not re.search(r"\b(?:new|another|different)\b.*\bgame\b", t, re.I):
        return {"change": t, "error": None, "name": slug_of(named.group(1)) if named else None}
    return None


def slug_of(words: str) -> str:
    return sites.slug(words)


def bee_of(row: dict, uid: str) -> dict | None:
    """The Bee that made a Sandbox item (by id; older items by name), if it is this user's."""
    from .bees import registry
    return (registry.mine(row["bee_id"], uid) if row.get("bee_id") else None) or registry.find(row.get("bee", ""), uid)


def _find(task: Task, name: str | None) -> tuple[dict, dict] | None:
    """The game to change: the one named, else the newest made for this user, with the Bee that made it."""
    from . import sandbox
    for r in sandbox.of(task.user):
        if r["kind"] == "game" and (not name or name in r["name"]):
            b = bee_of(r, task.user)
            return (r, b) if b else None
    return None


async def load(b: dict, name: str) -> dict:
    """The saved game's files from the Bee's computer (text as text, 3D models as bytes)."""
    from .cells import cells
    cell = cells.get(b["id"], b["name"], b["computer"])
    if cell.mode == "remote":
        return {}
    root = cell.safe_path(f"games/{name}")
    out: dict = {}
    for f in sorted(root.rglob("*")) if root.is_dir() else []:
        if not f.is_file() or f.name == "chroma.json" or f.stat().st_size > 3_000_000:
            continue
        rel = str(f.relative_to(root))
        out[rel] = f.read_bytes() if f.suffix == ".glb" else f.read_text(errors="replace")
    return out


async def handle_edit(task: Task, edit: dict) -> bool:
    """Changes the game made before instead of writing a new one: the same files, the change made in them, checked, put up
    again at the same name. False when there is no game to change (the message then goes on as usual)."""
    found = _find(task, edit["name"])
    if not found or not brain.is_connected():
        return False
    rec, b = found
    files = await load(b, rec["name"])
    if "game.js" not in files:
        return False
    goal = rec.get("goal") or rec["name"].replace("-", " ")
    kit = kit_files()
    files = {**kit, **{p: v for p, v in files.items() if p not in KIT}}
    bee = b["name"]
    s = await task.step("writing", f"Changing {rec['name']}" if not edit["error"] else f"Fixing {rec['name']}: {edit['error'][:90]}", bee=bee)
    did = [f"Opened the saved game {rec['name']} and changed it (not a new game)"]
    repaired = _repair(files, did)
    if repaired is not files and edit["error"] and "is not defined" in edit["error"] and not check(repaired):
        files = repaired   # the missing import was the whole problem: no need to ask the model
        await task.finish_step(s, text="Repaired the code without the model")
    else:
        try:
            files = _repair(await write_game(task, goal, repaired, change=edit["change"], step=s), did)
            await task.finish_step(s)
        except Exception as exc:
            await task.finish_step(s, status="error", text=_why(exc)[:160])
            await task.answer(f"I couldn't change it: {_why(exc)}. The game is unchanged: {rec['url']}")
            return True
    await ship(task, b, rec["name"], goal, files, [], did)
    return True


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
        files = await write_game(task, goal, files, step=s)
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

    await ship(task, b, name, goal, files, note, did)


async def ship(task: Task, b: dict, name: str, goal: str, files: dict, note: list, did: list) -> None:
    """Everything after the model has written the code: repair what can be repaired, check it, put it up privately, play it,
    fix what broke, save it and tell the user. The answer comes last, after the checks."""
    from .apps import _shot_id
    bee = b["name"]
    files = _repair(files, did)
    for _ in range(FIX_ROUNDS):   # fix what can be seen without running it, before anyone opens a blank page
        found = check(files)
        if not found:
            break
        s = await task.step("writing", "Fixing problems found before running it", bee=bee)
        try:
            files = _repair(await write_game(task, goal, files, found, step=s), did)
            await task.finish_step(s)
        except Exception as exc:
            await task.finish_step(s, status="error", text=_why(exc)[:160])
            break
    stuck = check(files)
    if stuck:
        note.append(f"It may still show a blank page ({stuck[0][:140]}); tell me what you see and I'll fix it.")

    result: dict = {}
    d: dict = {}
    for attempt in range(FIX_ROUNDS + 1):
        s = await task.step("running", "Deploying it privately" if attempt == 0 else f"Deploying fix {attempt}", bee=bee)
        try:
            await sites.chroma.ensure()
            d = await sites.chroma.deploy(task.user, name, shown(files), [], private=True)
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
            files = _repair(await write_game(task, goal, files, problems, step=s), did)
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
    sandbox.record(task.user, "game", name, bee, d["url"], goal=goal[:400], bee_id=b["id"])
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
