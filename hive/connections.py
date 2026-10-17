"""Connected apps: getting Blinkit (and the other delivery apps) ready, and saying plainly what's missing when they aren't.

Blinkit, Zepto, Instamart and the like show nothing until you've told them where to deliver, and need a sign-in at checkout.
So before a Bee shops, it checks the page and, if something is missing, asks you for it right in the chat:

  "connect blinkit"            the Bee opens Blinkit, sets your delivery place (you type it in a box) and, if you want,
                               signs you in (phone number, then the OTP you get; the OTP box is a secret box)
  "is blinkit ready?"          opens it, checks, and says what's missing
  "disconnect blinkit"         forgets that it was set up (the Bee's browser keeps its own cookies; clear them in its Cell)
  "add KitKat in blinkit"      does the same check first, then shops

None of this needs an AI model: it reads the page's boxes and buttons and fills them. What you type goes straight into the
site; the phone number and OTP are never saved, shown to a model or kept in the chat. All that's remembered (data/connections.json)
is that the app is set up, where it delivers, and when.
Blocked by the site (a captcha, "access denied": common from a server's network address) or a page that never draws itself
is detected and explained, with a screenshot, instead of the Bee going round in circles.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from . import brain
from .core import Task

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "connections.json"

# Apps that show nothing (or can't be ordered from) until a delivery place is set.
SETUP = ("blinkit", "zepto", "instamart", "bigbasket", "jiomart", "swiggy", "zomato", "dmart", "licious", "countrydelight",
         "dominos", "pizzahut", "kfc", "mcdonalds")
POPULAR = ("blinkit", "zepto", "instamart", "bigbasket", "swiggy", "amazon", "flipkart", "meesho", "myntra")   # "connect all" does these
FIRST = POPULAR + ("zomato", "jiomart", "nykaa", "ajio", "dmart")   # shown first on the Apps page

BLOCKED = re.compile(r"access denied|you don'?t have permission to access|request blocked|unusual traffic|are you a (?:robot|human)|"
                     r"verify you are (?:a )?human|captcha|attention required|error 403|403 forbidden|temporarily blocked|"
                     r"sorry,? you have been blocked|bot detected", re.I)
LOC_WORDS = re.compile(r"location|deliver|address|pin\s*code|pincode|area|locality|city", re.I)
LOC_PROMPT = re.compile(r"(?:select|choose|enter|set|detect|change|search)\s+(?:your\s+)?(?:delivery\s+)?(?:location|address|area|pin\s*code)|"
                        r"where\s+(?:should|do)\s+we\s+deliver|delivery\s+location|detect\s+my\s+location|use\s+current\s+location", re.I)
NOT_A_PLACE = re.compile(r"detect|current|my location|use my|gps|cancel|close|log\s*in|sign\s*in|sign\s*up|skip|later|dismiss|download|app store|google play", re.I)
LOGIN_BTN = re.compile(r"^\s*(?:log\s*in|login|sign\s*in)\b", re.I)
ACCOUNT = re.compile(r"^\s*(?:my\s+)?(?:account|profile|orders?|my orders)\b|\blogout\b|\bsign\s*out\b|\blog\s*out\b", re.I)
GO_ON = re.compile(r"^\s*(?:continue|next|proceed|send\s*otp|get\s*otp|request\s*otp|submit|login|log\s*in|verify|confirm)\b", re.I)
CONFIRM_PLACE = re.compile(r"confirm|continue|proceed|save|done|deliver\s+here|use\s+this|set\s+location|apply", re.I)
OTP_PROMPT = re.compile(r"\botp\b|verification\s+code|enter\s+(?:the\s+)?code|one[- ]time|6[- ]digit|4[- ]digit", re.I)


# ---------------------------------------------------------------- what's remembered
def _read() -> dict:
    try:
        return json.loads(FILE.read_text())
    except (OSError, ValueError):
        return {}


def _write(data: dict) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.chmod(0o600)
    tmp.replace(FILE)


def get(uid: str) -> dict:
    return _read().get(uid, {})


def mark(uid: str, app: str, **fields) -> None:
    data = _read()
    data.setdefault(uid, {})[app] = {**data.get(uid, {}).get(app, {}), **fields, "at": int(time.time())}
    _write(data)


def forget(uid: str, app: str) -> bool:
    data = _read()
    had = bool(data.get(uid, {}).pop(app, None))
    _write(data)
    return had


# ---------------------------------------------------------------- reading the page
def _is_input(e: dict) -> bool:
    k = e.get("kind", "")
    return (k.startswith(("input", "textarea")) or k.startswith("div:editable")) and not k.endswith((":checkbox", ":radio", ":button", ":submit", ":hidden", ":file"))


def _find(obs: dict, pattern: re.Pattern | str, inputs: bool | None = None, after: int = -1) -> int | None:
    rx = re.compile(pattern, re.I) if isinstance(pattern, str) else pattern
    for i, e in enumerate(obs.get("elements") or []):
        if i <= after or (inputs is not None and _is_input(e) != inputs):
            continue
        if rx.search(e.get("label") or ""):
            return i
    return None


def classify(obs: dict) -> str:
    """What this page is waiting for: blocked | empty | location | signed_out | ready."""
    text = f"{obs.get('title', '')} {obs.get('text', '')}"
    els = obs.get("elements") or []
    if BLOCKED.search(text) and len(els) < 12:
        return "blocked"
    if len(els) < 2 and len(text.strip()) < 40:
        return "empty"
    has_search = any(_is_input(e) and re.search(r"search", e["label"] or "", re.I) and not LOC_WORDS.search(e["label"] or "") for e in els)
    loc_input = any(_is_input(e) and LOC_WORDS.search(e["label"] or "") for e in els)
    if loc_input or (LOC_PROMPT.search(text) and not has_search):     # a "search delivery location" box means the place is being asked for
        return "location"
    if any(LOGIN_BTN.search(e["label"] or "") for e in els) and not any(ACCOUNT.search(e["label"] or "") for e in els):
        return "signed_out"
    return "ready"


ADVICE = {
    "blocked": "The site refused this browser (a captcha or “access denied”). That's common from a server's network address; "
               "run the Hive on a home connection, or open Watch live and solve it yourself.",
    "empty": "The page never drew itself. It may need a full browser (Chromium) or a better connection.",
}


# ---------------------------------------------------------------- doing the steps
async def _settle(surface, seconds: float = 1.6) -> dict:
    await asyncio.sleep(seconds)
    return await surface.observe()


async def set_location(surface, place: str) -> str:
    """Type the place into the location box, pick the first matching suggestion, confirm. Returns the page's state afterwards."""
    obs = await surface.observe()
    box = _find(obs, LOC_WORDS, inputs=True)
    if box is None:
        box = next((i for i, e in enumerate(obs["elements"]) if _is_input(e)), None)
    if box is None:
        return "no_box"
    await surface.type(box, place)
    obs = await _settle(surface)
    words = [w for w in re.findall(r"\w{3,}", place.lower())]
    cands = [(i, e) for i, e in enumerate(obs["elements"]) if i > box and not _is_input(e) and (e["label"] or "").strip() and not NOT_A_PLACE.search(e["label"])]
    pick = next((i for i, e in cands if any(w in e["label"].lower() for w in words)), cands[0][0] if cands else None)
    if pick is not None:
        await surface.click(pick)
        obs = await _settle(surface)
    ok = _find(obs, CONFIRM_PLACE, inputs=False)
    if ok is not None and classify(obs) == "location":
        await surface.click(ok)
        obs = await _settle(surface)
    state = classify(obs)
    return "ready" if state != "location" else "location"


async def sign_in(surface, ask_phone, ask_otp) -> bool:
    """Open the login box, give it the phone number, then the OTP. The numbers come from the user's boxes."""
    obs = await surface.observe()
    btn = _find(obs, LOGIN_BTN, inputs=False)
    if btn is not None:
        await surface.click(btn)
        obs = await _settle(surface)
    phone_box = _find(obs, r"phone|mobile|number|\+91", inputs=True)
    if phone_box is None:
        phone_box = next((i for i, e in enumerate(obs["elements"]) if _is_input(e)), None)
    if phone_box is None:
        return False
    phone = await ask_phone()
    if not phone:
        return False
    await surface.type(phone_box, phone)
    go = _find(obs, GO_ON, inputs=False, after=phone_box)       # the button below the box, not the page's own Login
    if go is not None:
        await surface.click(go)
    else:
        await surface.press("Enter")
    obs = await _settle(surface, 2.2)
    otp_box = _find(obs, r"otp|code|verification|digit", inputs=True)
    if otp_box is None and OTP_PROMPT.search(obs.get("text", "")):
        otp_box = next((i for i, e in enumerate(obs["elements"]) if _is_input(e)), None)
    if otp_box is None:
        return False
    otp = await ask_otp()
    if not otp:
        return False
    await surface.type(otp_box, otp)                       # boxes that show one digit each move on by themselves as it types
    obs = await _settle(surface, 1.2)
    go = _find(obs, GO_ON, inputs=False, after=otp_box)
    if go is not None and OTP_PROMPT.search(obs.get("text", "")):
        await surface.click(go)
    obs = await _settle(surface, 2.5)
    return classify(obs) != "signed_out" and not OTP_PROMPT.search(obs.get("text", ""))


async def _shot(task: Task, surface, bee: str, title: str, live: str | None) -> None:
    from .apps import _shot_id
    try:
        img, mime = await surface.shot()
    except Exception:
        return
    sid = _shot_id(task.user, img, mime)
    if sid:
        await task.emit("screen", bee=bee, title=title, shot=sid, live=live)


async def ensure_ready(task: Task, surface, key: str, bee: str, live: str | None, bee_id: str = "", interactive: bool = True, notify=None) -> bool:
    """Check the app's page; set the delivery place (and sign in, if you want) when it's asking. False = it can't go on, and why was said."""
    from .apps import APPS
    from .cloud_chat import ask
    app = APPS[key]
    say = notify or task.answer
    name = app["name"]
    if getattr(surface, "kind", "") != "web" or type(surface).__name__ == "ReaderSurface":
        await say(f"{name} is a modern web app that only works in a full browser (Chromium), and this machine has none (the Bee "
                          f"can only read it as plain text). Install it with `playwright install --with-deps chromium`, or run the Hive "
                          f"on a computer that has it. Until then {name} can't be used from here.")
        return False
    s = await task.step("reading", f"Checking what {name} needs", bee=bee)
    obs = await surface.observe()
    state = classify(obs)
    if state == "empty":
        obs = await _settle(surface, 3.5)
        state = classify(obs)
    await task.finish_step(s, text={"ready": "ready to shop", "location": "needs a delivery place", "signed_out": "ready, not signed in",
                                    "blocked": "blocked by the site", "empty": "the page didn't draw"}[state])
    if state in ADVICE:
        await _shot(task, surface, bee, f"{name} didn't open properly", live)
        await say(f"{name} isn't ready: {ADVICE[state]}" + (" Open Watch live to look." if live else ""))
        return False
    if state == "location":
        if not interactive:
            await say(f"{name} needs a delivery place. Say “connect {name.lower()}” and tell the Bee where to deliver.")
            return False
        for attempt in range(2):
            await _shot(task, surface, bee, f"{name} is asking where to deliver", live)
            reply = await ask(task, "clarify", title=f"Where should {name} deliver?", service=name, ask_anyway=True,
                              text=f"{name} won't show products until it knows the delivery place. Type your area, street or pincode; the Bee will "
                                   f"pick it for you." + (" That one wasn't found; try the area or pincode." if attempt else ""),
                              fields=[{"name": "place", "label": "Area, street or pincode", "required": True, "input": "text"}],
                              actions=[{"id": "go", "label": "Set location"}, {"id": "stop", "label": "Not now", "style": "danger"}])
            place = str((reply.get("values") or {}).get("place") or "").strip()
            if reply["action"] != "go" or not place:
                await say(f"Okay. {name} still needs a delivery place before it can show anything.")
                return False
            s = await task.step("running", f"Setting {name}'s delivery place", bee=bee)
            result = await set_location(surface, place)
            await task.finish_step(s, status="done" if result == "ready" else "error", text="set" if result == "ready" else "it didn't take")
            if result == "ready":
                mark(task.user, key, bee=bee_id, location=place[:80], signed_in=False)
                await _shot(task, surface, bee, f"{name} delivers to {place[:40]}", live)
                state = "signed_out" if classify(await surface.observe()) == "signed_out" else "ready"
                break
        else:
            await _shot(task, surface, bee, f"{name} still asks for the place", live)
            await say(f"I couldn't set the delivery place on {name}. Open Watch live to pick it yourself, then say “add … in {name.lower()}” again.")
            return False
    if state == "signed_out" and interactive and not get(task.user).get(key, {}).get("skip_login"):
        reply = await ask(task, "confirm", title=f"Sign in to {name}?", service=name, ask_anyway=True,
                          text=f"You can sign in now (phone number, then an OTP you'll get), or later when you check out. Your number and the OTP "
                               f"are typed straight into {name}; they aren't saved or shown to any model.",
                          actions=[{"id": "go", "label": "Sign in now"}, {"id": "skip", "label": "Later, at checkout"}])
        if reply["action"] == "go":
            async def ask_phone() -> str:
                r = await ask(task, "clarify", title=f"Your {name} number", service=name, ask_anyway=True,
                              text="The mobile number for your account.",
                              fields=[{"name": "phone", "label": "Mobile number", "required": True, "input": "text"}],
                              actions=[{"id": "go", "label": "Send OTP"}, {"id": "stop", "label": "Cancel", "style": "danger"}])
                digits = re.sub(r"\D", "", str((r.get("values") or {}).get("phone") or ""))[-10:]
                return digits if r["action"] == "go" and len(digits) == 10 else ""

            async def ask_otp() -> str:
                await _shot(task, surface, bee, f"{name} is waiting for the OTP", live)
                r = await ask(task, "clarify", title=f"{name} OTP", service=name, ask_anyway=True,
                              text="Type the code you just got by SMS. It goes straight into the site.",
                              fields=[{"name": "otp", "label": "OTP", "required": True, "input": "password", "secret": True}],
                              actions=[{"id": "go", "label": "Verify"}, {"id": "stop", "label": "Cancel", "style": "danger"}])
                return re.sub(r"\D", "", str((r.get("values") or {}).get("otp") or "")) if r["action"] == "go" else ""

            s = await task.step("running", f"Signing in to {name}", bee=bee)
            ok = await sign_in(surface, ask_phone, ask_otp)
            await task.finish_step(s, status="done" if ok else "error", text="signed in" if ok else "didn't finish")
            await _shot(task, surface, bee, f"{name}: " + ("signed in" if ok else "sign-in didn't finish"), live)
            if ok:
                mark(task.user, key, bee=bee_id, signed_in=True)
            else:
                await task.step("error", "The sign-in didn't finish", bee=bee,
                                detail=f"You can still shop; sign in at checkout, or say “connect {name.lower()}” to try again.")
        else:
            mark(task.user, key, bee=bee_id, skip_login=True)
    if not get(task.user).get(key):
        mark(task.user, key, bee=bee_id, signed_in=state == "ready")
    return True


# ---------------------------------------------------------------- the chat: "connect blinkit"
def _app_key(text: str) -> str | None:
    from .apps import _app_in
    from .apps import APPS
    k = _app_in(text)
    return k if k in APPS else None


CONNECT = re.compile(r"\b(?:connect|set\s*up|setup|link|sign\s*in\s+to|log\s*in\s+to|login\s+to|set\s+(?:my\s+)?(?:delivery\s+)?(?:location|address)|"
                     r"change\s+(?:my\s+)?(?:delivery\s+)?(?:location|address)|update\s+(?:my\s+)?(?:delivery\s+)?(?:location|address))\b", re.I)
STATUS = re.compile(r"\b(?:is|are)\s+(?:\w+\s+){0,3}(?:ready|connected|working|set\s*up)\b|\bstatus\b|\b(?:check|test)\s+(?:the\s+)?(?:\w+\s+)?connection\b|"
                    r"\bconnection\s+(?:status|ready)\b", re.I)
SHOP = re.compile(r"\b(?:add|put|order|buy|purchase|search|find|look\s+up|get\s+me|remove|reorder|track|cancel)\b", re.I)
ALL = re.compile(r"\bconnect\s+(?:all|everything|my\s+apps|all\s+(?:my\s+|the\s+)?apps|them\s+all)\b|\bset\s*up\s+(?:all|everything|my\s+apps)\b", re.I)
DISCONNECT = re.compile(r"\b(?:disconnect|unlink|forget|remove\s+(?:my\s+)?(?:connection|account|login))\b", re.I)


def parse(text: str) -> dict | None:
    t = " ".join((text or "").split())
    if ALL.search(t):
        return {"app": None, "action": "connect_all"}
    key = _app_key(t)
    if not key:
        return None
    if DISCONNECT.search(t):
        return {"app": key, "action": "disconnect"}
    if STATUS.search(t):
        return {"app": key, "action": "status"}
    if SHOP.search(t):                                       # "login to zomato and order biryani": a shopping request, apps.py does the whole thing
        return None
    if CONNECT.search(t):
        return {"app": key, "action": "connect"}
    return None


async def connect_all(task: Task) -> None:
    """One go through the popular apps: each opens in the Bee's browser and asks for what it needs (place, sign-in), all skippable."""
    from . import apps
    from .reader import explain_browser
    b = apps._pick_bee(task)
    if not b:
        await task.answer("A Bee does this in its own browser. Make one first (New Bee in Peace, or “make a bee called Shopper that orders my "
                          "groceries”), then say “connect all”.")
        return
    done, notes = [], []

    async def note(text: str) -> None:
        notes.append(text)
    for key in POPULAR:
        name = apps.APPS[key]["name"]
        s = await task.step("opening", f"Opening {name} in {b['name']}'s browser", bee=b["name"])
        try:
            surface, live, s = await apps.open_web(task, b, apps.APPS[key]["web"], name, b["name"], s)
        except Exception as exc:
            await task.finish_step(s, status="error", text=explain_browser(str(exc))[:160])
            notes.append(f"{name}: couldn't open it ({explain_browser(str(exc))})")
            continue
        await task.finish_step(s)
        before = len(notes)
        if await ensure_ready(task, surface, key, b["name"], live, b["id"], interactive=True, notify=note):
            rec = get(task.user).get(key, {})
            done.append(name + (f" (delivers to {rec['location']})" if rec.get("location") else "") + (", signed in" if rec.get("signed_in") else ""))
        elif len(notes) == before:
            notes.append(f"{name}: skipped")
    lines = (["**Connected:** " + "; ".join(done)] if done else []) + (["**Not connected:**\n" + "\n".join(f"- {n}" for n in notes)] if notes else [])
    await task.emit("result", bee=b["name"], service="Apps", title=f"{len(done)} of {len(POPULAR)} apps connected", badge="Connected" if done else "Report",
                    details=[["Connected", ", ".join(d.split(" (")[0] for d in done) or "none"]] + [["Not connected", n[:140]] for n in notes[:6]])
    await task.answer("\n\n".join(lines) or "Nothing was connected.")


async def handle(task: Task, intent: dict) -> None:
    from . import apps
    if intent["action"] == "connect_all":
        await connect_all(task)
        return
    key, action = intent["app"], intent["action"]
    name = apps.APPS[key]["name"]
    if action == "disconnect":
        had = forget(task.user, key)
        await task.answer(f"{name} is no longer marked as set up" + ("." if had else " (it wasn't).")
                          + " The Bee's browser keeps its own cookies; to sign out of the site, clear them in the Bee's Cell.")
        return
    b = apps._pick_bee(task)
    if not b:
        await task.answer(f"A Bee does this in its own browser. Make one first (New Bee in Peace, or “make a bee called Shopper that orders my "
                          f"groceries”), then say “connect {name.lower()}”.")
        return
    s = await task.step("opening", f"Opening {name} in {b['name']}'s browser", bee=b["name"])
    try:
        surface, live, s = await apps.open_web(task, b, apps.APPS[key]["web"], name, b["name"], s)
    except Exception as exc:
        from .reader import explain_browser
        await task.finish_step(s, status="error", text=explain_browser(str(exc))[:200])
        await task.answer(f"I couldn't open {name}: {explain_browser(str(exc))}.")
        return
    await task.finish_step(s)
    ok = await ensure_ready(task, surface, key, b["name"], live, b["id"], interactive=action == "connect")
    if not ok:
        return
    rec = get(task.user).get(key, {})
    await task.emit("result", bee=b["name"], service=name, title=f"{name} is ready", badge="Connected",
                    details=[["Delivers to", rec.get("location") or "set on the site"], ["Signed in", "yes" if rec.get("signed_in") else "not yet (at checkout)"],
                             ["Bee", b["name"]], ["Next", f"“add KitKat in {name.lower()}”, “order 2 milk from {name.lower()}” …"]])
    model = "" if brain.is_connected() else " No AI model is running on this Hive yet, so a Bee can't shop by itself until the owner sets one up."
    await task.answer(f"{name} is ready in {b['name']}'s browser" + (f", delivering to {rec['location']}" if rec.get("location") else "")
                      + f". Try “add KitKat in {name.lower()}”." + model)


# ---------------------------------------------------------------- for the Apps page
def status(uid: str, bees: list[dict]) -> dict:
    from .apps import APPS
    from .browser import find_chromium
    have = get(uid)
    order = [k for k in FIRST if k in APPS] + sorted((k for k in APPS if k not in FIRST), key=lambda k: APPS[k]["name"].lower())
    apps = []
    for key in order:
        rec = have.get(key) or {}
        apps.append({"key": key, "name": APPS[key]["name"], "web": APPS[key]["web"], "connected": bool(rec), "location": rec.get("location") or "",
                     "signed_in": bool(rec.get("signed_in")), "at": rec.get("at") or 0, "bee": rec.get("bee") or "", "popular": key in FIRST})
    chrome = find_chromium()
    return {"checks": [
        {"id": "bee", "ok": bool(bees), "label": "A Bee", "detail": bees[0]["name"] if bees else "Make a Bee first: it does the shopping in its own browser."},
        {"id": "browser", "ok": bool(chrome), "label": "A browser for the Bee",
         "detail": "Chromium is installed." if chrome else "No Chromium on this machine. Run: playwright install --with-deps chromium"},
        {"id": "model", "ok": brain.is_connected(), "label": "An AI model",
         "detail": "Connected." if brain.is_connected() else "None connected: the Bee can set up Blinkit (location, sign-in) but can't pick items by itself. The Hive's owner sets one up in Settings."},
        {"id": "place", "ok": any(a["connected"] for a in apps), "label": "A delivery place",
         "detail": next((f"{a['name']}: {a['location']}" for a in apps if a["connected"] and a["location"]), "Not set yet: say “connect blinkit”.")},
    ], "apps": apps}
