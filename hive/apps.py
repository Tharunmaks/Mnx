"""Bees use any website, and (for the Hive's owner) any app on the paired phone, like a person does.

  "add KitKat in blinkit"            the Bee opens Blinkit in its own browser, finds KitKat and adds it to the cart
  "order 2 milk from zepto"          … up to checkout, then asks you before paying
  "search for running shoes on amazon"
  "open swiggy on my phone and …"    the owner's phone, through the Phone Drone (drone/drone.py)

How it works: the Bee looks at the page (the buttons, links and boxes it can see, and the text), the model picks
one step (click, type, scroll, go to a page …), the Bee does it, and looks again, until the goal is done.

Things only you know come from you, in the chat:
  - a login, an OTP, a password, an API key, a UPI ID, an address: a card with boxes to fill. Secret boxes are
    password fields; what you type goes straight into the website and is never saved, never shown to the model
    (it only sees {{placeholders}}) and never kept in the chat. The Bee's browser stays signed in afterwards, so
    you rarely log in twice.
  - paying, placing an order, booking or sending: a card with the total and a Pay button. The Bee never presses a
    pay / place order / book / send button without your OK in this chat (checked here, not left to the model).
    UPI collect requests are approved in your UPI app; OTPs come back to you as a box to fill.
  - captchas or anything stuck: "Watch live" opens the Bee's browser so you can tap it yourself.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import uuid
from collections import OrderedDict
from urllib.parse import quote_plus

from . import brain
from .core import Task

# Apps the Bees know by name: their website, a search page, and the Android package for the phone.
APPS = {
    "blinkit": {"name": "Blinkit", "web": "https://blinkit.com", "search": "https://blinkit.com/s/?q={q}",
                "package": "com.grofers.customerapp", "words": ["blinkit", "blink it", "grofers"]},
    "zepto": {"name": "Zepto", "web": "https://www.zeptonow.com", "search": "https://www.zeptonow.com/search?query={q}",
              "package": "com.zeptoconsumerapp", "words": ["zepto"]},
    "instamart": {"name": "Swiggy Instamart", "web": "https://www.swiggy.com/instamart", "search": "https://www.swiggy.com/instamart/search?query={q}",
                  "package": "in.swiggy.android", "words": ["instamart", "swiggy instamart"]},
    "swiggy": {"name": "Swiggy", "web": "https://www.swiggy.com", "search": "https://www.swiggy.com/search?query={q}",
               "package": "in.swiggy.android", "words": ["swiggy"]},
    "zomato": {"name": "Zomato", "web": "https://www.zomato.com", "search": None, "package": "com.application.zomato", "words": ["zomato"]},
    "bigbasket": {"name": "BigBasket", "web": "https://www.bigbasket.com", "search": "https://www.bigbasket.com/ps/?q={q}",
                  "package": "com.bigbasket.mobileapp", "words": ["bigbasket", "big basket"]},
    "amazon": {"name": "Amazon", "web": "https://www.amazon.in", "search": "https://www.amazon.in/s?k={q}",
               "package": "in.amazon.mShop.android.shopping", "words": ["amazon"]},
    "flipkart": {"name": "Flipkart", "web": "https://www.flipkart.com", "search": "https://www.flipkart.com/search?q={q}",
                 "package": "com.flipkart.android", "words": ["flipkart"]},
    "myntra": {"name": "Myntra", "web": "https://www.myntra.com", "search": None, "package": "com.myntra.android", "words": ["myntra"]},
    "jiomart": {"name": "JioMart", "web": "https://www.jiomart.com", "search": "https://www.jiomart.com/search/{q}",
                "package": "com.jpl.jiomart", "words": ["jiomart", "jio mart"]},
    "nykaa": {"name": "Nykaa", "web": "https://www.nykaa.com", "search": "https://www.nykaa.com/search/result/?q={q}",
              "package": "com.fsn.nykaa", "words": ["nykaa"]},
    "meesho": {"name": "Meesho", "web": "https://www.meesho.com", "search": "https://www.meesho.com/search?q={q}",
               "package": "com.meesho.supply", "words": ["meesho"]},
    "bookmyshow": {"name": "BookMyShow", "web": "https://in.bookmyshow.com", "search": None, "package": "com.bt.bms",
                   "words": ["bookmyshow", "book my show", "bms"]},
    "makemytrip": {"name": "MakeMyTrip", "web": "https://www.makemytrip.com", "search": None, "package": "com.makemytrip",
                   "words": ["makemytrip", "make my trip", "mmt"]},
    "irctc": {"name": "IRCTC", "web": "https://www.irctc.co.in/nget/train-search", "search": None, "package": "cris.org.in.prs.ima",
              "words": ["irctc"]},
    "uber": {"name": "Uber", "web": "https://m.uber.com", "search": None, "package": "com.ubercab", "words": ["uber"]},
    "ola": {"name": "Ola", "web": "https://book.olacabs.com", "search": None, "package": "com.olacabs.customer", "words": ["ola", "ola cabs"]},
    "rapido": {"name": "Rapido", "web": "https://www.rapido.bike", "search": None, "package": "com.rapido.passenger", "words": ["rapido"]},
    "youtube": {"name": "YouTube", "web": "https://www.youtube.com", "search": "https://www.youtube.com/results?search_query={q}",
                "package": "com.google.android.youtube", "words": ["youtube"]},
    "gmail": {"name": "Gmail", "web": "https://mail.google.com", "search": None, "package": "com.google.android.gm", "words": ["gmail"]},
    "whatsapp": {"name": "WhatsApp", "web": "https://web.whatsapp.com", "search": None, "package": "com.whatsapp", "words": ["whatsapp"]},
}
VERB = (r"(?:add|put|order|buy|purchase|book|reserve|search(?:\s+for)?|find|look\s+up|get\s+me|get|check|track|cancel|reorder|"
        r"pay(?:\s+for)?|log\s*in|sign\s*in|open|play|send|message|post|subscribe|apply|remove|empty|show\s+me|compare)")
VERB_RE = re.compile(rf"\b{VERB}\b", re.I)
DOMAIN_RE = re.compile(r"\b((?:https?://)?(?:[a-z0-9-]+\.)+(?:com|in|co\.in|org|net|io|app|shop|store)(?:/[^\s]*)?)", re.I)
PHONE_RE = re.compile(r"\b(?:on|using|with|from)\s+my\s+phone\b|\bphone'?s?\b.*\bapp\b|\bin\s+the\s+[\w ]{2,20}\s+app\b", re.I)
ITEM_RE = re.compile(rf"\b{VERB}\s+(?:me\s+)?(?P<item>.+?)\s+(?:in|on|from|to|at|using|via)\s+(?:the\s+)?(?:my\s+)?(?P<where>[\w .-]+?)"
                     r"(?:\s+(?:app|cart|website|site|basket))?\s*[.!?]?$", re.I)

# Buttons the Bee never presses without your OK in this chat: they spend money or reach other people.
RISKY = re.compile(r"\b(?:pay|place\s+order|buy\s+now|confirm\s+(?:order|booking|payment)|proceed\s+to\s+pay|book\s+now|"
                   r"complete\s+(?:order|purchase|payment)|make\s+payment|send|submit\s+order|checkout\s+&\s+pay|order\s+now|"
                   r"slide\s+to\s+pay|swipe\s+to\s+pay)\b", re.I)
MONEY = re.compile(r"₹\s?([\d,]+(?:\.\d{1,2})?)")
STEPS = {"low": 15, "med": 25, "high": 40, "ultra": 60, "maxxxx": 60}
SHOTS: "OrderedDict[str, tuple[str, bytes, str]]" = OrderedDict()  # id → (user, image, mime); the newest 300
MAX_SHOTS = 300


# ---------------------------------------------------------------- what the user asked for
def _app_in(text: str) -> str | None:
    low = text.lower()
    for key, a in APPS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", low) for w in a["words"]):
            return key
    return None


def parse(text: str) -> dict | None:
    """ "add KitKat in blinkit" → {"app": "blinkit", "item": "KitKat", "url": None, "phone": False, "goal": …}"""
    t = " ".join((text or "").split())
    if not t or not VERB_RE.search(t):
        return None
    from .cloud_chat import CODE_ASK
    if CODE_ASK.search(t):  # "write a script to scrape amazon", "how do I order on swiggy" are questions for the model
        return None
    app = _app_in(t)
    dm = DOMAIN_RE.search(t)
    if not app and not dm:
        return None
    if not app and re.search(r"@", t):  # an email address isn't a website
        return None
    item = None
    m = ITEM_RE.search(t)
    if m and (app is None or _app_in(m.group("where")) == app or (dm and dm.group(1) in m.group("where"))):
        item = re.sub(r"^(?:a|an|the|some)\s+", "", m.group("item").strip(" .,'\""), flags=re.I) or None
    url = None
    if dm and not app:
        url = dm.group(1) if dm.group(1).lower().startswith("http") else "https://" + dm.group(1)
    return {"app": app, "item": item, "url": url, "phone": bool(PHONE_RE.search(t)), "goal": t}


def start_url(intent: dict) -> str:
    a = APPS.get(intent.get("app") or "")
    if a:
        if intent.get("item") and a.get("search"):
            return a["search"].format(q=quote_plus(intent["item"]))
        return a["web"]
    return intent.get("url") or "https://www.google.com"


# ---------------------------------------------------------------- where the Bee works: a browser, or the phone
class WebSurface:
    """The Bee's own Chromium (it stays signed in between tasks)."""
    kind = "web"

    def __init__(self, session):
        self.s = session
        self.els: list[dict] = []

    async def open(self, url: str) -> None:
        await self.s.do("goto", url=url)

    async def observe(self) -> dict:
        st = await self.s.do("elements")
        self.els = st.get("elements") or []
        try:
            text = (await self.s.do("read")).get("text") or ""
        except Exception:
            text = ""
        return {"place": st.get("url", ""), "title": st.get("title", ""), "text": text[:2500],
                "elements": [{"label": e.get("label") or "", "kind": e.get("tag", "") + (f":{e['type']}" if e.get("type") else "")}
                             for e in self.els]}

    def _el(self, i) -> dict:
        try:
            return self.els[int(i)]
        except (ValueError, TypeError, IndexError) as exc:
            raise ValueError(f"There's no element {i}") from exc

    async def click(self, i) -> None:
        e = self._el(i)
        await self.s.do("click", x=e["x"], y=e["y"])

    async def type(self, i, text: str, enter: bool = False) -> None:
        if i is not None:
            await self.click(i)
            await self.s.do("press", key="Control+A")
            await self.s.do("press", key="Backspace")
        await self.s.do("type", text=text)
        if enter:
            await self.s.do("press", key="Enter")

    async def press(self, key: str) -> None:
        await self.s.do("press", key={"enter": "Enter", "back": "Escape", "tab": "Tab"}.get(key.lower(), key))

    async def scroll(self, down: bool = True) -> None:
        await self.s.do("scroll", dy=600 if down else -600)

    async def back(self) -> None:
        await self.s.do("back")

    async def goto(self, url: str) -> None:
        await self.s.do("goto", url=url)

    async def shot(self) -> tuple[bytes, str]:
        st = await self.s.do("screenshot")
        data = st.get("image", "")
        return base64.b64decode(data.split(",", 1)[1]) if "," in data else b"", "image/jpeg"


class PhoneSurface:
    """The owner's paired phone (ADB through drone.py): taps, typing and keys on whatever app is open."""
    kind = "phone"

    def __init__(self, phone):
        self.p = phone
        self.els: list[dict] = []

    async def open(self, package: str) -> None:
        await self.p.call("open_app", {"package": package}, timeout=20)
        await asyncio.sleep(2.5)

    async def observe(self) -> dict:
        nodes = await self.p.call("ui_tree", {}, timeout=60)
        self.els = [n for n in nodes if n.get("clickable") or n.get("editable") or n.get("text") or n.get("desc")][:150]
        text = " · ".join(n.get("text") or n.get("desc") for n in self.els if (n.get("text") or n.get("desc")))[:2500]
        return {"place": "phone", "title": "", "text": text,
                "elements": [{"label": (n.get("text") or n.get("desc") or n.get("id") or "")[:80],
                              "kind": ("box" if n.get("editable") else "button" if n.get("clickable") else "text")} for n in self.els]}

    def _el(self, i) -> dict:
        try:
            return self.els[int(i)]
        except (ValueError, TypeError, IndexError) as exc:
            raise ValueError(f"There's no element {i}") from exc

    async def click(self, i) -> None:
        e = self._el(i)
        await self.p.call("tap", {"x": e["x"], "y": e["y"]})
        await asyncio.sleep(1.2)

    async def type(self, i, text: str, enter: bool = False) -> None:
        if i is not None:
            await self.click(i)
        await self.p.call("text", {"text": text})
        if enter:
            await self.p.call("key", {"key": "enter"})
        await asyncio.sleep(0.8)

    async def press(self, key: str) -> None:
        await self.p.call("key", {"key": key.lower()})
        await asyncio.sleep(0.8)

    async def scroll(self, down: bool = True) -> None:
        y1, y2 = (1600, 600) if down else (600, 1600)
        await self.p.call("swipe", {"x1": 540, "y1": y1, "x2": 540, "y2": y2, "ms": 350})
        await asyncio.sleep(1)

    async def back(self) -> None:
        await self.press("back")

    async def goto(self, url: str) -> None:
        await self.p.call("open_url", {"url": url}, timeout=20)
        await asyncio.sleep(2.5)

    async def shot(self) -> tuple[bytes, str]:
        st = await self.p.call("screenshot", {"width": 540}, timeout=30)
        data = st.get("image", "")
        mime = data[5:data.index(";")] if data.startswith("data:") else "image/png"
        return base64.b64decode(data.split(",", 1)[1]) if "," in data else b"", mime


# ---------------------------------------------------------------- the loop
SYSTEM = """You operate {where} for the user, step by step, like a careful person. Goal: {goal}

Each turn you see the current screen: the place, its visible text, and numbered elements. Reply with ONE JSON
object and nothing else:
  {{"do": "click", "id": 3, "why": "…"}}
  {{"do": "type", "id": 5, "text": "KitKat", "enter": true}}        (type into box 5; use {{placeholders}} for the user's values)
  {{"do": "press", "key": "Enter"}}  {{"do": "scroll", "down": true}}  {{"do": "back"}}  {{"do": "goto", "url": "https://…"}}
  {{"do": "wait", "seconds": 10, "why": "…"}}                       (e.g. while the user approves a UPI request)
  {{"do": "ask", "question": "…", "fields": [{{"name": "phone", "label": "Mobile number", "secret": false}}]}}
      when you need something only the user knows: login, mobile number, OTP, password, API key, address, UPI ID.
      Mark passwords, OTPs, PINs, card numbers, CVVs and keys "secret": true. You will then see their values only
      as placeholders like {{{{phone}}}}; type the placeholder and the real value is typed for you.
  {{"do": "confirm", "summary": "2 × KitKat, delivery to Home", "amount": "₹45", "methods": ["UPI", "Cash on delivery"]}}
      BEFORE anything that spends money, places an order, books, or sends something to someone. Wait for the answer.
  {{"do": "done", "summary": "…"}}   {{"do": "fail", "reason": "…"}}
Rules: never guess a password or OTP; never pay or place an order without a confirm the user approved; if a
captcha appears, ask the user to solve it in Watch live (do "ask" with no fields). Stop with done as soon as the
goal is met (adding to the cart is done once it's in the cart). Values the user gave so far: {values}."""


def _shot_id(user: str, img: bytes, mime: str) -> str | None:
    if not img:
        return None
    sid = uuid.uuid4().hex[:16]
    SHOTS[sid] = (user, img, mime)
    while len(SHOTS) > MAX_SHOTS:
        SHOTS.popitem(last=False)
    return sid


def shot_for(sid: str, user: str) -> tuple[bytes, str] | None:
    hit = SHOTS.get(sid)
    if not hit or (hit[0] != user and user != "owner"):
        return None
    return hit[1], hit[2]


def _json(raw: str) -> dict | None:
    try:
        return json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except (ValueError, TypeError):
        return None


class Vault:
    """What the user typed into the cards, for this task only. The model sees {{name}}; the page gets the value."""

    def __init__(self):
        self.values: dict[str, str] = {}
        self.secret: set[str] = set()

    def put(self, name: str, value: str, secret: bool) -> None:
        self.values[name] = value
        if secret:
            self.secret.add(name)

    def fill(self, text: str) -> str:
        return re.sub(r"\{\{\s*([\w-]+)\s*\}\}", lambda m: self.values.get(m.group(1), m.group(0)), text or "")

    def hide(self, text: str) -> str:
        """Every value the user gave, back to its placeholder: nothing they typed reaches the model or the chat."""
        for name, v in sorted(self.values.items(), key=lambda kv: -len(kv[1])):
            if len(v) >= 3:
                text = text.replace(v, "{{" + name + "}}")
        return text

    def describe(self) -> str:
        return ", ".join("{{" + n + "}}" + (" (secret)" if n in self.secret else "") for n in self.values) or "none yet"

    def clear(self) -> None:
        self.values.clear()


def _screen_text(obs: dict, vault: Vault) -> str:
    rows = [f"[{i}] {e['kind']} {json.dumps(e['label'][:80], ensure_ascii=False)}" for i, e in enumerate(obs["elements"])]
    return vault.hide(f"Place: {obs['place']}  {obs['title']}\nText: {obs['text']}\nElements:\n" + "\n".join(rows))


async def run(task: Task, surface, intent: dict, bee_name: str, live_url: str | None) -> None:
    """Work on the goal until done, failed, stopped or out of steps."""
    from .cloud_chat import ask
    vault = Vault()
    approved_amount: float | None = None  # the total you said yes to; a bigger one asks again
    history: list[str] = []
    where = APPS[intent["app"]]["name"] if intent.get("app") else (intent.get("url") or "the web")
    place = "your phone" if surface.kind == "phone" else f"{bee_name}'s browser"
    limit = STEPS.get(task.effort, 25)

    async def screen(title: str, parent: str | None = None) -> None:
        try:
            img, mime = await surface.shot()
        except Exception:
            return
        sid = _shot_id(task.user, img, mime)
        if sid:
            await task.emit("screen", bee=bee_name, title=title, shot=sid, live=live_url, parent=parent)

    try:
        for n in range(limit):
            try:
                obs = await surface.observe()
            except Exception as exc:
                await task.answer(f"I couldn't see {where}: {str(exc)[:200]}")
                return
            msgs = [{"role": "system", "content": SYSTEM.format(where=f"{where} in {place}", goal=intent["goal"], values=vault.describe())},
                    {"role": "user", "content": ("Done so far:\n" + "\n".join(history[-14:]) + "\n\n" if history else "")
                     + "Screen now:\n" + _screen_text(obs, vault)}]
            try:
                raw = await brain.brain.chat(msgs, task.effort, max_tokens=300, temperature=0.1)
            except brain.BrainNotConnected as exc:
                await task.answer(f"The model didn't answer ({exc}), so I stopped.")
                return
            act = _json(raw)
            if not act or not act.get("do"):
                history.append("(your last reply wasn't one JSON object)")
                continue
            kind = str(act["do"]).lower()
            why = vault.hide(str(act.get("why") or ""))[:120]

            if kind == "done":
                await screen(f"Done on {where}")
                await task.answer(vault.hide(str(act.get("summary") or "Done.")))
                return
            if kind == "fail":
                await screen(f"Stuck on {where}")
                await task.answer(f"I couldn't finish on {where}: {vault.hide(str(act.get('reason') or 'no reason given'))}"
                                  + (" Open Watch live to do it yourself." if live_url else ""))
                return

            if kind == "ask":
                fields = [f for f in (act.get("fields") or []) if isinstance(f, dict) and re.fullmatch(r"[\w-]{1,30}", str(f.get("name", "")))][:6]
                await screen(f"{where} needs you")
                reply = await ask(task, "clarify", title=f"{where} needs you", service=where, ask_anyway=True,
                                  text=vault.hide(str(act.get("question") or "Fill these in to continue."))
                                  + ("\n\nSecret boxes go straight into the website; they aren't saved or shown to the model."
                                     if any(f.get("secret") for f in fields) else ""),
                                  fields=[{"name": f["name"], "label": str(f.get("label") or f["name"])[:60], "required": True,
                                           "input": "password" if f.get("secret") else "text", "secret": bool(f.get("secret"))}
                                          for f in fields],
                                  actions=[{"id": "go", "label": "Continue"}, {"id": "stop", "label": "Stop", "style": "danger"}])
                if reply["action"] != "go":
                    await task.answer(f"Okay, I stopped. {where} is still open in {place}.")
                    return
                for f in fields:
                    vault.put(f["name"], str((reply.get("values") or {}).get(f["name"]) or ""), bool(f.get("secret")))
                history.append(f"{n}. asked the user; they gave: {vault.describe()}")
                continue

            if kind == "confirm":
                amount = _money(str(act.get("amount") or "")) or _money(obs["text"])
                methods = [str(m)[:40] for m in (act.get("methods") or []) if str(m).strip()][:6]
                await screen(f"Ready to pay on {where}")
                fields = [{"name": "method", "label": "Pay with", "input": "select", "value": methods[0],
                           "options": methods}] if methods else []
                reply = await ask(task, "confirm" if not fields else "clarify", title=f"Pay on {where}?", service=where, ask_anyway=True,
                                  text=vault.hide(str(act.get("summary") or "")) + (f"\nTotal: ₹{amount:,.2f}" if amount else ""),
                                  details=[["Total", f"₹{amount:,.2f}" if amount else str(act.get("amount") or "see the screen")],
                                           ["Where", where]],
                                  fields=fields,
                                  actions=[{"id": "pay", "label": f"Pay ₹{amount:,.0f}" if amount else "Pay"},
                                           {"id": "stop", "label": "Don't pay", "style": "danger"}])
                if reply["action"] != "pay":
                    await task.answer(f"Okay, nothing was paid. Everything is still in the cart on {where}.")
                    return
                approved_amount = amount if amount else float("inf")
                chosen = (reply.get("values") or {}).get("method")
                history.append(f"{n}. the user approved paying" + (f" ₹{amount:,.2f}" if amount else "") + (f" with {chosen}" if chosen else ""))
                continue

            # actions on the screen
            label = ""
            if act.get("id") is not None:
                try:
                    label = obs["elements"][int(act["id"])]["label"]
                except (ValueError, TypeError, IndexError):
                    history.append(f"{n}. there's no element {act.get('id')}")
                    continue
            if kind == "click" and RISKY.search(label):
                total = _money(obs["text"])
                if approved_amount is None or (total and total > approved_amount + 0.5):
                    reply = await ask(task, "confirm", title=f"Press “{vault.hide(label)[:40]}” on {where}?", service=where, ask_anyway=True,
                                      text="This spends money, places an order or sends something. Nothing happens without your OK.",
                                      details=[["Total on the page", f"₹{total:,.2f}" if total else "not shown"]],
                                      actions=[{"id": "pay", "label": "Yes, go ahead"}, {"id": "stop", "label": "No", "style": "danger"}])
                    if reply["action"] != "pay":
                        await task.answer(f"Okay, I didn't press it. {where} is still open in {place}.")
                        return
                    approved_amount = total or float("inf")
            s = await task.step("running", _say(kind, act, vault.hide(label)), bee=bee_name, detail=why or None)
            try:
                if kind == "click":
                    await surface.click(act["id"])
                elif kind == "type":
                    text = str(act.get("text") or "")
                    await surface.type(act.get("id"), vault.fill(text), bool(act.get("enter")))
                elif kind == "press":
                    await surface.press(str(act.get("key") or "Enter"))
                elif kind == "scroll":
                    await surface.scroll(act.get("down", True) is not False)
                elif kind == "back":
                    await surface.back()
                elif kind == "goto":
                    url = str(act.get("url") or "")
                    if "{{" in url or not url.startswith(("http://", "https://")):
                        raise ValueError("only plain http(s) addresses")  # a secret never goes into an address
                    await surface.goto(url)
                elif kind == "wait":
                    await asyncio.sleep(max(1, min(float(act.get("seconds") or 5), 30)))
                else:
                    raise ValueError(f"unknown step {kind}")
            except Exception as exc:
                await task.finish_step(s, status="error", text=vault.hide(str(exc))[:160])
                history.append(f"{n}. {kind} failed: {vault.hide(str(exc))[:120]}")
                continue
            await task.finish_step(s)
            history.append(f"{n}. {_say(kind, act, vault.hide(label))}")
        await screen(f"Where I stopped on {where}")
        await task.answer(f"I used my {limit} steps and stopped. {where} is still open in {place}; ask again to carry on"
                          + (", or open Watch live." if live_url else "."))
    finally:
        vault.clear()


def _say(kind: str, act: dict, label: str) -> str:
    lab = f"“{label[:50]}”" if label else ""
    text = str(act.get("text") or "")
    shown = text if text.startswith("{{") or len(text) < 40 else text[:37] + "…"
    return {"click": f"Tapping {lab}", "type": f"Typing {shown} {('into ' + lab) if lab else ''}".strip(),
            "press": f"Pressing {act.get('key') or 'Enter'}", "scroll": "Scrolling", "back": "Going back",
            "goto": f"Opening {str(act.get('url') or '')[:80]}", "wait": f"Waiting {act.get('seconds') or 5}s"}.get(kind, kind)


def _money(text: str) -> float | None:
    """The total on a page: the amount after "total" / "to pay" if there is one, else the largest ₹ amount."""
    m = re.search(r"(?:grand\s+total|to\s+pay|total\s+payable|order\s+total|bill\s+total|total)\D{0,20}₹\s?([\d,]+(?:\.\d{1,2})?)", text or "", re.I)
    if m:
        return float(m.group(1).replace(",", ""))
    vals = [float(v.replace(",", "")) for v in MONEY.findall(text or "") if v.replace(",", "").replace(".", "").isdigit()]
    return max(vals) if vals else None


# ---------------------------------------------------------------- in the chat
def _pick_bee(task: Task) -> dict | None:
    from .bees import registry
    if task.bee:
        return registry.mine(task.bee, task.user)
    mine = registry.of(task.user)
    words = re.compile(r"\b(?:shop\w*|order\w*|grocer\w*|buy\w*|cart|deliver\w*|food|app|web|brows\w*)\b", re.I)
    return next((b for b in mine if words.search(b["skill"]) or words.search(b["name"])), mine[0] if mine else None)


async def handle(task: Task, intent: dict) -> None:
    from .cells import cells
    from . import phone as phone_mod
    app = APPS.get(intent.get("app") or "")
    where = app["name"] if app else (intent.get("url") or "that website")
    use_phone = intent.get("phone") and task.user == "owner" and app and phone_mod.PHONES
    b = _pick_bee(task)
    if not use_phone and not b:
        await task.answer(f"A Bee does this in its own browser. Make one first (New Bee in Peace, or “make a bee called Shopper that "
                          f"orders my groceries”), then ask again: it will open {where} for you.")
        return
    name = "Phone" if use_phone else b["name"]
    s = await task.step("opening", f"Opening {where}" + (" on your phone" if use_phone else f" in {b['name']}'s browser"), bee=name)
    try:
        if use_phone:
            p = phone_mod.get()
            if "ui_tree" not in p.capabilities:
                raise RuntimeError("the phone needs ADB (Wireless debugging) for the Bee to see and tap apps")
            surface, live = PhoneSurface(p), "phone.html"
            await surface.open(app["package"])
        else:
            cell = cells.get(b["id"], b["name"], b["computer"])
            surface, live = WebSurface(cell.browser), f"browser.html?bee={b['id']}"
            await surface.open(start_url(intent))
    except Exception as exc:
        await task.finish_step(s, status="error", text=str(exc)[:200])
        await task.answer(f"I couldn't open {where}: {exc}")
        return
    await task.finish_step(s)
    if not brain.is_connected():  # no model: show where it is and let the user finish it
        img, mime = await surface.shot()
        sid = _shot_id(task.user, img, mime)
        if sid:
            await task.emit("screen", bee=name, title=f"{where}" + (f": {intent['item']}" if intent.get("item") else ""), shot=sid, live=live)
        await task.answer(f"I opened {where}" + (f" with “{intent['item']}”" if intent.get("item") else "")
                          + ". No model is running on this Hive yet, so I can't tap through it by myself: open Watch live to finish it.")
        return
    await run(task, surface, intent, name, live)
