"""Calls: a Bee rings you, talks with its Fish Audio voice and does what you tell it.

  "call me" · "call me in 10 minutes" · "call me at 9am and remind me to check the training"
  "have the Scout bee call me about the prices" · "call me on my phone" · "talk to me" (Talk mode)

Two ways to ring:
  in the Hive app   every open Hive page rings (and a paired phone running drone.py vibrates). You answer, the Bee
                    speaks with the Fish Audio voice, listens (Fish Audio speech to text, or the browser's when no
                    key is set) and runs what you say as a Hive request, answering out loud.
  your phone        a real phone call through your Twilio account, when your number, Twilio and MNX_PUBLIC_URL are
                    set (Settings → Voice). The Bee's lines are Fish Audio MP3s played into the call; what you say
                    is heard by Twilio's speech recognition.

Cards a request shows (confirm, choose) are read out: say "yes" or "no", or a choice's name. Cards that need typing
stay on the screen. "Bye" or "hang up" ends the call; "stop" stops what the Bee is doing.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import re
import secrets
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

import httpx

from .bees import registry
from .core import NOT_THIS, Task, now
from .voice import VoiceError, public_url, voice

DATA = Path(__file__).resolve().parent.parent / "data"
CALLS_FILE = DATA / "calls.json"
TWILIO_API = "https://api.twilio.com/2010-04-01"
RING_SECONDS = 45
KEEP_ENDED = 20
SILENT_TURNS = 3  # phone calls: hang up after this many silent turns
ENDED = ("ended", "missed", "declined", "failed", "cancelled")

CALL_ME = re.compile(r"\b(?:call|ring|phone|dial)\s+me\b|\bgive\s+me\s+a\s+(?:call|ring)\b|\b(?:talk|speak)\s+(?:to|with)\s+me\b"
                     r"|\bvoice\s+(?:mode|call|chat)\b|\blet'?s\s+talk\b", re.I)
IN_RE = re.compile(r"\bin\s+(?P<n>\d+|an?|one|two|three|five|ten|fifteen|twenty|thirty|half\s+an?)\s*"
                   r"(?P<u>sec(?:ond)?s?|min(?:ute)?s?|h(?:ou)?rs?|hours?)\b", re.I)
AT_RE = re.compile(r"\bat\s+(?P<h>\d{1,2})(?:[:.](?P<m>\d{2}))?\s*(?P<ap>a\.?m\.?|p\.?m\.?)?(?=\W|$)|\bat\s+(?P<noon>noon|midnight)\b", re.I)
REASON_RE = re.compile(r"\b(?:and|to)\s+(?P<verb>tell|remind|ask|say to|update)\s+me\s+(?:about\s+|that\s+|of\s+)?(?P<m>.+)$"
                       r"|\babout\s+(?P<m2>.+)$", re.I)
PHONE_RE = re.compile(r"\b(?:on|to)\s+my\s+(?:phone|mobile|cell|number)\b|\b(?:real|phone)\s+call\b", re.I)
APP_RE = re.compile(r"\b(?:in|on)\s+the\s+(?:app|hive|browser|page)\b", re.I)
WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "five": 5, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30}

HANGUP = re.compile(r"^\s*(?:ok(?:ay)?[,\s]+)?(?:bye|goodbye|good\s+bye|hang\s+up|end\s+(?:the\s+)?call|that'?s\s+all|nothing\s+else|"
                    r"stop\s+the\s+call)\b", re.I)
YES = re.compile(r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|go\s+ahead|do\s+it|confirm|please\s+do|haan?|correct|right)\b", re.I)
NO = re.compile(r"^\s*(?:no|nope|nah|cancel|don'?t|not\s+now|later|nahi|keep)\b", re.I)
STOP_TASK = re.compile(r"^\s*(?:stop|cancel|never\s*mind|forget\s+it)\b", re.I)
LANG = {"en": "en-IN", "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN", "bn": "bn-IN"}


# ---------------------------------------------------------------- understanding "call me …"
def parse_call(text: str, tz: str = "Asia/Kolkata", now_dt: datetime | None = None) -> dict | None:
    t = " ".join((text or "").split())
    m = CALL_ME.search(t)
    if not m:
        return None
    zone = ZoneInfo(tz or "Asia/Kolkata")
    now_dt = now_dt or datetime.now(zone)
    at = None
    im = IN_RE.search(t)
    if im:
        n_raw = im.group("n").lower()
        if n_raw.startswith("half"):
            n = 0.5
        elif n_raw.isdigit():
            n = int(n_raw)
        else:
            n = WORDS.get(n_raw, 1)
        u = im.group("u").lower()
        secs = n * (3600 if u.startswith("h") else 1 if u.startswith("s") else 60)
        at = now_dt + timedelta(seconds=secs)
    am = AT_RE.search(t)
    if am and not at:
        if am.group("noon"):
            h, mi = (12, 0) if am.group("noon").lower() == "noon" else (0, 0)
        else:
            h, mi = int(am.group("h")), int(am.group("m") or 0)
            ap = (am.group("ap") or "").lower().replace(".", "")
            if ap == "pm" and h < 12:
                h += 12
            elif ap == "am" and h == 12:
                h = 0
        if h < 24 and mi < 60:
            at = now_dt.replace(hour=h, minute=mi, second=0, microsecond=0)
            if re.search(r"\btomorrow\b", t, re.I) or at <= now_dt:
                at += timedelta(days=1)
    reason, request = "", ""
    rm = REASON_RE.search(t[m.end():]) or REASON_RE.search(t)
    if rm:
        reason = (rm.group("m") or rm.group("m2") or "").strip()
        verb = (rm.group("verb") or "").lower()
        reason = IN_RE.sub("", AT_RE.sub("", reason))
        reason = re.sub(r"\b(?:tomorrow|today|on my (?:phone|mobile)|in the app)\b", "", reason, flags=re.I)
        reason = " ".join(reason.split()).strip(" .,!?")
        if verb == "remind" and reason:
            reason = "This is your reminder: " + re.sub(r"^to\s+", "", reason, flags=re.I)
        elif reason:  # "tell me / update me about / about …": the Bee finds out when you pick up, and says it
            request = reason
    from .bee_chat import _bee_named
    caller = _bee_named(t[:m.start()]) if m.start() else None
    channel = "phone" if PHONE_RE.search(t) else "app" if APP_RE.search(t) else "auto"
    talk = bool(re.search(r"\b(?:talk|speak)\s+(?:to|with)\s+me\b|\bvoice\s+(?:mode|chat)\b|\blet'?s\s+talk\b", t, re.I))
    return {"at": at, "reason": reason, "request": request, "bee": caller["id"] if caller else None, "channel": channel, "talk": talk}


# ---------------------------------------------------------------- cards, read out and answered by voice
def _defaults(card: dict) -> tuple[dict, bool]:
    """Every field's default value, and whether the card still needs typing (an empty required field or a file)."""
    values, typing = {}, False
    for f in card.get("fields") or []:
        if not isinstance(f, dict):
            continue
        v = f.get("value")
        if v in (None, "") and f.get("input") == "select" and f.get("options"):
            o = f["options"][0]
            v = o.get("value") if isinstance(o, dict) else o
        values[f.get("name", "")] = "" if v is None else v
        if f.get("input") == "file" or (f.get("required") and v in (None, "")):
            typing = True
    return values, typing


def _go(card: dict) -> dict | None:
    acts = [a for a in card.get("actions") or [] if isinstance(a, dict)]
    return next((a for a in acts if a.get("id") not in NOT_THIS), None)


def _back(card: dict) -> dict | None:
    acts = [a for a in card.get("actions") or [] if isinstance(a, dict)]
    return next((a for a in acts if a.get("id") in NOT_THIS), None)


def card_prompt(card: dict) -> str:
    parts = [str(card.get("title") or "I need your answer").rstrip("?.! ") + "?"]
    text = str(card.get("text") or "")
    if text:
        parts.append(re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0])
    for row in (card.get("details") or [])[:3]:
        if isinstance(row, (list, tuple)) and len(row) == 2:
            parts.append(f"{row[0]}: {row[1]}.")
    _, typing = _defaults(card)
    go = _go(card)
    if typing:
        parts.append("This one needs typing, so it's on your screen.")
    elif go:
        choices = [f.get("label") for f in card.get("fields") or [] if isinstance(f, dict) and f.get("input") == "select"]
        parts.append(f"Say yes to {str(go.get('label') or 'go ahead').lower()}, or no." +
                     (f" You can also name a different {choices[0].lower()}." if choices else ""))
    return " ".join(parts)


def card_answer(card: dict, said: str) -> tuple[str, dict] | None:
    """What the spoken words choose on the card: (button id, field values), or None if they don't."""
    values, typing = _defaults(card)
    words = said.lower()
    picked = False
    for f in card.get("fields") or []:
        if not isinstance(f, dict) or f.get("input") != "select":
            continue
        for o in f.get("options") or []:
            label = str(o.get("label") if isinstance(o, dict) else o).lower()
            value = o.get("value") if isinstance(o, dict) else o
            key = re.split(r"\s[·(-]\s?", label)[0].strip()
            if key and len(key) > 2 and key in words:
                values[f["name"]] = value
                picked = True
                break
    acts = [a for a in card.get("actions") or [] if isinstance(a, dict)]
    for a in acts:  # the button's own words ("rent it", "keep it")
        label = str(a.get("label") or "").lower().strip()
        if label and len(label) > 3 and label in words:
            return str(a["id"]), values
    if NO.match(words):
        back = _back(card)
        return (str(back["id"]) if back else "cancel"), values
    if (YES.match(words) or picked) and not typing:
        go = _go(card)
        if go:
            return str(go["id"]), values
    return None


# ---------------------------------------------------------------- one call
class Call:
    def __init__(self, bee_id: str | None, reason: str = "", channel: str = "auto", at: float | None = None, talk: bool = False,
                 request: str = "", user: str | None = None):
        from .users import who
        self.user = user or who()  # whose call: it rings only their pages, and a phone only for the Hive's owner
        b = registry.mine(bee_id, self.user) if bee_id else None
        self.id = "call_" + uuid.uuid4().hex[:10]
        self.secret = secrets.token_urlsafe(18)
        self.bee_id = b["id"] if b else None
        self.bee = b["name"] if b else "Mnx"
        self.reason, self.channel, self.talk = reason, channel, talk
        self.request = request  # a request to run as soon as you answer ("call me and tell me how training is going")
        self.at = at
        self.state = "scheduled" if at else "new"
        self.via: str | None = None  # app | phone
        self.created = time.time()
        self.answered_at: float | None = None
        self.ended_at: float | None = None
        self.transcript: list[dict] = []
        self.audio: dict[int, bytes] = {}
        self.seq = 0
        self.played = 0  # phone: the last line already played into the call
        self.silent = 0
        self.task: Task | None = None
        self.runner: asyncio.Task | None = None
        self.card: dict | None = None
        self.sid: str | None = None  # Twilio's call id
        self.error: str | None = None

    @property
    def working(self) -> bool:
        return bool(self.runner and not self.runner.done())

    def view(self) -> dict:
        return {"id": self.id, "bee": self.bee, "bee_id": self.bee_id, "reason": self.reason, "request": self.request, "channel": self.channel,
                "via": self.via, "state": self.state, "at": self.at, "created": self.created, "answered_at": self.answered_at,
                "ended_at": self.ended_at, "transcript": self.transcript[-60:], "working": self.working,
                "card": self.card, "error": self.error, "voice": "fish" if voice.fish_ready else "browser", "talk": self.talk}

    def saved(self) -> dict:
        return {"id": self.id, "secret": self.secret, "bee_id": self.bee_id, "reason": self.reason, "channel": self.channel,
                "at": self.at, "created": self.created, "talk": self.talk, "request": self.request, "user": self.user}

    @classmethod
    def load(cls, d: dict) -> "Call":
        c = cls(d.get("bee_id"), d.get("reason") or "", d.get("channel") or "auto", d.get("at"), bool(d.get("talk")), d.get("request") or "",
                d.get("user") or "owner")
        c.id, c.secret, c.created = d["id"], d["secret"], d.get("created") or time.time()
        return c


class Calls:
    def __init__(self, path: Path = CALLS_FILE):
        self.path = path
        self.calls: dict[str, Call] = {}
        try:
            for d in json.loads(path.read_text()):
                c = Call.load(d)
                self.calls[c.id] = c
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            pass
        # set by the gateway
        self.broadcast: Callable[[dict], Awaitable[None]] | None = None
        self.run: Callable[[Task], Awaitable[None]] | None = None  # runs a request like a chat message
        self.notify_phones: Callable[[Call], Awaitable[None]] | None = None
        self.transport: httpx.AsyncBaseTransport | None = None  # tests swap in a fake Twilio
        self.ring_seconds = RING_SECONDS
        self._loop: asyncio.Task | None = None

    # ----- bookkeeping -----
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([c.saved() for c in self.calls.values() if c.state == "scheduled"]))
        tmp.replace(self.path)

    async def _send(self, event: dict) -> None:
        if self.broadcast:
            await self.broadcast(event)

    async def _state(self, call: Call) -> None:
        await self._send({"type": "call", "call": call.view(), "user": call.user})

    def get(self, call_id: str) -> Call:
        c = self.calls.get(call_id)
        if not c:
            raise KeyError("No such call")
        return c

    def views(self, user: str | None = None) -> list[dict]:
        return [c.view() for c in sorted(self.calls.values(), key=lambda c: c.created, reverse=True) if user is None or c.user == user]

    def _prune(self) -> None:
        ended = sorted((c for c in self.calls.values() if c.state in ENDED), key=lambda c: c.ended_at or 0)
        for c in ended[:-KEEP_ENDED]:
            self.calls.pop(c.id, None)

    # ----- starting and ringing -----
    async def start(self, bee_id: str | None = None, reason: str = "", at: datetime | float | None = None,
                    channel: str = "auto", answer: bool = False, talk: bool = False, request: str = "") -> Call:
        when = at.timestamp() if isinstance(at, datetime) else at
        if when and when <= time.time() + 5:
            when = None
        call = Call(bee_id, reason, channel if channel in ("auto", "app", "phone") else "auto", when, talk, request)
        self.calls[call.id] = call
        self._prune()
        if when:
            self._save()
            await self._state(call)
            return call
        if answer:  # Talk mode: you started it from the app, so it's already picked up
            call.via = "app"
            await self.answer(call.id)
        else:
            await self.ring(call)
        return call

    async def ring(self, call: Call) -> None:
        call.at = None
        owner = call.user == "owner"  # the phone number in Settings is the owner's
        if owner and call.channel in ("auto", "phone") and voice.twilio_ready:
            await self._twilio_place(call)
            return
        if call.channel == "phone":
            call.state, call.error = "failed", ("Phone calls need " + voice.view()["phone_calls"].removeprefix("needs ") if owner
                                                else "Calls ring in the Hive app")
            call.ended_at = time.time()
            await self._state(call)
            return
        call.state, call.via = "ringing", "app"
        await self._state(call)
        if self.notify_phones:
            try:
                await self.notify_phones(call)
            except Exception:
                pass
        asyncio.get_running_loop().create_task(self._ring_timeout(call))

    async def _ring_timeout(self, call: Call) -> None:
        await asyncio.sleep(self.ring_seconds)
        if call.state == "ringing":
            await self.end(call, "missed")

    async def answer(self, call_id: str) -> Call:
        call = self.get(call_id)
        if call.state not in ("ringing", "new"):
            raise ValueError(f"The call is {call.state}")
        call.state, call.answered_at = "live", time.time()
        await self._state(call)
        await self.say(call, self.greeting(call))
        if call.request:
            self._start_task(call, call.request)
        return call

    def greeting(self, call: Call) -> str:
        who = "your Hive" if call.bee == "Mnx" else f"the {call.bee} Bee"
        if call.talk:
            return f"Hi, it's {who}. I'm listening. What should I do?"
        if call.request:
            return f"Hi, it's {who}. You asked me about {call.request.rstrip('.?')}. One moment."
        if call.reason:
            return f"Hi, it's {who}. {call.reason.rstrip('.')}. Anything you want me to do?"
        return f"Hi, it's {who}, calling like you asked. What should I do?"

    async def decline(self, call_id: str) -> None:
        call = self.get(call_id)
        if call.state in ("ringing", "new", "scheduled"):
            await self.end(call, "declined" if call.state != "scheduled" else "cancelled")

    async def end(self, call: Call, state: str = "ended") -> None:
        if call.state in ENDED:
            return
        was_scheduled = call.state == "scheduled"
        call.state, call.ended_at = state, time.time()
        if call.runner and not call.runner.done():
            call.runner.cancel()
        if call.sid and call.via == "phone":
            asyncio.get_running_loop().create_task(self._twilio_hangup(call))
        if was_scheduled:
            self._save()
        await self._state(call)

    async def hangup(self, call_id: str) -> None:
        await self.end(self.get(call_id), "ended")

    # ----- talking -----
    async def say(self, call: Call, text: str) -> int:
        call.seq += 1
        n = call.seq
        entry = {"who": "bee", "text": text, "n": n, "at": now(), "audio": False}
        if voice.fish_ready:
            try:
                from .voice import voice_for
                call.audio[n] = await voice.tts(text, voice_for(call.bee_id))
                entry["audio"] = True
                for old in [k for k in call.audio if k < n - 30]:
                    call.audio.pop(old, None)
            except (VoiceError, httpx.HTTPError) as exc:
                call.error = f"Fish Audio: {exc}"
        call.transcript.append(entry)
        await self._send({"type": "call_said", "call": call.id, "n": n, "text": text, "audio": entry["audio"], "error": call.error})
        return n

    async def heard(self, call_id: str, text: str) -> None:
        """You said something on the call."""
        call = self.get(call_id)
        if call.state != "live":
            raise ValueError(f"The call is {call.state}")
        text = " ".join((text or "").split())[:2000]
        if not text:
            return
        call.silent = 0
        call.transcript.append({"who": "you", "text": text, "at": now()})
        await self._send({"type": "call_heard", "call": call.id, "text": text})
        if HANGUP.match(text):
            await self.say(call, "Okay, bye!")
            await self.end(call, "ended")
            return
        if call.card and call.task:
            ans = card_answer(call.card, text)
            if not ans:
                await self.say(call, "Sorry, I didn't get that. Say yes or no.")
                return
            card, call.card = call.card, None
            call.task.resolve(card["id"], ans[0], ans[1])
            await self._state(call)
            return
        if call.working:
            if STOP_TASK.match(text):
                call.runner.cancel()
                await self.say(call, "Stopped.")
                return
            await self.say(call, "I'm still on the last thing. Say stop to cancel it.")
            return
        self._start_task(call, text)

    async def tap(self, call_id: str, action: str, values: dict) -> None:
        """A card's button tapped on the call screen."""
        call = self.get(call_id)
        if not (call.card and call.task):
            raise ValueError("Nothing to answer")
        card, call.card = call.card, None
        call.task.resolve(card["id"], action, values)
        await self._state(call)

    def _start_task(self, call: Call, text: str) -> None:
        async def emit(ev: dict) -> None:
            await self._send({"type": "call_event", "call": call.id, "event": ev})
            kind = ev.get("type")
            if kind == "answer" and ev.get("text"):
                await self.say(call, ev["text"])
            elif kind in ("confirm", "clarify", "connect"):
                call.card = ev
                await self._state(call)
                await self.say(call, card_prompt(ev))
            elif kind == "resolved":
                call.card = None
            elif kind == "task_error" and ev.get("text") and "Stopped" not in ev["text"]:
                await self.say(call, ev["text"])
        # a Bee's call is that Bee's own chat (Peace); a call from the Hive has every Bee on it (Rush)
        task = Task(f"t_{uuid.uuid4().hex[:10]}", call.id, text, emit, bee=call.bee_id, mode="peace" if call.bee_id else "rush",
                    user=call.user)
        call.task = task

        async def go() -> None:
            before = call.seq
            await self._state(call)  # working: sent from here so it can't arrive after the "done" below
            try:
                if self.run:
                    await self.run(task)
                else:
                    from . import queen
                    await queen.handle(task)
            finally:
                call.task, call.card = None, None
            if call.seq == before and call.state == "live":
                await self.say(call, "Done.")
            call.runner = None  # finished: the state below must not say "working"
            await self._state(call)
        call.runner = asyncio.get_running_loop().create_task(go())

    def audio_for(self, call_id: str, n: int) -> bytes:
        data = self.get(call_id).audio.get(int(n))
        if not data:
            raise KeyError("No audio for that line")
        return data

    # ----- scheduled calls -----
    def start_loop(self) -> None:
        if not self._loop:
            self._loop = asyncio.get_running_loop().create_task(self._scheduler())

    def stop_loop(self) -> None:
        if self._loop:
            self._loop.cancel()
            self._loop = None

    async def due(self, at: float | None = None) -> list[Call]:
        at = at or time.time()
        ready = [c for c in self.calls.values() if c.state == "scheduled" and c.at and c.at <= at]
        for c in ready:
            c.state = "new"
            await self.ring(c)
        if ready:
            self._save()
        return ready

    async def _scheduler(self) -> None:
        while True:
            try:
                await self.due()
            except Exception as exc:  # keep ringing later calls
                print(f"[mnx] calls: {exc}", flush=True)
            await asyncio.sleep(10)

    # ----- real phone calls (Twilio) -----
    def base(self, call: Call) -> str:
        return f"{public_url()}/voice/twilio/{call.id}/{call.secret}"

    async def _twilio(self, method: str, path: str, data: dict | None = None) -> httpx.Response:
        sid, token = voice.s["twilio_sid"], voice.s["twilio_token"]
        async with httpx.AsyncClient(timeout=30, transport=self.transport) as c:
            return await c.request(method, f"{TWILIO_API}/Accounts/{sid}{path}", data=data, auth=(sid, token))

    async def _twilio_place(self, call: Call) -> None:
        base = self.base(call)
        call.via = "phone"
        try:
            r = await self._twilio("POST", "/Calls.json", {
                "To": voice.s["phone"], "From": voice.s["twilio_from"], "Url": base + "/start", "Method": "POST",
                "StatusCallback": base + "/status", "StatusCallbackMethod": "POST", "Timeout": str(self.ring_seconds)})
            if r.status_code >= 300:
                raise VoiceError(f"Twilio answered {r.status_code}: {(r.json() or {}).get('message') or r.text[:200]}")
            call.sid = r.json().get("sid")
            call.state = "ringing"
        except (VoiceError, httpx.HTTPError, ValueError) as exc:
            call.state, call.error, call.ended_at = "failed", str(exc)[:300], time.time()
        await self._state(call)

    async def _twilio_hangup(self, call: Call) -> None:
        try:
            await self._twilio("POST", f"/Calls/{call.sid}.json", {"Status": "completed"})
        except httpx.HTTPError:
            pass

    def check(self, call_id: str, secret: str) -> Call:
        call = self.get(call_id)
        if not secrets.compare_digest(call.secret, secret or ""):
            raise KeyError("No such call")
        return call

    async def twilio(self, call: Call, step: str, form: dict) -> str:
        """Twilio's webhook for this call: returns the TwiML that plays the Bee's next lines and listens."""
        if step == "status":
            st = form.get("CallStatus", "")
            if st in ("completed", "busy", "no-answer", "failed", "canceled"):
                await self.end(call, {"completed": "ended", "no-answer": "missed", "busy": "missed", "canceled": "cancelled"}.get(st, "failed"))
            return "<Response/>"
        if step == "start" and call.state in ("ringing", "new"):
            await self.answer(call.id)
        elif step == "heard":
            said = (form.get("SpeechResult") or "").strip()
            if said and call.state == "live":
                await self.heard(call.id, said)
            elif call.state == "live" and not call.working:
                call.silent += 1
                if call.silent >= SILENT_TURNS:
                    await self.say(call, "I'll hang up now. Call me any time from the Hive.")
                    await self.end(call, "ended")
        return self.twiml(call)

    def twiml(self, call: Call) -> str:
        base = self.base(call)
        out = []
        for e in call.transcript:
            if e.get("who") == "bee" and e.get("n", 0) > call.played:
                out.append(f"<Play>{escape(base)}/audio/{e['n']}.mp3</Play>" if e.get("audio") and e["n"] in call.audio
                           else f"<Say>{escape(e['text'])}</Say>")
                call.played = e["n"]
        if call.state in ENDED:
            out.append("<Hangup/>")
        elif call.working:
            out.append(f'<Pause length="2"/><Redirect method="POST">{escape(base)}/wait</Redirect>')
        else:
            lang = LANG.get(voice.s.get("language") or "en", "en-IN")
            out.append(f'<Gather input="speech" action="{escape(base)}/heard" method="POST" speechTimeout="auto" '
                       f'language="{lang}" actionOnEmptyResult="true"/>')
        return '<?xml version="1.0" encoding="UTF-8"?><Response>' + "".join(out) + "</Response>"


def twilio_signature_ok(url: str, params: dict, signature: str, token: str) -> bool:
    """Twilio signs each webhook: base64(HMAC-SHA1(auth token, url + sorted key+value pairs))."""
    payload = url + "".join(k + str(params[k]) for k in sorted(params))
    expected = base64.b64encode(hmac.new(token.encode(), payload.encode(), hashlib.sha1).digest()).decode()
    return bool(signature) and hmac.compare_digest(expected, signature)


calls = Calls()


# ---------------------------------------------------------------- the chat: "call me …"
def matches(text: str) -> bool:
    return bool(CALL_ME.search(text or ""))


async def handle_chat(task: Task) -> bool:
    p = parse_call(task.text, voice.s.get("timezone") or "Asia/Kolkata")
    if not p:
        return False
    bee_id = p["bee"] or (task.bee if getattr(task, "bee", None) else None)
    call = await calls.start(bee_id, p["reason"], p["at"], p["channel"], talk=p["talk"], request=p["request"])
    if call.state == "failed":
        await task.answer(f"I couldn't place the call: {call.error}. Say “call me” without “on my phone” to ring the Hive app instead.")
        return True
    how = (f"your phone {voice.s['phone']}" if call.via == "phone" or (task.user == "owner" and call.channel != "app" and voice.twilio_ready)
           else "the Hive app: keep a Hive page open (a paired phone running drone.py vibrates too)")
    tz = voice.s.get("timezone") or "Asia/Kolkata"
    when = (datetime.fromtimestamp(call.at, ZoneInfo(tz)).strftime("%a %d %b, %I:%M %p") + f" ({tz})") if call.at else "now"
    sound = (f"Fish Audio · {voice.s.get('voice_name') or voice.s.get('voice_id') or 'default voice'}" if voice.fish_ready
             else "your browser's voice (add a Fish Audio key in Settings → Voice for the Fish voice)")
    details = [["From", call.bee if call.bee != "Mnx" else "your Hive"], ["When", when], ["Rings", how], ["Voice", sound]]
    if call.reason and not call.request:
        details.append(["It says", call.reason])
    if call.request:
        details.append(["When you answer", f"it finds out and tells you: {call.request}"])
    await task.emit("result", bee="Queen", service="Voice", title="Calling you now" if not call.at else f"I'll call you at {when}",
                    badge="Call", details=details)
    await task.answer(("Calling you now. " if not call.at else f"Okay, I'll call you {when}. ")
                      + "Answer and tell me what to do; I'll speak with the Fish Audio voice and do it. Say “bye” to hang up.")
    return True
