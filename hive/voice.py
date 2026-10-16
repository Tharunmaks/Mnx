"""Fish Audio (fish.audio): the Hive's voice.

The Bees speak with a Fish Audio voice (text to speech, POST /v1/tts) and hear you with Fish Audio speech to text
(POST /v1/asr), in the chat's Talk mode and on calls (hive/calls.py). The API key and the voice stay on the server;
the browser only gets the audio.

Settings (data/voice.json, owner-only), set on the chat's Settings → Voice:
  fish_key   Fish Audio API key (fish.audio → API keys)
  voice_id   a voice model id from fish.audio (the "reference_id"); empty uses Fish Audio's default voice
  model      Fish Audio's TTS model header: s1 (default), s2-pro, s2.1-pro
  language   for speech to text, e.g. en, hi, ta
  phone      your own phone number, for real phone calls (+91…)
  twilio_*   a Twilio account that places those calls (calls.py); without it, calls ring in the Hive app
"""

from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path

import httpx

DATA = Path(__file__).resolve().parent.parent / "data"
SETTINGS_FILE = DATA / "voice.json"
FISH_URL = (os.getenv("MNX_FISH_URL") or "https://api.fish.audio").rstrip("/")
MODELS = ("s1", "s2-pro", "s2.1-pro")
SECRET = ("fish_key", "twilio_token")
DEFAULTS = {"fish_key": "", "voice_id": "", "voice_name": "", "model": "s1", "language": "en", "speed": 1.0,
            "phone": "", "twilio_sid": "", "twilio_token": "", "twilio_from": "", "timezone": "Asia/Kolkata"}
MAX_TTS_CHARS = 1500


class VoiceError(RuntimeError):
    pass


# ---------------------------------------------------------------- MessagePack (what Fish Audio's speech to text takes)
def packb(obj) -> bytes:
    """A small MessagePack encoder: dict, list, str, bytes, int, float, bool, None."""
    if obj is None:
        return b"\xc0"
    if obj is True:
        return b"\xc3"
    if obj is False:
        return b"\xc2"
    if isinstance(obj, int):
        if 0 <= obj < 128:
            return bytes([obj])
        if -32 <= obj < 0:
            return struct.pack("b", obj)
        return b"\xd3" + struct.pack(">q", obj)
    if isinstance(obj, float):
        return b"\xcb" + struct.pack(">d", obj)
    if isinstance(obj, str):
        raw = obj.encode()
        n = len(raw)
        head = bytes([0xA0 | n]) if n < 32 else b"\xd9" + bytes([n]) if n < 256 else b"\xda" + struct.pack(">H", n) if n < 65536 \
            else b"\xdb" + struct.pack(">I", n)
        return head + raw
    if isinstance(obj, (bytes, bytearray)):
        n = len(obj)
        head = b"\xc4" + bytes([n]) if n < 256 else b"\xc5" + struct.pack(">H", n) if n < 65536 else b"\xc6" + struct.pack(">I", n)
        return head + bytes(obj)
    if isinstance(obj, (list, tuple)):
        n = len(obj)
        head = bytes([0x90 | n]) if n < 16 else b"\xdc" + struct.pack(">H", n)
        return head + b"".join(packb(x) for x in obj)
    if isinstance(obj, dict):
        n = len(obj)
        head = bytes([0x80 | n]) if n < 16 else b"\xde" + struct.pack(">H", n)
        return head + b"".join(packb(str(k)) + packb(v) for k, v in obj.items())
    raise TypeError(f"can't pack {type(obj).__name__}")


# ---------------------------------------------------------------- settings
class Voice:
    def __init__(self, path: Path = SETTINGS_FILE):
        self.path = path
        try:
            saved = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            saved = {}
        self.s = {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}
        if not self.s["fish_key"] and os.getenv("FISH_API_KEY"):
            self.s["fish_key"] = os.environ["FISH_API_KEY"]
        self.transport: httpx.AsyncBaseTransport | None = None  # tests swap in a fake Fish Audio

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.s, indent=2))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    def update(self, change: dict) -> dict:
        for k, v in change.items():
            if k not in DEFAULTS or v is None:
                continue
            if k in SECRET and v == "":  # an empty secret field means "keep it"; clear with "-"
                continue
            v = "" if (k in SECRET and v == "-") else v
            if k == "model" and v not in MODELS:
                raise VoiceError(f"Model must be one of {', '.join(MODELS)}")
            if k == "speed":
                v = min(2.0, max(0.5, float(v)))
            elif k == "phone" and v:
                v = re.sub(r"[\s()-]", "", str(v))
                if not re.fullmatch(r"\+\d{8,15}", v):
                    raise VoiceError("Write the phone number with its country code, like +919876543210")
            elif k == "twilio_from" and v:
                v = re.sub(r"[\s()-]", "", str(v))
            elif k == "voice_id" and v and not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", str(v)):
                raise VoiceError("That doesn't look like a Fish Audio voice id")
            elif k == "timezone" and v:
                from zoneinfo import ZoneInfo
                try:
                    ZoneInfo(str(v))
                except Exception as exc:
                    raise VoiceError("Unknown time zone") from exc
            self.s[k] = str(v)[:200] if isinstance(v, str) else v
        self.save()
        return self.view()

    @property
    def fish_ready(self) -> bool:
        return bool(self.s["fish_key"])

    @property
    def twilio_ready(self) -> bool:
        return all(self.s[k] for k in ("twilio_sid", "twilio_token", "twilio_from", "phone")) and bool(public_url())

    def view(self) -> dict:
        out = {k: v for k, v in self.s.items() if k not in SECRET}
        out.update(fish_key_set=bool(self.s["fish_key"]), twilio_token_set=bool(self.s["twilio_token"]),
                   fish_ready=self.fish_ready, twilio_ready=self.twilio_ready, public_url=public_url(), models=list(MODELS),
                   phone_calls=("ready" if self.twilio_ready else "needs " + ", ".join(
                       n for n, ok in (("your phone number", self.s["phone"]), ("a Twilio account", self.s["twilio_sid"] and self.s["twilio_token"]),
                                       ("a Twilio number", self.s["twilio_from"]), ("MNX_PUBLIC_URL", public_url())) if not ok)))
        return out

    # ---------------------------------------------------------------- Fish Audio
    def _headers(self, **extra) -> dict:
        if not self.s["fish_key"]:
            raise VoiceError("Add your Fish Audio API key in Settings → Voice")
        return {"Authorization": f"Bearer {self.s['fish_key']}", **extra}

    @staticmethod
    def _error(r: httpx.Response) -> VoiceError:
        if r.status_code in (401, 403):
            return VoiceError("Fish Audio rejected the API key")
        if r.status_code == 402:
            return VoiceError("Your Fish Audio account is out of credit")
        if r.status_code == 404:
            return VoiceError("Fish Audio doesn't know that voice id")
        return VoiceError(f"Fish Audio answered {r.status_code}: {r.text[:200]}")

    async def tts(self, text: str) -> bytes:
        """Speak `text` with the chosen Fish Audio voice. Returns MP3 bytes."""
        text = speakable(text)
        if not text:
            raise VoiceError("Nothing to say")
        body = {"text": text, "format": "mp3", "mp3_bitrate": 128, "normalize": True, "latency": "balanced",
                "prosody": {"speed": float(self.s["speed"] or 1.0), "volume": 0}}
        if self.s["voice_id"]:
            body["reference_id"] = self.s["voice_id"]
        async with httpx.AsyncClient(timeout=90, transport=self.transport) as c:
            r = await c.post(FISH_URL + "/v1/tts", json=body, headers=self._headers(model=self.s["model"] or "s1"))
        if r.status_code != 200:
            raise self._error(r)
        if not r.content:
            raise VoiceError("Fish Audio sent no audio")
        return r.content

    async def asr(self, audio: bytes, language: str | None = None) -> str:
        """What was said in `audio` (webm, ogg, wav, mp3 …), as text."""
        if not audio:
            return ""
        if len(audio) > 20 * 1024 * 1024:
            raise VoiceError("That recording is too long")
        body = packb({"audio": audio, "language": language or self.s["language"] or None, "ignore_timestamps": True})
        async with httpx.AsyncClient(timeout=90, transport=self.transport) as c:
            r = await c.post(FISH_URL + "/v1/asr", content=body, headers=self._headers(**{"Content-Type": "application/msgpack"}))
        if r.status_code != 200:
            raise self._error(r)
        return str((r.json() or {}).get("text") or "").strip()

    async def voices(self, query: str = "") -> list[dict]:
        """Voices on fish.audio matching `query` (to pick a voice id)."""
        params = {"page_size": 12, "page_number": 1, "sort_by": "score"}
        if query:
            params["title"] = query[:60]
        async with httpx.AsyncClient(timeout=30, transport=self.transport) as c:
            r = await c.get(FISH_URL + "/model", params=params, headers=self._headers())
        if r.status_code != 200:
            raise self._error(r)
        out = []
        for m in (r.json() or {}).get("items") or []:
            out.append({"id": m.get("_id"), "title": m.get("title"), "languages": m.get("languages") or [],
                        "description": (m.get("description") or "")[:140], "likes": m.get("like_count")})
        return [v for v in out if v["id"]]


def speakable(text: str) -> str:
    """Chat text made fit to read aloud: no markdown, links or code; at most MAX_TTS_CHARS."""
    t = re.sub(r"```.*?```", " (code on screen) ", text or "", flags=re.S)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"https?://\S+", "the link on screen", t)
    t = re.sub(r"[*_#>|]+", " ", t)
    t = " ".join(t.split())
    if len(t) > MAX_TTS_CHARS:
        cut = t[:MAX_TTS_CHARS]
        t = cut[:cut.rfind(". ") + 1] if ". " in cut else cut
    return t


def public_url() -> str:
    return os.getenv("MNX_PUBLIC_URL", "").rstrip("/")


voice = Voice()
