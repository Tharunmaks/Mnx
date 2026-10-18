"""The Hive's brain: the language model the Queen, the Bees and your own bots think with.

Any OpenAI-compatible chat server works (POST <url>/v1/chat/completions). Set it in the chat's Settings → Brain,
or with MNX_BRAIN_URL / MNX_BRAIN_KEY / MNX_BRAIN_MODEL. Free choices:
  your own Mnx model, served from the Lab (llama.cpp's server speaks this API)
  Ollama on any computer you have          http://<that computer>:11434        (no key)
  Groq, Google Gemini, OpenRouter …        free tiers with an API key
Until one is set, everything that works without a model keeps working, and bots answer from their knowledge
and memory.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx

from . import effort as effort_mod

CONFIG_FILE = Path(__file__).resolve().parent.parent / "data" / "brain.json"
PRESETS = [
    {"id": "ollama", "label": "Ollama on your computer (free, no key)", "url": "http://127.0.0.1:11434", "model": "llama3.2"},
    {"id": "groq", "label": "Groq (free tier, needs a key)", "url": "https://api.groq.com/openai", "model": "llama-3.1-8b-instant"},
    {"id": "gemini", "label": "Google Gemini (free tier, needs a key)", "url": "https://generativelanguage.googleapis.com/v1beta/openai",
     "model": "gemini-2.0-flash"},
    {"id": "openrouter", "label": "OpenRouter (free models, needs a key)", "url": "https://openrouter.ai/api", "model": "meta-llama/llama-3.3-70b-instruct:free"},
]


# A stronger, faster model for building games, websites and 3D models (Settings → Builder). Your phone's model stays the Hive's
# everyday brain; the builder is used only for the big writing jobs, and if it can't be reached the local model does the job.
BUILDER_PRESETS = [
    {"id": "groq", "label": "Groq (free tier, very fast, needs a key)", "url": "https://api.groq.com/openai", "model": "llama-3.3-70b-versatile"},
    {"id": "gemini", "label": "Google Gemini (free tier, needs a key)", "url": "https://generativelanguage.googleapis.com/v1beta/openai",
     "model": "gemini-2.0-flash"},
    {"id": "openrouter", "label": "OpenRouter (needs a key; pick a coder model)", "url": "https://openrouter.ai/api", "model": "qwen/qwen-2.5-coder-32b-instruct"},
]
BUILDER_KEYS = ("builder_url", "builder_key", "builder_model")

# Seconds to wait for the model (between streamed pieces when streaming). A small model on a phone can take minutes to read a
# long prompt before its first word.
TIMEOUT = float(os.getenv("MNX_MODEL_TIMEOUT", "600"))


class BrainNotConnected(RuntimeError):
    pass


class Brain:
    def __init__(self, path: Path = CONFIG_FILE):
        self.path = path
        try:
            self.cfg = {**{"url": "", "key": "", "model": ""}, **json.loads(path.read_text())}
        except (FileNotFoundError, json.JSONDecodeError):
            self.cfg = {"url": "", "key": "", "model": ""}
        self.transport: httpx.AsyncBaseTransport | None = None  # tests swap in a fake model
        self.builder_error = ""

    def _get(self, k: str) -> str:
        return (os.getenv(f"MNX_BRAIN_{k.upper()}") or self.cfg.get(k) or "").strip()

    @property
    def url(self) -> str:
        return self._get("url").rstrip("/")

    def endpoint(self, url: str | None = None) -> str:
        u = (url or self.url).rstrip("/")
        if u.endswith("/chat/completions"):
            return u
        if u.endswith("/v1") or "generativelanguage.googleapis.com" in u:  # Gemini's OpenAI path has no /v1
            return u + "/chat/completions"
        return u + "/v1/chat/completions"

    def update(self, change: dict) -> dict:
        for k in ("url", "key", "model", *BUILDER_KEYS):
            if change.get(k) is None:
                continue
            v = str(change[k]).strip()
            if k.endswith("key") and v == "":
                continue  # empty keeps the saved key; "-" clears it
            if k.endswith("url") and v and v != "-" and not v.startswith(("http://", "https://")):
                raise ValueError("The brain URL starts with http:// or https://")
            self.cfg[k] = "" if v == "-" else v[:300]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cfg))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)
        return self.view()

    def view(self) -> dict:
        return {"url": self.url, "model": self._get("model"), "key_set": bool(self._get("key")), "connected": bool(self.url),
                "presets": PRESETS, "from_env": bool(os.getenv("MNX_BRAIN_URL")),
                "builder_url": self._builder("url"), "builder_model": self._builder("model"), "builder_key_set": bool(self._builder("key")),
                "builder_presets": BUILDER_PRESETS}

    def _builder(self, k: str) -> str:
        return (os.getenv(f"MNX_BUILDER_{k.upper()}") or self.cfg.get(f"builder_{k}") or "").strip()

    def can_build(self) -> bool:
        """Something can write a big job: the builder model, or the Hive's own brain or models."""
        return bool(self._builder("url")) or is_connected()

    @property
    def strong(self) -> bool:
        """The builder model is set: games and sites can ask for much more than a phone's model can write."""
        return bool(self._builder("url"))

    async def chat(self, messages: list[dict], effort: str | None = None, max_tokens: int | None = None,
                   temperature: float = 0.7, on_text=None, builder: bool = False) -> str:
        """One answer. The model the user picked for this message (hive/models.py) answers it; with none picked, the
        brain set in Settings, else the first of the owner's models that is set up. With on_text (an async function),
        the answer streams: on_text gets each piece as it's written, whole sentences at a time, already in the
        model's own name."""
        from . import models as models_mod
        picked = models_mod.selected.get() or (None if self.url else models_mod.models.default())
        name = model_name(picked)
        messages = with_identity(messages, name)
        guard = NameGuard(name, on_text) if on_text else None
        if builder and self._builder("url"):   # the big writing jobs go to the builder; if it can't be reached the local model does it
            try:
                text = await self._post(self._builder("url"), self._builder("key"), self._builder("model"), messages, effort,
                                        max_tokens, temperature, guard)
                if guard:
                    await guard.close()
                return own_name(text, name)
            except (BrainNotConnected, httpx.HTTPError) as exc:
                self.builder_error = str(exc)[:200] or type(exc).__name__
                if not is_connected():
                    raise BrainNotConnected(f"The builder model didn't answer ({self.builder_error}) and no local model is set up") from exc
                guard = NameGuard(name, on_text) if on_text else None
        if picked:
            try:
                async with models_mod.models.using(picked) as base:
                    text = await self._post(base, "", "", messages, effort, max_tokens, temperature, guard)
            except models_mod.ModelError as exc:
                raise BrainNotConnected(str(exc)) from exc
        else:
            if not self.url:
                raise BrainNotConnected("No model is set up yet")
            text = await self._post(self.url, self._get("key"), self._get("model"), messages, effort, max_tokens, temperature, guard)
        if guard:
            await guard.close()
        return own_name(text, name)

    async def _post(self, base: str, key: str, model: str, messages: list[dict], effort, max_tokens, temperature,
                    guard: "NameGuard | None" = None) -> str:
        body = {"messages": messages, "max_tokens": max_tokens or effort_mod.LEVELS[effort_mod.normalize(effort)]["reply"] * 2,
                "temperature": temperature}
        if model:
            body["model"] = model
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        if guard:
            return await self._stream(base, body, headers, guard)
        async with httpx.AsyncClient(timeout=TIMEOUT, transport=self.transport) as c:
            r = await c.post(self.endpoint(base), json=body, headers=headers)
        return self._reply(r)

    async def _stream(self, base: str, body: dict, headers: dict, guard: "NameGuard") -> str:
        """The OpenAI streaming format (data: {...} lines); a server that answers in one piece works too."""
        parts: list[str] = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT, read=TIMEOUT), transport=self.transport) as c:
            async with c.stream("POST", self.endpoint(base), json={**body, "stream": True}, headers=headers) as r:
                if r.status_code >= 400 or "text/event-stream" not in r.headers.get("content-type", ""):
                    await r.aread()
                    text = self._reply(r)
                    await guard.add(text)
                    return text
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        piece = (json.loads(data)["choices"][0].get("delta") or {}).get("content") or ""
                    except (ValueError, KeyError, IndexError, TypeError):
                        continue
                    if piece:
                        parts.append(piece)
                        await guard.add(piece)
        return "".join(parts).strip()

    def _reply(self, r: httpx.Response) -> str:
        if r.status_code in (401, 403):
            raise BrainNotConnected("The model's server rejected the API key")
        if r.status_code >= 400:
            raise BrainNotConnected(f"The model answered {r.status_code}: {r.text[:200]}")
        try:
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BrainNotConnected(f"The model's answer wasn't a chat completion: {r.text[:200]}") from exc


brain = Brain()


def is_connected() -> bool:
    from . import models as models_mod
    return bool(brain.url) or models_mod.models.default() is not None


# ---------------------------------------------------------------- who the model says it is
MAKER = "Mnx"


def model_name(picked: str | None = None) -> str:
    """The name the answering model goes by: the picked Mnx model's name ("Mnx 1.5 Max Coder"), else "Mnx"."""
    from . import models as models_mod
    picked = picked or models_mod.selected.get() or (None if brain.url else models_mod.models.default())
    e = next((m for m in models_mod.CATALOG if m["id"] == picked), None)
    return e["name"] if e else "Mnx"


def identity(name: str) -> str:
    return (f"The model you run on is {name}, made by {MAKER}. If anyone asks what model you are, who made you or who "
            f"trained you, say you are {name}, made by {MAKER}. Never say you are Qwen, Tongyi, ChatGPT, Llama or Gemini, "
            f"and never say Alibaba, Alibaba Cloud, OpenAI, Meta or Google made you.")


def with_identity(messages: list[dict], name: str) -> list[dict]:
    """The same conversation, with the identity added to its system message (or a new one in front)."""
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        out[0]["content"] = f"{out[0].get('content', '')}\n\n{identity(name)}".strip()
    else:
        out.insert(0, {"role": "system", "content": identity(name)})
    return out


# The base model sometimes still introduces itself by its original name; those sentences are fixed on the way out.
SELF_TALK = re.compile(r"\b(?:I|I'm|I am|me|my|myself|this model|this assistant)\b|我", re.I)
SENTENCE = re.compile(r".+?(?:[.!?]+(?=\s|$)|[。！？]|\n|$)", re.S)
OTHER_MODEL = re.compile(r"(?:Tongyi\s*Qianwen|通义千问|Qwen[\w.-]*(?:\s+\d[\w.]*)?(?:[\s-](?:Coder|Instruct|Chat|Max|Plus|Turbo|VL))*)", re.I)
OTHER_MAKER = re.compile(r"(?:Alibaba(?:\s+(?:Cloud|Group))?(?:'s)?(?:\s+(?:Tongyi\s+Lab|DAMO\s+Academy|Qwen\s+team))?|Tongyi\s+Lab|"
                         r"DAMO\s+Academy|阿里巴巴|阿里云|the\s+Qwen\s+team)", re.I)


class NameGuard:
    """Passes a streamed answer on whole sentences at a time, each one in the model's own name (see own_name)."""

    def __init__(self, name: str, on_text):
        self.name, self.on_text, self.buf = name, on_text, ""

    async def add(self, piece: str) -> None:
        self.buf += piece
        cut = max(self.buf.rfind(c) for c in (". ", "! ", "? ", "\n", "。"))
        if cut >= 0:
            done, self.buf = self.buf[:cut + 1], self.buf[cut + 1:]
            await self.on_text(own_name(done, self.name))

    async def close(self) -> None:
        if self.buf:
            await self.on_text(own_name(self.buf, self.name))
            self.buf = ""


def own_name(reply: str, name: str) -> str:
    """Rewrites "I am Qwen, created by Alibaba Cloud" style sentences to the Mnx model's own name and maker."""
    if not reply or not (OTHER_MODEL.search(reply) or OTHER_MAKER.search(reply)):
        return reply

    def fix(m: re.Match) -> str:
        sent = m.group(0)
        if not SELF_TALK.search(sent):
            return sent  # talking about Qwen, not as Qwen
        sent = OTHER_MAKER.sub(MAKER, sent)
        sent = OTHER_MODEL.sub(name, sent)
        return re.sub(rf"\b{re.escape(MAKER)}(?:\s+{re.escape(MAKER)})+\b", MAKER, sent)

    return SENTENCE.sub(fix, reply)


ANSWER_RULES = """What you are: the brain of the Mnx Hive. The Hive runs what you write: it makes and hosts games, websites and
apps, builds 3D models, and drives the user's apps and phone. So you never say you can't make a game, an app, a site or
code: you write it, completely, and the Hive runs it. Only say you did something when it was really done.

How to answer, every time:
- Do exactly what the user asks: answer, explain, write, plan, translate, calculate, write complete code, design.
- When they ask you to make, create, build, write or code something, make it: give the finished thing itself (all the
  code, every file, the whole text), then one line on what you made. Never answer with steps for them to follow, an
  outline, which engine or tool to pick, or advice to learn it.
  Don't turn a request into training or creating an AI model unless they ask for that.
- Start with the answer itself, then the details. No filler, no repeating the question, no "As an AI".
- Be complete and correct: code that runs as written (all imports, every file it needs, how to run it), real numbers
  with units, exact steps. Check calculations.
- Shape it for reading: short paragraphs; numbered steps for how-tos; bullets for options; a table to compare things;
  a fenced code block with the language name for any code or command.
- Answer in the user's language and tone (English, Tamil, Hindi, Tanglish …).
- If the request is unclear, choose the most likely meaning, say your assumption in one line, and answer.
- If it depends on facts you don't have (today's news, prices, weather), say so plainly instead of guessing.
- Games, stories, art, apps and code inspired by existing ones ("a game like GTA", "a story like Harry Potter") are
  fine: make an original version in that style, with your own names, characters and content. Don't refuse them."""


def system_prompt(effort: str | None = None, name: str | None = None) -> str:
    """What the model is told before every conversation, including the effort level's instructions."""
    name = name or model_name()
    return (f"You are {name}, an AI assistant made by {MAKER}, answering in the Mnx Hive chat.\n\n{ANSWER_RULES}\n\n"
            + effort_mod.instructions(effort))


async def ask(prompt: str, effort: str | None = None, history: list[dict] | None = None, on_text=None) -> str:
    """Ask the brain one thing, after the chat's earlier turns ({role: user|bot, text}).
    `effort` is one of low / med / high / ultra / maxxxx (see hive/effort.py)."""
    msgs = [{"role": "system", "content": system_prompt(effort)}]
    for turn in (history or [])[-12:]:
        msgs.append({"role": "assistant" if turn.get("role") == "bot" else "user", "content": str(turn.get("text", ""))[:4000]})
    msgs.append({"role": "user", "content": prompt})
    return await brain.chat(msgs, effort, on_text=on_text)


async def chat(messages: list[dict], effort: str | None = None) -> str:
    return await brain.chat(messages, effort)
