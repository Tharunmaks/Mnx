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
        for k in ("url", "key", "model"):
            if change.get(k) is None:
                continue
            v = str(change[k]).strip()
            if k == "key" and v == "":
                continue  # empty keeps the saved key; "-" clears it
            if k == "url" and v and v != "-" and not v.startswith(("http://", "https://")):
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
                "presets": PRESETS, "from_env": bool(os.getenv("MNX_BRAIN_URL"))}

    async def chat(self, messages: list[dict], effort: str | None = None, max_tokens: int | None = None,
                   temperature: float = 0.7) -> str:
        """One answer. The model the user picked for this message (hive/models.py) answers it; with none picked, the
        brain set in Settings, else the first of the owner's models that is set up."""
        from . import models as models_mod
        picked = models_mod.selected.get() or (None if self.url else models_mod.models.default())
        name = model_name(picked)
        messages = with_identity(messages, name)
        if picked:
            try:
                async with models_mod.models.using(picked) as base:
                    return own_name(await self._post(base, "", "", messages, effort, max_tokens, temperature), name)
            except models_mod.ModelError as exc:
                raise BrainNotConnected(str(exc)) from exc
        if not self.url:
            raise BrainNotConnected("No model is set up yet")
        return own_name(await self._post(self.url, self._get("key"), self._get("model"), messages, effort, max_tokens, temperature),
                        name)

    async def _post(self, base: str, key: str, model: str, messages: list[dict], effort, max_tokens, temperature) -> str:
        body = {"messages": messages, "max_tokens": max_tokens or effort_mod.LEVELS[effort_mod.normalize(effort)]["reply"] * 2,
                "temperature": temperature}
        if model:
            body["model"] = model
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        async with httpx.AsyncClient(timeout=180, transport=self.transport) as c:
            r = await c.post(self.endpoint(base), json=body, headers=headers)
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


def system_prompt(effort: str | None = None, name: str | None = None) -> str:
    """What the model is told before every conversation, including the effort level's instructions."""
    name = name or model_name()
    return (f"You are {name}, an AI assistant made by {MAKER}, answering in the Mnx Hive chat. Do what the user asks: answer "
            "questions, write complete working code, and build whatever they describe (apps, websites, scripts, games, "
            "documents, plans). Don't turn a request into training or creating an AI model unless they ask for that. "
            "Use Markdown, and put code in fenced code blocks with the language name. "
            + effort_mod.instructions(effort))


async def ask(prompt: str, effort: str | None = None, history: list[dict] | None = None) -> str:
    """Ask the brain one thing, after the chat's earlier turns ({role: user|bot, text}).
    `effort` is one of low / med / high / ultra / maxxxx (see hive/effort.py)."""
    msgs = [{"role": "system", "content": system_prompt(effort)}]
    for turn in (history or [])[-12:]:
        msgs.append({"role": "assistant" if turn.get("role") == "bot" else "user", "content": str(turn.get("text", ""))[:4000]})
    msgs.append({"role": "user", "content": prompt})
    return await brain.chat(msgs, effort)


async def chat(messages: list[dict], effort: str | None = None) -> str:
    return await brain.chat(messages, effort)
