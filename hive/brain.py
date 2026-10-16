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
        if picked:
            try:
                async with models_mod.models.using(picked) as base:
                    return await self._post(base, "", "", messages, effort, max_tokens, temperature)
            except models_mod.ModelError as exc:
                raise BrainNotConnected(str(exc)) from exc
        if not self.url:
            raise BrainNotConnected("No model is set up yet")
        return await self._post(self.url, self._get("key"), self._get("model"), messages, effort, max_tokens, temperature)

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


def system_prompt(effort: str | None = None) -> str:
    """What Mnx is told before every conversation, including the effort level's instructions."""
    return ("You are Mnx, the brain of the Mnx Hive. You plan tasks, hand steps to Bees and merge their results. "
            + effort_mod.instructions(effort))


async def ask(prompt: str, effort: str | None = None) -> str:
    """Ask the brain one thing. `effort` is one of low / med / high / ultra / maxxxx (see hive/effort.py)."""
    return await brain.chat([{"role": "system", "content": system_prompt(effort)}, {"role": "user", "content": prompt}], effort)


async def chat(messages: list[dict], effort: str | None = None) -> str:
    return await brain.chat(messages, effort)
