"""Bees as your own bots: every Bee a user makes has a profile, its own chat and its own memory, like Grok's bots
and OpenAI's GPTs. There are no premade ones: you write who it is and what it's responsible for.

  profile    tagline (its responsibility), instructions, tone, greeting, up to 4 conversation starters
  knowledge  your files (text, markdown, csv, json, code); it answers from them
  memory     "remember that I'm vegetarian": it keeps facts about you and uses them
  voice      a Fish Audio voice of its own, for Talk mode and calls
  computer   "$ …" runs on its computer (switch off for a chat-only Bee)
  sharing    private, or listed for everyone on this Hive to remix; export a file and import it elsewhere

Talking freely needs a model (the owner sets them up). Without one a Bee still remembers, forgets, answers from
its knowledge and runs commands.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
BOTS_DIR = DATA / "bots"
TONES = {"friendly": "warm and friendly", "playful": "playful and witty, with light humour", "professional": "clear and professional",
         "concise": "brief: answer in as few words as you can", "caring": "gentle, patient and caring"}
KNOWLEDGE_TYPES = {".txt", ".md", ".csv", ".json", ".py", ".js", ".html", ".xml", ".yaml", ".yml", ".tsv", ".log", ".rst"}
MAX_FILE = 2 * 1024 * 1024
MAX_FILES = 20
MAX_MEMORIES = 200
CHUNK = 900
HISTORY = 12  # turns of the conversation the bot sees

class BotError(ValueError):
    pass


# ---------------------------------------------------------------- the profile
def clean_profile(p: dict, base: dict | None = None) -> dict:
    out = dict(base or {"emoji": "", "color": "", "personality": "", "tone": "friendly", "greeting": "", "starters": [],
                        "memory": True, "computer": True, "voice_id": "", "voice_name": "", "visibility": "private",
                        "template": None, "created": int(time.time())})
    p = dict(p or {})
    if "emoji" in p:
        out["emoji"] = str(p["emoji"] or "").strip()[:8]
    if "color" in p:
        c = str(p["color"] or "")
        out["color"] = c if re.fullmatch(r"#[0-9a-fA-F]{6}", c) else ""
    if "personality" in p:
        out["personality"] = str(p["personality"] or "").strip()[:4000]
    if "tone" in p:
        if p["tone"] not in TONES:
            raise BotError(f"Tone must be one of {', '.join(TONES)}")
        out["tone"] = p["tone"]
    if "greeting" in p:
        out["greeting"] = str(p["greeting"] or "").strip()[:300]
    if "starters" in p:
        out["starters"] = [str(s).strip()[:90] for s in (p["starters"] or []) if str(s).strip()][:4]
    for k in ("memory", "computer"):
        if k in p:
            out[k] = bool(p[k])
    if "voice_id" in p:
        v = str(p["voice_id"] or "")
        if v and not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", v):
            raise BotError("That doesn't look like a Fish Audio voice id")
        out["voice_id"] = v
    if "voice_name" in p:
        out["voice_name"] = str(p["voice_name"] or "")[:80]
    if "visibility" in p:
        if p["visibility"] not in ("private", "hive"):
            raise BotError("Visibility is private or hive")
        out["visibility"] = p["visibility"]
    out["template"] = None
    return out


def is_bot(b: dict | None) -> bool:
    """Every Bee has a profile now; older saved Bees get one when the Hive loads."""
    return bool(b and isinstance(b.get("bot"), dict))


def system_prompt(b: dict, knowledge: list[dict], memories: list[str], user_name: str = "") -> str:
    p = b["bot"]
    parts = [f"You are {b['name']}, a Bee the user made in their Mnx Hive. Your responsibility: {b.get('skill') or 'help the user'}.",
             p.get("personality") or "Be helpful.",
             f"Your tone: {TONES.get(p.get('tone'), TONES['friendly'])}."]
    if user_name:
        parts.append(f"The user's name is {user_name}.")
    if memories:
        parts.append("What you remember about the user:\n" + "\n".join(f"- {m}" for m in memories[-40:]))
    if knowledge:
        parts.append("Your knowledge files (use them, and name the file you used):\n" +
                     "\n\n".join(f"[{k['file']}]\n{k['text']}" for k in knowledge))
    if p.get("computer"):
        parts.append("You have your own computer; the user runs commands on it by starting a message with $.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- knowledge and memory, on disk
class Store:
    def __init__(self, root: Path = BOTS_DIR):
        self.root = root

    def _dir(self, bot_id: str) -> Path:
        if not re.fullmatch(r"[a-z0-9_]{1,40}", bot_id or ""):
            raise BotError("bad bot id")
        return self.root / bot_id

    # knowledge
    def files(self, bot_id: str) -> list[dict]:
        d = self._dir(bot_id) / "knowledge"
        if not d.is_dir():
            return []
        return [{"name": p.name, "size": p.stat().st_size} for p in sorted(d.iterdir()) if p.is_file()]

    def add_file(self, bot_id: str, name: str, data: bytes) -> dict:
        name = re.sub(r"[^\w.\- ]", "_", Path(name or "").name).strip()[:80]
        if not name or Path(name).suffix.lower() not in KNOWLEDGE_TYPES:
            raise BotError(f"Knowledge takes text files: {', '.join(sorted(KNOWLEDGE_TYPES))}")
        if len(data) > MAX_FILE:
            raise BotError("A knowledge file can be up to 2 MB")
        if len(self.files(bot_id)) >= MAX_FILES and not (self._dir(bot_id) / "knowledge" / name).exists():
            raise BotError(f"A bot keeps up to {MAX_FILES} knowledge files")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        d = self._dir(bot_id) / "knowledge"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text)
        return {"name": name, "size": len(data)}

    def remove_file(self, bot_id: str, name: str) -> None:
        p = (self._dir(bot_id) / "knowledge" / Path(name).name)
        p.unlink(missing_ok=True)

    def chunks(self, bot_id: str) -> list[dict]:
        out = []
        d = self._dir(bot_id) / "knowledge"
        for f in sorted(d.iterdir()) if d.is_dir() else []:
            text = f.read_text(errors="replace")
            paras = re.split(r"\n\s*\n", text)
            buf = ""
            for para in paras:
                if len(buf) + len(para) > CHUNK and buf:
                    out.append({"file": f.name, "text": buf.strip()})
                    buf = ""
                buf += para + "\n\n"
                while len(buf) > CHUNK * 2:  # one huge paragraph
                    out.append({"file": f.name, "text": buf[:CHUNK].strip()})
                    buf = buf[CHUNK:]
            if buf.strip():
                out.append({"file": f.name, "text": buf.strip()})
        return out

    def search(self, bot_id: str, query: str, k: int = 3) -> list[dict]:
        """The passages that best match the question (BM25 over words)."""
        chunks = self.chunks(bot_id)
        q = [w for w in _words(query) if w not in STOP]
        if not chunks or not q:
            return []
        docs = [Counter(_words(c["text"])) for c in chunks]
        avg = sum(sum(d.values()) for d in docs) / len(docs)
        scored = []
        for c, d in zip(chunks, docs):
            length = sum(d.values()) or 1
            s = 0.0
            for w in set(q):
                df = sum(1 for x in docs if w in x)
                if not df or not d[w]:
                    continue
                idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
                s += idf * d[w] * 2.2 / (d[w] + 1.2 * (0.25 + 0.75 * length / avg))
            if s > 0:
                scored.append({**c, "score": round(s, 3)})
        return sorted(scored, key=lambda x: -x["score"])[:k]

    # memory
    def memories(self, bot_id: str) -> list[dict]:
        try:
            return json.loads((self._dir(bot_id) / "memory.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_mem(self, bot_id: str, items: list[dict]) -> None:
        d = self._dir(bot_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "memory.json").write_text(json.dumps(items[-MAX_MEMORIES:], indent=1))

    def remember(self, bot_id: str, fact: str) -> dict:
        fact = " ".join(fact.split()).strip(" .")[:300]
        if not fact:
            raise BotError("Nothing to remember")
        items = [m for m in self.memories(bot_id) if m["text"].lower() != fact.lower()]
        item = {"text": fact, "at": int(time.time())}
        self._save_mem(bot_id, items + [item])
        return item

    def forget(self, bot_id: str, what: str | None = None) -> int:
        items = self.memories(bot_id)
        if not what:
            self._save_mem(bot_id, [])
            return len(items)
        words = set(_words(what)) - STOP
        keep = [m for m in items if not words or not words <= set(_words(m["text"]))]
        self._save_mem(bot_id, keep)
        return len(items) - len(keep)

    def remove(self, bot_id: str) -> None:
        shutil.rmtree(self._dir(bot_id), ignore_errors=True)

    def copy(self, src: str, dst: str) -> None:
        for f in self.files(src):
            self.add_file(dst, f["name"], (self._dir(src) / "knowledge" / f["name"]).read_bytes())


store = Store()
STOP = {"the", "a", "an", "is", "are", "was", "of", "to", "in", "on", "and", "or", "for", "what", "who", "how", "do", "does",
        "i", "me", "my", "you", "your", "it", "this", "that", "about", "with", "can", "tell", "please", "from", "at", "be"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9ऀ-෿]+", (text or "").lower())


# ---------------------------------------------------------------- export / import (share a bot as a file)
def export(b: dict) -> dict:
    files = {f["name"]: (store._dir(b["id"]) / "knowledge" / f["name"]).read_text(errors="replace") for f in store.files(b["id"])}
    prof = {k: v for k, v in b["bot"].items() if k not in ("created",)}
    return {"mnx_bot": 1, "name": b["name"], "tagline": b.get("skill", ""), "profile": prof, "knowledge": files}


# ---------------------------------------------------------------- in a bot's own chat
REMEMBER = re.compile(r"^\s*(?:please\s+)?(?:remember|note|don'?t forget)\s+(?:that\s+)?(?P<f>.+)$", re.I)
RECALL = re.compile(r"\bwhat do you (?:remember|know) about me\b|\bwhat do you remember\b|\byour memor(?:y|ies)\b", re.I)
FORGET = re.compile(r"^\s*forget\s+(?P<all>everything|all|it all)?\s*(?:about\s+|that\s+)?(?P<w>.*)$", re.I)


async def handle(task, b: dict) -> bool:
    """A message in a bot's own chat (Peace). Returns True when the bot answered."""
    from . import brain
    p = b["bot"]
    name = b["name"]
    text = task.text.strip()
    if p.get("memory", True):
        m = REMEMBER.match(text)
        if m:
            item = store.remember(b["id"], m.group("f"))
            await task.emit("memory", f"Remembered: {item['text']}", bee=name, status="done")
            await task.answer(f"Got it, I'll remember that {item['text']}.")
            return True
        if RECALL.search(text):
            mem = store.memories(b["id"])
            if not mem:
                await task.answer("I don't remember anything about you yet. Tell me “remember that …”.")
            else:
                await task.emit("result", bee=name, service="Memory", title=f"{len(mem)} things I remember", badge="Memory",
                                details=[[time.strftime("%d %b", time.localtime(x["at"])), x["text"]] for x in mem[-20:]])
                await task.answer("That's what I remember. Say “forget that …” or “forget everything” to change it.")
            return True
        m = FORGET.match(text)
        if m:
            n = store.forget(b["id"], None if m.group("all") or not m.group("w").strip() else m.group("w"))
            await task.answer(f"Done, I forgot {n} thing{'s' if n != 1 else ''}." if n else "I didn't remember anything like that.")
            return True
    hits = store.search(b["id"], text)
    if brain.is_connected():
        mem = [x["text"] for x in store.memories(b["id"])] if p.get("memory", True) else []
        msgs = [{"role": "system", "content": system_prompt(b, hits, mem, getattr(task, "user_name", "") or "")}]
        for turn in (getattr(task, "history", None) or [])[-HISTORY:]:
            msgs.append({"role": "assistant" if turn.get("role") == "bot" else "user", "content": str(turn.get("text", ""))[:4000]})
        msgs.append({"role": "user", "content": text})
        s = await task.step("thinking", "Thinking", bee=name)
        try:
            reply = await brain.chat(msgs, task.effort)
        except brain.BrainNotConnected as exc:
            await task.finish_step(s, status="error", text=str(exc)[:200])
            await task.answer(f"My brain didn't answer: {exc}")
            return True
        await task.finish_step(s, text="Answered" + (f" from {', '.join(sorted({h['file'] for h in hits}))}" if hits else ""))
        await task.answer(reply)
        return True
    if hits:  # no brain: answer straight from its knowledge, with the lines that match best
        best = hits[0]
        lines = best_lines(best["text"], text)
        await task.emit("result", bee=name, service="Knowledge", title=f"From {best['file']}", badge="Knowledge",
                        details=[[h["file"], h["text"][:300] + ("…" if len(h["text"]) > 300 else "")] for h in hits])
        await task.answer(f"From {best['file']}: {' '.join(lines)}" if lines else f"Here's what my knowledge says ({best['file']}):\n\n{best['text'][:1200]}")
        return True
    return False


def best_lines(passage: str, question: str, n: int = 2) -> list[str]:
    """The sentences of a passage that share the most words with the question."""
    q = set(_words(question)) - STOP
    parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", passage) if x.strip()]
    scored = sorted(((len(q & set(_words(x))), i, x) for i, x in enumerate(parts)), key=lambda t: (-t[0], t[1]))
    top = [x for x in scored if x[0] > 0][:n]
    return [x[2] for x in sorted(top, key=lambda t: t[1])]


def offline_reply(b: dict) -> str:
    p = b["bot"]
    can = []
    if p.get("memory", True):
        can.append("remember things (“remember that …”)")
    if store.files(b["id"]):
        can.append("answer from my knowledge files")
    if p.get("computer", True):
        can.append("run commands on my computer (“$ ls”)")
    can.append("call you (“call me”)")
    hello = p.get("greeting") or f"Hi, I'm {b['name']}."
    return (f"{hello}\n\nNo model is running on this Hive yet, so I can't chat freely. Right now I can {', '.join(can)}.")


# ---------------------------------------------------------------- "make a bee …" in the chat
MAKE = re.compile(r"\b(?:make|create|build|design|spawn|hatch)\s+(?:me\s+)?(?:a|an|another|one|my(?:\s+own)?|our|new|child)?\s*"
                  r"(?:own\s+)?(?:new\s+)?(?:custom\s+)?(?:ai\s+)?(?:bee|bot|chatbot|companion|gpt)\b", re.I)
# "build a chatbot for my website", "make a telegram bot": something to build, which the model helps with
FOR_SOMETHING = re.compile(r"\b(?:website|site|web\s*app|app|discord|telegram|whatsapp|slack|instagram|api|server|html|react|"
                           r"flask|django|node)\b", re.I)
PARENT = re.compile(r"\b(?:tell|ask|have|get|let)\s+(?:the\s+|my\s+)?(?P<p>[\w -]{1,30}?)(?:\s+bee)?\s+to\b", re.I)
NAMED = re.compile(r"\b(?:called|named)\s+[\"“']?(?P<n>[A-Za-z][\w -]{0,23}?)[\"”']?(?=\s+(?:that|which|who|to|for|with|and)\b|[,.!?]|$)", re.I)
WHAT = re.compile(r"(?:^|\s)(?:that|which|who|to|for)\s+(?:is\s+|acts as\s+|will be\s+|is responsible for\s+|handles\s+)?(?P<s>.+)$", re.I)


def parse(text: str) -> dict | None:
    """ "make a bee called Scout that watches prices", "tell Scout to make a bee called Tester that runs the tests" """
    t = " ".join((text or "").split())
    m = MAKE.search(t)
    if not m:
        return None
    from .cloud_chat import CODE_ASK
    if CODE_ASK.search(t) or FOR_SOMETHING.search(t):  # "make an AI chatbot in python" asks for code, which the model writes
        return None
    pm = PARENT.search(t[:m.start()] + " ")
    rest = t[m.end():]
    nm = NAMED.search(rest)
    wm = WHAT.search(rest[nm.end():] if nm else rest)
    desc = (wm.group("s") if wm else "").strip(" .!?")
    return {"name": nm.group("n").strip() if nm else "", "description": desc, "parent": pm.group("p").strip() if pm else None}


# ---------------------------------------------------------------- making and changing Bees
async def create_bot(name: str, tagline: str, profile: dict, created_by: str = "you", knowledge: dict | None = None,
                     owner: str | None = None) -> dict:
    from .bees import registry
    b = await registry.create(name, tagline or f"Helps with whatever {name} is asked", created_by, ["bot"], None, False,
                              profile=profile, owner=owner)
    for fname, text in (knowledge or {}).items():
        store.add_file(b["id"], fname, text.encode() if isinstance(text, str) else text)
    return b


def update_bot(b: dict, change: dict) -> dict:
    from .bees import registry
    if not is_bot(b):
        raise BotError("That isn't one of your Bees")
    b["bot"] = clean_profile(change.get("profile") or {}, b["bot"])
    if change.get("tagline"):
        b["skill"] = " ".join(str(change["tagline"]).split())[:90]
    if change.get("name"):
        name = " ".join(str(change["name"]).split())[:24]
        if any(x is not b and x["name"].lower() == name.lower() for x in registry.of(b.get("owner", "owner"))):
            raise BotError(f"You already have a Bee called {name}")
        b["name"] = name
    registry.save()
    return b


async def create_flow(task, intent: dict) -> None:
    """ "make a bee called Scout that watches prices" → a card to shape it → the Bee, with its own chat."""
    from .bees import BeeError, registry
    from .cloud_chat import ask
    parent_id = task.bee if getattr(task, "bee", None) and registry.mine(task.bee, task.user) else "you"  # Peace: a child of this Bee
    if intent.get("parent"):
        p = registry.find(intent["parent"], task.user)
        if not p:
            await task.answer(f"You don't have a Bee called {intent['parent']}. Say “list my bees” to see them.")
            return
        parent_id = p["id"]
    desc = intent.get("description") or ""
    name = intent.get("name") or ""
    maker = "" if parent_id == "you" else f" for {registry.get(parent_id)['name']}"
    reply = await ask(task, "clarify", title=f"New Bee{maker}", service="Bees", fields=[
        {"name": "name", "label": "Name", "value": name, "placeholder": "e.g. Scout", "required": True},
        {"name": "tagline", "label": "Its responsibility", "value": desc[:90], "required": True,
         "placeholder": "e.g. watches prices and tells me when they drop"},
        {"name": "personality", "label": "Instructions (optional)", "input": "textarea", "value": "",
         "placeholder": "How it works and talks, what it should and shouldn't do"},
        {"name": "tone", "label": "Tone", "input": "select", "value": "friendly",
         "options": [{"value": k, "label": k.capitalize()} for k in TONES]}],
        text="It gets its own chat, its own memory and its own computer (your plan's). Add knowledge files and a voice on the Bees page.",
        actions=[{"id": "make", "label": "Make Bee"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
    if reply["action"] != "make":
        await task.answer("Okay, no new Bee.")
        return
    v = reply.get("values") or {}
    try:
        b = await create_bot((v.get("name") or name).strip(), (v.get("tagline") or desc).strip(), {
            "personality": v.get("personality") or "", "tone": v.get("tone") or "friendly"}, created_by=parent_id, owner=task.user)
    except (BeeError, BotError) as exc:
        await task.answer(f"I couldn't make it: {exc}")
        return
    from .bees import registry as reg
    await task.emit("result", service="Bees", title=f"{b['name']} is ready", badge="New Bee",
                    details=[["Responsibility", b["skill"]], ["Computer", reg.view(b)["computer_text"]],
                             ["Talk to it", "Peace → " + b["name"]], ["Shape it", f"/bots.html#edit={b['id']}"]])
    await task.answer(f"{b['name']} is ready. Switch to Peace and pick {b['name']} to talk with it.")
