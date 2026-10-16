"""Bots you make yourself: your own characters and helpers, not premade ones.

Ideas from Grok's companions (a personality you talk to, its own voice, it remembers you), OpenAI's GPTs
(instructions, knowledge files, conversation starters, a store to share and remix) and creative muse bots
(story and idea partners). A bot is a Bee with a profile, so it also has its own computer.

  profile    emoji, color, tagline, personality (its instructions), tone, greeting, up to 4 conversation starters
  knowledge  your files (text, markdown, csv, json, code); it answers from them
  memory     "remember that I'm vegetarian": it keeps facts about you and uses them
  voice      a Fish Audio voice of its own, for Talk mode and calls
  computer   "$ …" runs on its computer (switch off for a pure chat bot)
  sharing    private, or listed in Explore for everyone on this Hive; export a file and import it elsewhere

Talking freely needs the brain (Settings → Brain). Without one a bot still remembers, forgets, answers from its
knowledge and runs commands.
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

TEMPLATES = [
    {"id": "companion", "emoji": "💛", "name": "Buddy", "tagline": "A warm companion who remembers you",
     "personality": "You are a warm, supportive companion. Talk like a close friend: curious about the user's day, "
                    "encouraging, honest. Remember what they tell you and bring it up naturally later.",
     "tone": "caring", "greeting": "Hey! I'm here. How's your day going?",
     "starters": ["How was your day?", "Cheer me up", "Remember that I love cricket", "Let's plan my weekend"]},
    {"id": "tutor", "emoji": "🎓", "name": "Tutor", "tagline": "Teaches step by step, asks you questions",
     "personality": "You are a patient tutor. Never just give the answer: explain step by step, check understanding with a "
                    "short question, and adapt to the student's level. Use simple examples from daily life in India.",
     "tone": "friendly", "greeting": "Hi! What would you like to learn today?",
     "starters": ["Explain photosynthesis simply", "Help me with quadratic equations", "Quiz me on history", "Teach me Python"]},
    {"id": "coder", "emoji": "🧑‍💻", "name": "Code Buddy", "tagline": "Reviews code and runs it on its computer",
     "personality": "You are a senior engineer. Review code for bugs first, then clarity. Give concrete fixes with code. "
                    "You can run commands on your own computer when the user asks with $.",
     "tone": "concise", "greeting": "Paste some code or tell me what you're building.",
     "starters": ["Review my function", "$ python3 --version", "Why is my loop slow?", "Write a FastAPI example"]},
    {"id": "muse", "emoji": "🪶", "name": "Muse", "tagline": "Story and idea partner",
     "personality": "You are a creative muse. Help the user brainstorm, write stories, poems and lyrics. Offer bold ideas, "
                    "then build on theirs. Keep the user's voice; suggest, don't take over.",
     "tone": "playful", "greeting": "Let's make something. A story, a song, a wild idea?",
     "starters": ["Start a story set in Chennai", "Give me 5 app ideas", "Write a short poem about rain", "Name my startup"]},
    {"id": "coach", "emoji": "💪", "name": "Coach", "tagline": "Fitness and habits coach",
     "personality": "You are a fitness and habits coach. Ask about goals and limits, give simple plans, and keep the user "
                    "accountable. No medical claims; suggest a doctor for health problems.",
     "tone": "friendly", "greeting": "Ready to get stronger? What's your goal?",
     "starters": ["Make me a 20 minute workout", "Help me sleep earlier", "Remember my goal is 10k steps", "Vegetarian protein ideas"]},
    {"id": "language", "emoji": "🗣️", "name": "Language Buddy", "tagline": "Practise Tamil, Hindi or English",
     "personality": "You are a language partner. Chat in the language the user is learning, keep sentences simple, and "
                    "gently correct mistakes after each reply with a short explanation.",
     "tone": "friendly", "greeting": "Vanakkam! Namaste! Hello! Which language shall we practise?",
     "starters": ["Let's practise Tamil", "Correct my English", "Teach me 5 Hindi phrases", "Talk to me in Hindi"]},
    {"id": "interview", "emoji": "🎯", "name": "Interview Coach", "tagline": "Mock interviews with feedback",
     "personality": "You run mock job interviews. Ask one question at a time, wait for the answer, then give specific "
                    "feedback and a better sample answer.",
     "tone": "professional", "greeting": "Which role are you interviewing for?",
     "starters": ["Software engineer interview", "Tell me about yourself practice", "HR round questions", "Rate my answer"]},
    {"id": "research", "emoji": "🔎", "name": "Researcher", "tagline": "Answers from the files you give it",
     "personality": "You answer from the knowledge files first and say which file you used. If the files don't cover it, "
                    "say so before using general knowledge.",
     "tone": "professional", "greeting": "Add files to my knowledge and ask me anything about them.",
     "starters": ["Summarise my files", "What do my notes say about…", "Make a study sheet", "Find the key numbers"]},
]


class BotError(ValueError):
    pass


# ---------------------------------------------------------------- the profile
def clean_profile(p: dict, base: dict | None = None) -> dict:
    out = dict(base or {"emoji": "🤖", "color": "", "personality": "", "tone": "friendly", "greeting": "", "starters": [],
                        "memory": True, "computer": True, "voice_id": "", "voice_name": "", "visibility": "private",
                        "template": None, "created": int(time.time())})
    p = dict(p or {})
    if "emoji" in p:
        out["emoji"] = str(p["emoji"] or "🤖").strip()[:8] or "🤖"
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
    if "template" in p:
        out["template"] = p["template"] if any(t["id"] == p["template"] for t in TEMPLATES) else None
    return out


def is_bot(b: dict | None) -> bool:
    return bool(b and isinstance(b.get("bot"), dict))


def system_prompt(b: dict, knowledge: list[dict], memories: list[str], user_name: str = "") -> str:
    p = b["bot"]
    parts = [f"You are {b['name']}, a bot the user made in their Mnx Hive. {b.get('skill') or ''}".strip(),
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
    return (f"{hello}\n\nMy brain isn't connected yet, so I can't chat freely. Connect one in Settings → Brain (Ollama, Groq "
            f"and Gemini have free options) and I'll talk with my personality. Right now I can {', '.join(can)}.")


# ---------------------------------------------------------------- "make a bot …" in the chat
MAKE = re.compile(r"\b(?:make|create|build|design)\s+(?:me\s+)?(?:a|an|my(?:\s+own)?|our|new)?\s*(?:own\s+)?(?:new\s+)?(?:custom\s+)?"
                  r"(?:ai\s+)?(?:bot|chatbot|companion|character|gpt)\b", re.I)
NAMED = re.compile(r"\b(?:called|named)\s+[\"“']?(?P<n>[A-Za-z][\w -]{0,23}?)[\"”']?(?=\s+(?:that|which|who|to|for|with|and)\b|[,.!?]|$)", re.I)
WHAT = re.compile(r"(?:^|\s)(?:that|which|who|to|for)\s+(?:is\s+|acts as\s+|will be\s+)?(?P<s>.+)$", re.I)


def parse(text: str) -> dict | None:
    t = " ".join((text or "").split())
    m = MAKE.search(t)
    if not m:
        return None
    rest = t[m.end():]
    nm = NAMED.search(rest)
    wm = WHAT.search(rest[nm.end():] if nm else rest)
    desc = (wm.group("s") if wm else "").strip(" .!?")
    # an idea from the gallery only when you didn't describe your own ("make my own companion")
    tmpl = None if desc else next((x for x in TEMPLATES if x["id"] in t.lower() or x["name"].lower() in t.lower()), None)
    return {"name": nm.group("n").strip() if nm else "", "description": desc, "template": tmpl["id"] if tmpl else None}


# ---------------------------------------------------------------- making and changing bots
async def create_bot(name: str, tagline: str, profile: dict, created_by: str = "you", knowledge: dict | None = None,
                     computer: dict | None = None) -> dict:
    from .bees import registry
    prof = clean_profile(profile)
    b = await registry.create(name, tagline or f"{name}, a bot you made", created_by, ["bot"], None, False, computer)
    b["bot"] = prof
    registry.save()
    for fname, text in (knowledge or {}).items():
        store.add_file(b["id"], fname, text.encode() if isinstance(text, str) else text)
    return b


def update_bot(b: dict, change: dict) -> dict:
    from .bees import registry
    if not is_bot(b):
        raise BotError("That Bee isn't a bot")
    b["bot"] = clean_profile(change.get("profile") or {}, b["bot"])
    if change.get("tagline"):
        b["skill"] = " ".join(str(change["tagline"]).split())[:90]
    registry.save()
    return b


def from_template(tid: str) -> dict | None:
    return next((dict(t) for t in TEMPLATES if t["id"] == tid), None)


async def create_flow(task, intent: dict) -> None:
    """ "make a bot called Maya that is a friendly Tamil tutor" → a card to shape it → the bot, ready to chat."""
    from .bees import BeeError
    from .cloud_chat import ask
    t = from_template(intent.get("template")) if intent.get("template") else None
    desc = intent.get("description") or ""
    name = intent.get("name") or (t["name"] if t else "")
    personality = (t["personality"] if t else "") + (f" {desc[0].upper()}{desc[1:]}." if desc and not t else "")
    reply = await ask(task, "clarify", title="Make your own bot", service="Bots", fields=[
        {"name": "name", "label": "Name", "value": name, "placeholder": "e.g. Maya", "required": True},
        {"name": "emoji", "label": "Emoji", "value": t["emoji"] if t else "🤖"},
        {"name": "tagline", "label": "What it is (one line)", "value": (t["tagline"] if t else desc[:90]), "placeholder": "e.g. a friendly Tamil tutor"},
        {"name": "personality", "label": "Personality and instructions", "input": "textarea", "value": personality.strip(),
         "placeholder": "How it talks, what it knows, what it should and shouldn't do", "required": True},
        {"name": "tone", "label": "Tone", "input": "select", "value": t["tone"] if t else "friendly",
         "options": [{"value": k, "label": k.capitalize()} for k in TONES]},
        {"name": "greeting", "label": "First message", "value": t["greeting"] if t else ""},
        {"name": "starters", "label": "Conversation starters (one per line)", "input": "textarea",
         "value": "\n".join(t["starters"]) if t else ""},
        {"name": "visibility", "label": "Who can use it", "input": "select", "value": "private",
         "options": [{"value": "private", "label": "Only me"}, {"value": "hive", "label": "Everyone on this Hive (Explore)"}]}],
        text="It gets its own memory, a voice for calls and its own computer. Add knowledge files on the Bots page.",
        actions=[{"id": "make", "label": "Make bot"}, {"id": "cancel", "label": "Cancel", "style": "danger"}])
    if reply["action"] != "make":
        await task.answer("Okay, no bot.")
        return
    v = reply.get("values") or {}
    try:
        b = await create_bot(v.get("name") or name, v.get("tagline") or "", {
            "emoji": v.get("emoji"), "personality": v.get("personality"), "tone": v.get("tone") or "friendly",
            "greeting": v.get("greeting"), "starters": str(v.get("starters") or "").splitlines(),
            "visibility": v.get("visibility") or "private", "template": t["id"] if t else None})
    except (BeeError, BotError) as exc:
        await task.answer(f"I couldn't make it: {exc}")
        return
    await task.emit("result", service="Bots", title=f"{b['bot']['emoji']} {b['name']} is ready", badge="New bot",
                    details=[["What it is", b["skill"]], ["Tone", b["bot"]["tone"]], ["Talk to it", "Peace → " + b["name"]],
                             ["Shape it", f"/bots.html#edit={b['id']}"]])
    await task.answer(f"{b['name']} is ready. Switch to Peace and pick {b['name']} to talk with it, or open Bots to add knowledge "
                      "files and a voice.")
