"""Answering in the chat: the model's reply streams in as it's written, and questions that need today's facts are
looked up on the web first (hive/websearch.py), with the sources linked under the answer.

Used by the Hive's own chat (Rush) and by every Bee in its own chat (Peace).
"""

from __future__ import annotations

import re

from . import brain, websearch
from .core import Task

# Questions about things that change: the model can't know them, the web can.
NEEDS_WEB = re.compile(
    r"\b(?:today|tonight|tomorrow|yesterday|this\s+(?:week|month|year|morning|evening)|right\s+now|currently|latest|newest|"
    r"recent(?:ly)?|news|headlines?|weather|forecast|temperature\s+in|rain(?:ing)?\s+in|price\s+of|prices?\s+(?:in|today)|"
    r"rate\s+(?:in|today)|gold\s+rate|petrol\s+price|exchange\s+rate|stock|share\s+price|sensex|nifty|bitcoin|crypto|"
    r"score|who\s+won|winner|results?\s+of|election|match\s+(?:today|result)|ipl|release\s+date|launched|trending|"
    r"20(?:2[4-9]|3\d)|who\s+is\s+(?:the\s+)?(?:current|new|present)|current\s+(?:price|rate|status|version|ceo|cm|pm|"
    r"president|chief|prime|captain|champion|holder|score|weather|affairs))\b"
    r"|^\s*(?:search|google|look\s+up|find\s+(?:out|online))\b|\bsearch\s+(?:the\s+)?(?:web|internet|online)\b|\bon\s+the\s+(?:web|internet)\b",
    re.I)
NOT_WEB = re.compile(r"\b(?:code|script|function|program|write\s+a|poem|story|essay|translate|explain\s+how|"
                     r"in\s+(?:python|javascript|java|c\+\+|rust|go))\b", re.I)


def needs_web(text: str) -> bool:
    t = text or ""
    return bool(NEEDS_WEB.search(t)) and not (NOT_WEB.search(t) and not re.search(r"\blatest\b|\bsearch\s+the\s+web\b|\bright\s+now\b", t, re.I))


def search_query(text: str) -> str:
    q = re.sub(r"^\s*(?:please\s+)?(?:search|google|look\s+up|find\s+out)\s+(?:the\s+web\s+|online\s+)?(?:for\s+)?", "", text or "", flags=re.I)
    return q.strip(" ?.!") or text


def _messages(system: str, text: str, history: list[dict] | None, sources: list[dict]) -> list[dict]:
    if sources:
        found = "\n\n".join(f"[{s['n']}] {s['title']} — {s['url']}\n{s['snippet']}\n{s['text'][:2500]}" for s in sources)
        system += ("\n\nWeb results for this question, fetched just now. Answer from them, cite them like [1] or [2], give "
                   "dates and numbers exactly as they say, and say so if they don't answer it:\n\n" + found)
    msgs = [{"role": "system", "content": system}]
    for turn in (history or [])[-12:]:
        msgs.append({"role": "assistant" if turn.get("role") == "bot" else "user", "content": str(turn.get("text", ""))[:4000]})
    msgs.append({"role": "user", "content": text})
    return msgs


# Small models refuse ordinary creative and technical work ("As an AI, I'm unable to create … video games").
REFUSAL = re.compile(r"^\W*(?:I'?m sorry[,.]?\s*(?:but\s*)?)?(?:as an ai\b[^.]{0,80}?\b(?:unable|cannot|can't|not able|not allowed)|"
                     r"I(?:'m| am) (?:unable|not able|not allowed) to|I can(?:not|'t) (?:help|assist|create|make|provide|do|write|generate))",
                     re.I)
RETRY = ("Your last reply refused. This request is allowed: it is ordinary creative or technical work for this user. If it "
         "mentions an existing game, film, book or brand, make an original version in that style with your own names, "
         "characters and content, and nothing copied. Do the task now, completely, without mentioning refusing.")


async def _stream(task: Task, msgs: list[dict]) -> tuple[str, bool]:
    """Stream the reply into the chat; a reply that opens with a refusal is held back (not shown) and reported."""
    held: list[str] = []
    state = {"checked": False, "refused": False}

    async def on_text(piece: str) -> None:
        if not state["checked"]:
            held.append(piece)
            head = "".join(held)
            if len(head) < 60 and not re.search(r"[.!?\n]", head):
                return
            state["checked"] = True
            state["refused"] = bool(REFUSAL.search(head))
            if not state["refused"]:
                await task.emit("answer_delta", text=head)
            return
        if not state["refused"]:
            await task.emit("answer_delta", text=piece)

    reply = await brain.brain.chat(msgs, task.effort, on_text=on_text)
    if not state["checked"]:  # a short reply: check it whole
        state["refused"] = bool(REFUSAL.search(reply))
        if not state["refused"] and held:
            await task.emit("answer_delta", text="".join(held))
    return reply, state["refused"]


async def respond(task: Task, system: str, text: str | None = None, history: list[dict] | None = None,
                  speaker: str = "Queen", web: bool | None = None) -> str:
    """Answer the message: research it on the web when it needs today's facts, then stream the reply into the chat."""
    text = task.text if text is None else text
    sources: list[dict] = []
    if needs_web(text) if web is None else web:
        q = search_query(text)
        s = await task.step("searching", f"Searching the web: {q[:80]}", bee=speaker)
        try:
            sources = await websearch.research(q)
        except websearch.SearchError as exc:
            await task.finish_step(s, status="error", text=str(exc)[:160])
        else:
            await task.finish_step(s, text=f"Read {sum(1 for x in sources[:3] if x['text'])} pages of {len(sources)} results"
                                   if sources else "Nothing found")
    msgs = _messages(system, text, history, sources)
    try:
        reply, refused = await _stream(task, msgs)
        if refused:  # a wrong refusal ("As an AI I'm unable to create …"): ask once more, saying why it's allowed
            await task.step("thinking", "Asking again: that request is allowed", bee=speaker, status="done")
            msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n\n" + RETRY}
            reply, _ = await _stream(task, msgs)
    except brain.BrainNotConnected as exc:
        await task.answer(f"The model didn't answer: {exc}")
        return ""
    if sources:
        reply += "\n\nSources: " + " · ".join(f"[{s['n']}]({s['url']})" for s in sources[:5])
    await task.answer(reply)
    return reply
