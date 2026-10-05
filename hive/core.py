"""Task context shared by the Queen and every Bee.

A Task streams events to the web app (steps, cards, the answer) and can pause
to ask the user something (connect an app, fill in details, confirm) and wait
for the button they tap.
"""

from __future__ import annotations

import asyncio
import itertools
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

Emit = Callable[[dict], Awaitable[None]]

_ids = itertools.count(1)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


from . import effort as effort_mod, tokens

ASK_ANYWAY = "ask_anyway"  # a card with ask_anyway=True stops for the user even on maxxxx (money, hardware)
NOT_THIS = ("cancel", "no", "later", "keep", "skip", "back")


def _auto_choice(card_type: str, card: dict) -> tuple[str, dict, str] | None:
    """What maxxxx picks on a card: the first safe button, with every field's default. None = must ask."""
    if card_type not in ("confirm", "clarify") or card.get(ASK_ANYWAY):
        return None
    values: dict = {}
    for f in card.get("fields") or []:
        if not isinstance(f, dict):
            continue
        value = f.get("value")
        if value in (None, "") and f.get("input") == "select" and f.get("options"):
            o = f["options"][0]
            value = o.get("value") if isinstance(o, dict) else o
        values[f.get("name", "")] = "" if value is None else value
        hidden = any(values.get(k) != v for k, v in (f.get("show_if") or {}).items())  # only shown for another choice
        if hidden:
            continue
        if f.get("required") and value in (None, ""):
            return None  # it needs something only the user knows (a host, a purpose…)
        if f.get("input") == "file":
            return None  # a file must be picked by hand
    actions = [a for a in card.get("actions") or [] if isinstance(a, dict)]
    wanted = card.get("default")  # a card may name the button maxxxx should press
    for a in actions:
        if wanted and a.get("id") != wanted:
            continue
        if not wanted and a.get("id") in NOT_THIS:  # the "don't do it" buttons, never picked for you
            continue
        return str(a.get("id")), values, str(a.get("label") or a.get("id"))
    return None


class Task:
    def __init__(self, task_id: str, chat_id: str, text: str, emit: Emit, effort: str | None = None):
        self.id = task_id
        self.chat_id = chat_id
        self.text = text
        self._emit = emit
        self._waiting: dict[str, asyncio.Future] = {}
        self.effort = effort_mod.normalize(effort)
        self.budget = effort_mod.budget(self.effort)
        self.tokens_in = tokens.count(text)  # token reading: what you wrote …
        self.tokens_out = 0  # … and what the Hive wrote back (steps, cards, answer)

    def usage(self) -> dict:
        return {"in": self.tokens_in, "out": self.tokens_out, "total": self.tokens_in + self.tokens_out,
                "budget": self.budget, "effort": self.effort}

    async def emit(self, type: str, text: str | None = None, bee: str | None = None, **fields: Any) -> str:
        """Send one event. Returns its id so a later event can update it (same id) or nest under it (parent=id)."""
        event_id = fields.pop("id", None) or f"e{next(_ids)}"
        event = {"task": self.id, "id": event_id, "type": type, "at": now(), **fields}
        if text is not None:
            event["text"] = text
        if bee:
            event["bee"] = bee
        self.tokens_out += tokens.count_event(event)
        await self._emit(event)
        return event_id

    async def step(self, type: str, text: str, bee: str | None = None, **fields: Any) -> str:
        return await self.emit(type, text, bee, **fields)

    async def finish_step(self, step_id: str, status: str = "done", **fields: Any) -> None:
        await self.emit("step", id=step_id, status=status, **fields)

    async def answer(self, text: str) -> None:
        await self.emit("answer", text)

    async def ask(self, card_type: str, **card: Any) -> dict:
        """Show a connect / clarify / confirm card and wait for the user's tap.

        Returns {"action": <button id>, "values": {...}}.
        """
        card_id = await self.emit(card_type, **card)
        if effort_mod.auto_approves(self.effort):
            choice = _auto_choice(card_type, card)
            if choice:  # maxxxx: no approval stops
                action, values, label = choice
                await self.emit("resolved", ref=card_id, action=action, text=f"Maxxxx chose “{label}” for you")
                return {"action": action, "values": values}
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiting[card_id] = fut
        try:
            return await fut
        finally:
            self._waiting.pop(card_id, None)

    def resolve(self, card_id: str, action: str, values: dict) -> bool:
        fut = self._waiting.get(card_id)
        if fut and not fut.done():
            fut.set_result({"action": action, "values": values})
            return True
        return False
