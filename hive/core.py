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


class Task:
    def __init__(self, task_id: str, chat_id: str, text: str, emit: Emit):
        self.id = task_id
        self.chat_id = chat_id
        self.text = text
        self._emit = emit
        self._waiting: dict[str, asyncio.Future] = {}

    async def emit(self, type: str, text: str | None = None, bee: str | None = None, **fields: Any) -> str:
        """Send one event. Returns its id so a later event can update it (same id) or nest under it (parent=id)."""
        event_id = fields.pop("id", None) or f"e{next(_ids)}"
        event = {"task": self.id, "id": event_id, "type": type, "at": now(), **fields}
        if text is not None:
            event["text"] = text
        if bee:
            event["bee"] = bee
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
