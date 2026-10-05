"""Phone Drone link: phones running drone/drone.py connect here and take commands.

The gateway never reaches into the phone; the phone dials out over WebSocket,
so it works behind mobile data and NAT.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from typing import Any

from fastapi import WebSocket

# Commands that send something to other people or cost money: the UI asks first,
# and Bees must use the Ask lane for them.
RISKY = {"sms_send", "call", "open_url"}

COMMANDS = {
    # screen (needs ADB / Shizuku)
    "screenshot", "tap", "long_press", "swipe", "text", "key", "open_app", "apps", "ui_tree", "tap_text", "screen_size",
    # Termux:API
    "battery", "notifications", "sms_list", "sms_send", "call", "torch", "volume", "clipboard_get", "clipboard_set",
    "tts", "vibrate", "location", "toast", "open_url", "wifi",
    # layer-by-layer streaming: run / train a model on the phone one layer at a time
    "stream_start", "stream_status", "stream_stop", "stream_chat",
}
SLOW = {"stream_chat": 3600, "stream_start": 120, "location": 60, "ui_tree": 60, "tap_text": 60}  # seconds the gateway waits


class Phone:
    def __init__(self, ws: WebSocket, hello: dict):
        self.ws = ws
        self.id = str(hello.get("device_id") or "phone")[:64]
        self.info = {k: hello.get(k) for k in ("name", "model", "android", "backend", "ram_gb", "storage_free_gb", "cores")}
        self.capabilities: list[str] = [c for c in hello.get("capabilities", []) if c in COMMANDS]
        self.connected_at = int(time.time())
        self.apps: list[dict] = []
        self._pending: dict[int, asyncio.Future] = {}
        self._ids = itertools.count(1)

    def view(self) -> dict:
        return {"id": self.id, **self.info, "capabilities": self.capabilities,
                "connected_at": self.connected_at, "risky": sorted(RISKY)}

    async def call(self, cmd: str, args: dict[str, Any] | None = None, timeout: float = 25) -> Any:
        if cmd not in COMMANDS:
            raise ValueError(f"Unknown command: {cmd}")
        if cmd not in self.capabilities:
            raise ValueError(f"This phone can't do '{cmd}' (missing ADB/Shizuku or Termux:API)")
        rid = next(self._ids)
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        try:
            await self.ws.send_json({"id": rid, "cmd": cmd, "args": args or {}})
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(rid, None)

    def on_reply(self, msg: dict) -> None:
        fut = self._pending.get(msg.get("id"))
        if not fut or fut.done():
            return
        if msg.get("ok"):
            fut.set_result(msg.get("result"))
        else:
            fut.set_exception(RuntimeError(str(msg.get("error") or "phone error")[:500]))

    def fail_all(self) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("Phone disconnected"))


PHONES: dict[str, Phone] = {}


def get(phone_id: str | None = None) -> Phone:
    if phone_id and phone_id in PHONES:
        return PHONES[phone_id]
    if not phone_id and PHONES:
        return next(iter(PHONES.values()))
    raise LookupError("No phone connected. Start drone.py in Termux on your phone.")


def all_apps() -> list[dict]:
    return [a for p in PHONES.values() for a in p.apps]
