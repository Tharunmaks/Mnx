"""The Hive's brain: your own Mnx model.

Not wired up yet — the web app works without it. When Mnx is ready, point
MNX_BRAIN_URL at its OpenAI-compatible server (e.g. llama.cpp's llama-server)
and implement `ask()`.
"""

from __future__ import annotations

import os

BRAIN_URL = os.getenv("MNX_BRAIN_URL", "").strip()


class BrainNotConnected(RuntimeError):
    pass


def is_connected() -> bool:
    # Flip this to a real health check once Mnx is served.
    return False


async def ask(prompt: str) -> str:
    raise BrainNotConnected("Mnx brain is not connected yet.")
