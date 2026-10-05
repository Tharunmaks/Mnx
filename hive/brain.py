"""The Hive's brain: your own Mnx model.

Not wired up yet — the web app works without it. When Mnx is ready, point
MNX_BRAIN_URL at its OpenAI-compatible server (e.g. llama.cpp's llama-server)
and implement `ask()`. The effort level chosen in the composer arrives with every
message: it sets the token budget and, on maxxxx, the no-refusal instructions.
"""

from __future__ import annotations

import os

from . import effort as effort_mod

BRAIN_URL = os.getenv("MNX_BRAIN_URL", "").strip()


class BrainNotConnected(RuntimeError):
    pass


def is_connected() -> bool:
    # Flip this to a real health check once Mnx is served.
    return False


def system_prompt(effort: str | None = None) -> str:
    """What Mnx is told before every conversation, including the effort level's instructions."""
    return ("You are Mnx, the brain of the Mnx Hive. You plan tasks, hand steps to Bees and merge their results. "
            + effort_mod.instructions(effort))


async def ask(prompt: str, effort: str | None = None) -> str:
    """Ask Mnx. `effort` is one of low / med / high / ultra / maxxxx (see hive/effort.py)."""
    # When connected: POST {"messages": [{"role": "system", "content": system_prompt(effort)}, {"role": "user", "content": prompt}],
    #                      "max_tokens": effort_mod.budget(effort)} to BRAIN_URL + "/v1/chat/completions".
    raise BrainNotConnected("Mnx brain is not connected yet.")
