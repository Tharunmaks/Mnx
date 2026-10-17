"""Who is asking: the Hive's owner, or one of its users.

Each user is known by their access token (gateway/auth.py), never stored here: the id is a short hash of it, so
a user's Bees and plan follow their token. The owner's id is "owner".

`current` holds the id for the request or chat message being handled, so plans, Bees and the GPU can tell users
apart without passing the id through every call. Work that runs for nobody in particular counts as the owner's.
"""

from __future__ import annotations

import hashlib
from contextvars import ContextVar

OWNER = "owner"
current: ContextVar[str] = ContextVar("mnx_user", default=OWNER)


def id_for_token(token: str) -> str:
    return "u_" + hashlib.sha256(token.encode()).hexdigest()[:12]


def who() -> str:
    return current.get()


def is_owner(uid: str | None = None) -> bool:
    return (uid or current.get()) == OWNER
