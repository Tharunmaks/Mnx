"""Owner access token.

The gateway can drive a browser and your phone, so every API call and socket
needs this token. Set MNX_TOKEN, or one is generated on first run and saved
to data/token.txt (printed in the server log).
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import Header, HTTPException, Query

DATA = Path(__file__).resolve().parent.parent / "data"
TOKEN_FILE = DATA / "token.txt"


def _load() -> str:
    env = os.getenv("MNX_TOKEN", "").strip()
    if env:
        return env
    try:
        saved = TOKEN_FILE.read_text().strip()
        if saved:
            return saved
    except FileNotFoundError:
        pass
    DATA.mkdir(exist_ok=True)
    token = secrets.token_urlsafe(24)
    TOKEN_FILE.write_text(token + "\n")
    os.chmod(TOKEN_FILE, 0o600)
    return token


TOKEN = _load()
print(f"[mnx] Access token: {TOKEN}  (enter it in the web app's Settings)", flush=True)


def valid(token: str | None) -> bool:
    return bool(token) and secrets.compare_digest(token, TOKEN)


def require(authorization: str | None = Header(default=None), token: str | None = Query(default=None)) -> None:
    """FastAPI dependency: accept 'Authorization: Bearer <token>' or ?token=."""
    given = token
    if authorization and authorization.lower().startswith("bearer "):
        given = authorization[7:].strip()
    if not valid(given):
        raise HTTPException(401, "Missing or wrong access token")
