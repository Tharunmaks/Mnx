"""Access tokens: the owner's and the users'.

The gateway can drive a browser and your phone, so every API call and socket needs a token.
  owner token   yours: set MNX_TOKEN, or one is generated on first run and saved to data/token.txt (printed in
                the server log). Only it can enter or change credentials (API keys for the GPU, cloud, voice and
                brain), add servers, and see the Hive's setup.
  user tokens   for everyone else: MNX_USER_TOKENS (comma separated) or one per line in data/user_tokens.txt
                (make one with  python3 -m hive.admin user-token). Each one has their own Bees and plan (Free, or Pro bought with
                Razorpay), and never sees or changes a credential.
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
USER_TOKENS_FILE = DATA / "user_tokens.txt"
print(f"[mnx] Access token: {TOKEN}  (enter it in the web app's Settings)", flush=True)


def user_tokens() -> list[str]:
    out = [t.strip() for t in os.getenv("MNX_USER_TOKENS", "").split(",") if t.strip()]
    try:
        out += [t.strip() for t in USER_TOKENS_FILE.read_text().splitlines() if t.strip() and not t.startswith("#")]
    except FileNotFoundError:
        pass
    return out


def role(token: str | None) -> str | None:
    """"owner", "user", or None."""
    if not token:
        return None
    if secrets.compare_digest(token, TOKEN):
        return "owner"
    return "user" if any(secrets.compare_digest(token, t) for t in user_tokens()) else None


def user_of(token: str | None) -> str | None:
    """The caller's id (hive/users.py): "owner", "u_<hash of the token>", or None for a wrong token."""
    from hive.users import OWNER, id_for_token
    r = role(token)
    return None if r is None else OWNER if r == "owner" else id_for_token(token)


def valid(token: str | None) -> bool:
    return role(token) is not None


def _given(authorization: str | None, token: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return token


def require(authorization: str | None = Header(default=None), token: str | None = Query(default=None)) -> None:
    """FastAPI dependency: accept 'Authorization: Bearer <token>' or ?token= (owner or user)."""
    if not valid(_given(authorization, token)):
        raise HTTPException(401, "Missing or wrong access token")


def require_owner(authorization: str | None = Header(default=None), token: str | None = Query(default=None)) -> None:
    """Credentials and the Hive's setup: the owner only."""
    r = role(_given(authorization, token))
    if r is None:
        raise HTTPException(401, "Missing or wrong access token")
    if r != "owner":
        raise HTTPException(403, "Only the Hive's owner can do that")
