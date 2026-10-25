"""Sign in to your own Google account for the Gmail and Drive agents.

MNX ships no API key and talks to no server of its own. You create a free
OAuth client in *your* Google Cloud project, and the sign-in runs on your
machine: your browser opens Google's consent page, Google redirects back to
a one-shot server on ``127.0.0.1``, and the token is saved to
``~/.mnx/google/token.json`` (readable only by you). Only the standard
library is used: the OAuth 2.0 installed-app flow with PKCE.

One-time setup::

    1. https://console.cloud.google.com → create a project
    2. APIs & Services → Library → enable "Gmail API" and "Google Drive API"
    3. OAuth consent screen → External, add yourself as a test user
    4. Credentials → Create credentials → OAuth client ID → Desktop app → Download JSON
    5. save it as ~/.mnx/google/credentials.json

    python -m mnx.agent.tools.google login         # Gmail + Drive
    python -m mnx.agent.tools.google login --gmail # just Gmail
    python -m mnx.agent.tools.google status
    python -m mnx.agent.tools.google logout        # deletes the token and revokes it

Scopes are the narrowest that work: Gmail read-only plus compose (to save
drafts; MNX never sends mail), and Drive read-only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.server
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Dict, Iterable, List, Optional

AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
REVOKE_URI = "https://oauth2.googleapis.com/revoke"

SCOPES = {
    "gmail": ["https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/gmail.compose"],
    "drive": ["https://www.googleapis.com/auth/drive.readonly"],
}

SETUP_HELP = (
    "To use Gmail or Drive, MNX needs your own Google sign-in (no API key is bundled). One-time setup: "
    "create a Desktop OAuth client in Google Cloud, save its JSON as ~/.mnx/google/credentials.json, "
    "then run: python -m mnx.agent.tools.google login. Details are in docs/tools.md."
)


class NotSignedIn(RuntimeError):
    pass


def default_dir() -> Path:
    return Path(os.environ.get("MNX_GOOGLE_DIR", "~/.mnx/google")).expanduser()


def _post_form(url: str, data: Dict[str, str]) -> Dict:
    req = urllib.request.Request(url, urllib.parse.urlencode(data).encode(), {"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"Google said {e.code}: {detail}") from e


class GoogleAuth:
    """Your OAuth client and token, kept in ``~/.mnx/google``."""

    def __init__(self, folder: Optional[Path] = None):
        self.dir = Path(folder).expanduser() if folder else default_dir()
        self.token_path = self.dir / "token.json"
        self.creds_path = self.dir / "credentials.json"
        self._token: Optional[Dict] = None

    # ----------------------------------------------------------- files
    def client(self) -> Dict[str, str]:
        if not self.creds_path.exists():
            raise NotSignedIn(SETUP_HELP)
        raw = json.loads(self.creds_path.read_text())
        c = raw.get("installed") or raw.get("web") or raw
        if "client_id" not in c:
            raise NotSignedIn(f"{self.creds_path} doesn't look like an OAuth client file. " + SETUP_HELP)
        return c

    def token(self) -> Dict:
        if self._token is None:
            if not self.token_path.exists():
                self.client()  # no OAuth client yet: explain the one-time setup first
                raise NotSignedIn("You're not signed in to Google yet. Run: python -m mnx.agent.tools.google login")
            self._token = json.loads(self.token_path.read_text())
        return self._token

    def _save(self, tok: Dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.token_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(tok))
        os.chmod(tmp, 0o600)
        tmp.replace(self.token_path)
        self._token = tok

    def scopes(self) -> List[str]:
        try:
            return self.token().get("scope", "").split()
        except NotSignedIn:
            return []

    def has(self, service: str) -> bool:
        granted = set(self.scopes())
        return bool(granted) and all(s in granted for s in SCOPES[service][:1])

    # ----------------------------------------------------------- tokens
    def access_token(self) -> str:
        tok = self.token()
        if tok.get("expires_at", 0) - 60 < time.time():
            if not tok.get("refresh_token"):
                raise NotSignedIn("Your Google sign-in expired. Run: python -m mnx.agent.tools.google login")
            c = self.client()
            new = _post_form(c.get("token_uri", TOKEN_URI), {
                "client_id": c["client_id"], "client_secret": c.get("client_secret", ""),
                "refresh_token": tok["refresh_token"], "grant_type": "refresh_token"})
            tok = {**tok, **new, "expires_at": time.time() + int(new.get("expires_in", 3600))}
            self._save(tok)
        return tok["access_token"]

    def login(self, services: Iterable[str] = ("gmail", "drive"), open_browser: bool = True, timeout: float = 300) -> Dict:
        """Run the loopback + PKCE flow; returns the saved token."""
        c = self.client()
        scopes = sorted({s for svc in services for s in SCOPES[svc]})
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        state = secrets.token_urlsafe(16)
        result: Dict[str, str] = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server API
                qs = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
                if qs.get("state") == state:
                    result.update(qs)
                ok = "code" in result
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                msg = "MNX is signed in. You can close this tab." if ok else "Sign-in failed or was cancelled. You can close this tab."
                self.wfile.write(f"<html><body style='font-family:sans-serif'><h3>{msg}</h3></body></html>".encode())

            def log_message(self, *a):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        redirect = f"http://127.0.0.1:{server.server_port}/"
        url = c.get("auth_uri", AUTH_URI) + "?" + urllib.parse.urlencode({
            "client_id": c["client_id"], "redirect_uri": redirect, "response_type": "code", "scope": " ".join(scopes),
            "state": state, "code_challenge": challenge, "code_challenge_method": "S256",
            "access_type": "offline", "prompt": "consent"})
        print("Opening Google sign-in in your browser. If it doesn't open, visit:\n" + url)
        if open_browser:
            webbrowser.open(url)
        deadline = time.time() + timeout
        server.timeout = 1
        while "code" not in result and "error" not in result and time.time() < deadline:
            server.handle_request()
        server.server_close()
        if "code" not in result:
            raise RuntimeError(f"sign-in didn't finish: {result.get('error', 'timed out')}")
        tok = _post_form(c.get("token_uri", TOKEN_URI), {
            "client_id": c["client_id"], "client_secret": c.get("client_secret", ""), "code": result["code"],
            "code_verifier": verifier, "redirect_uri": redirect, "grant_type": "authorization_code"})
        tok["expires_at"] = time.time() + int(tok.get("expires_in", 3600))
        old = self.token_path.exists() and json.loads(self.token_path.read_text())
        if old and "refresh_token" not in tok and old.get("refresh_token"):
            tok["refresh_token"] = old["refresh_token"]
        self._save(tok)
        return tok

    def logout(self) -> None:
        if not self.token_path.exists():
            return
        tok = json.loads(self.token_path.read_text())
        try:
            _post_form(REVOKE_URI, {"token": tok.get("refresh_token") or tok.get("access_token", "")})
        except Exception:
            pass  # deleting the local token is what matters
        self.token_path.unlink()
        self._token = None


class GoogleClient:
    """Tiny JSON-over-HTTPS client for Google's REST APIs."""

    def __init__(self, auth: Optional[GoogleAuth] = None):
        self.auth = auth or GoogleAuth()

    def request(self, method: str, url: str, params: Optional[Dict] = None, body: Optional[Dict] = None, raw: bool = False):
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params, doseq=True)
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {self.auth.access_token()}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data, headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                payload = r.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail)["error"]["message"]
            except Exception:
                detail = detail[:300]
            if e.code in (401, 403):
                raise NotSignedIn(f"Google refused access ({e.code}: {detail}). Try: python -m mnx.agent.tools.google login") from e
            raise RuntimeError(f"Google API error {e.code}: {detail}") from e
        if raw:
            return payload.decode("utf-8", errors="replace")
        return json.loads(payload or b"{}")

    def get(self, url: str, **params):
        return self.request("GET", url, params)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m mnx.agent.tools.google", description="Sign MNX in to your own Google account.")
    p.add_argument("command", choices=["login", "logout", "status"])
    p.add_argument("--gmail", action="store_true", help="only ask for Gmail access")
    p.add_argument("--drive", action="store_true", help="only ask for Drive access")
    p.add_argument("--dir", default=None, help="where credentials.json and token.json live (default ~/.mnx/google)")
    p.add_argument("--no-browser", action="store_true", help="print the sign-in link instead of opening a browser")
    args = p.parse_args(argv)
    auth = GoogleAuth(Path(args.dir) if args.dir else None)
    try:
        if args.command == "login":
            services = [s for s in ("gmail", "drive") if getattr(args, s)] or ["gmail", "drive"]
            auth.login(services, open_browser=not args.no_browser)
            print(f"Signed in. Token saved to {auth.token_path} (scopes: {', '.join(services)}).")
        elif args.command == "logout":
            auth.logout()
            print("Signed out; the local token is deleted.")
        else:
            granted = [svc for svc in SCOPES if auth.has(svc)]
            print(f"credentials: {'found' if auth.creds_path.exists() else 'missing'} ({auth.creds_path})")
            print(f"signed in for: {', '.join(granted) or 'nothing'}")
    except (NotSignedIn, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
