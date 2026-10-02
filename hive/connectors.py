"""Connector Hub: one catalog over every layer the Hive can use to reach an app.

Layers (from the blueprint):
  1. Native Bees      hand-built connectors (listed here, built one by one)
  2. MCP registry     live search of registry.modelcontextprotocol.io; remote servers plug in directly
  3. Zapier MCP       paste your Zapier MCP URL (it is a remote MCP server)
  4. Pipedream        paste your Pipedream MCP URL (same)
  5. Auto-connector   written by Coder Bee later (needs the Mnx brain)
  6. Browser / Phone  any website through the Browser Bee, any Android app through the Phone Drone

Tool descriptions from outside servers are data, never instructions. New tools
start in the Ask lane.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import httpx

DATA = Path(__file__).resolve().parent.parent / "data"
CONN_FILE = DATA / "connectors.json"
VAULT_FILE = DATA / "vault.json"
REGISTRY = os.getenv("MNX_MCP_REGISTRY", "https://registry.modelcontextprotocol.io").rstrip("/")

LANES = ("go", "ask", "you")

LAYERS = [
    {"n": 1, "name": "Native Bees", "reach": "~20 key apps", "how": "Hand-built and tested"},
    {"n": 2, "name": "MCP registry", "reach": "18,000+ servers listed", "how": "Remote servers plug in directly"},
    {"n": 3, "name": "Zapier MCP", "reach": "9,000+ apps", "how": "Paste your Zapier MCP URL"},
    {"n": 4, "name": "Pipedream", "reach": "3,000+ APIs", "how": "Paste your Pipedream MCP URL"},
    {"n": 5, "name": "Auto-connector", "reach": "Any app with API docs", "how": "Coder Bee writes it (needs the Mnx brain)"},
    {"n": 6, "name": "Browser + Phone", "reach": "Any website, any Android app", "how": "Browser Bee and Phone Drone use them like you do"},
]

NATIVE = [
    ("telegram", "Telegram", "Chat"), ("whatsapp", "WhatsApp", "Chat"), ("gmail", "Gmail", "Email"),
    ("google_calendar", "Google Calendar", "Calendar"), ("google_drive", "Google Drive", "Files"),
    ("swiggy", "Swiggy", "Food"), ("instamart", "Swiggy Instamart", "Groceries"), ("dineout", "Swiggy Dineout", "Tables"),
    ("zomato", "Zomato", "Food"), ("upi", "UPI (GPay / PhonePe)", "Payments"), ("uber", "Uber", "Rides"),
    ("ola", "Ola", "Rides"), ("rapido", "Rapido", "Rides"), ("makemytrip", "MakeMyTrip", "Travel"),
    ("irctc", "IRCTC", "Trains"), ("youtube", "YouTube", "Video"), ("spotify", "Spotify", "Music"),
    ("github", "GitHub", "Code"), ("notion", "Notion", "Notes"), ("sms", "SMS", "Phone"),
]


# ---------- storage ----------
def _read(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write(path: Path, data, private: bool = False) -> None:
    DATA.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    if private:
        os.chmod(tmp, 0o600)
    tmp.replace(path)


class Hub:
    def __init__(self) -> None:
        self.conns: dict[str, dict] = {c["id"]: c for c in _read(CONN_FILE, [])}
        self._vault: dict[str, dict] = _read(VAULT_FILE, {})
        self._search_cache: dict[str, tuple[float, list]] = {}

    def save(self) -> None:
        _write(CONN_FILE, list(self.conns.values()))
        _write(VAULT_FILE, self._vault, private=True)

    # ----- public views (never include secrets) -----
    def list(self) -> list[dict]:
        return [self.view(c) for c in self.conns.values()]

    def view(self, c: dict) -> dict:
        return {k: v for k, v in c.items()} | {"has_secrets": bool(self._vault.get(c["id"]))}

    # ----- catalog search across layers -----
    async def search(self, q: str, phone_apps: list[dict]) -> dict:
        q = q.strip()
        ql = q.lower()
        native = [
            {"id": f"native:{i}", "name": n, "category": cat, "layer": 1, "status": "planned"}
            for i, n, cat in NATIVE if not ql or ql in n.lower() or ql in cat.lower()
        ]
        connected = [self.view(c) for c in self.conns.values() if not ql or ql in c["name"].lower()]
        registry: list[dict] = []
        registry_error = None
        if q:
            try:
                registry = await self.registry_search(q)
            except Exception as exc:  # network down etc.
                registry_error = f"MCP registry unreachable ({type(exc).__name__})"
        phone = [a for a in phone_apps if ql and (ql in a["label"].lower() or ql in a["package"].lower())][:20]
        browser = [{"name": q, "url": _guess_url(q)}] if q else []
        return {
            "query": q, "connected": connected, "native": native, "registry": registry,
            "registry_error": registry_error, "browser": browser, "phone": phone,
        }

    async def registry_search(self, q: str) -> list[dict]:
        key = q.lower()
        hit = self._search_cache.get(key)
        if hit and time.time() - hit[0] < 600:
            return hit[1]
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(f"{REGISTRY}/v0/servers", params={"search": q, "limit": 40})
            r.raise_for_status()
            data = r.json()
        out: dict[str, dict] = {}
        for item in data.get("servers", []):
            s = item.get("server", item)
            meta = (item.get("_meta") or {}).get("io.modelcontextprotocol.registry/official", {})
            if meta and meta.get("isLatest") is False:
                continue
            name = s.get("name", "")
            remotes = [
                {
                    "type": rm.get("type"),
                    "url": rm.get("url"),
                    "headers": [
                        {k: h.get(k) for k in ("name", "description", "isRequired", "isSecret")}
                        for h in rm.get("headers") or []
                    ],
                }
                for rm in s.get("remotes") or []
                if rm.get("type") in ("streamable-http", "sse") and rm.get("url")
            ]
            out[name] = {
                "id": f"mcp:{name}",
                "name": s.get("title") or name.split("/")[-1],
                "registry_name": name,
                "description": (s.get("description") or "")[:240],
                "version": s.get("version"),
                "repo": (s.get("repository") or {}).get("url"),
                "remotes": remotes,
                "local_only": not remotes,
                "layer": 2,
            }
        result = list(out.values())
        self._search_cache[key] = (time.time(), result)
        return result

    # ----- connect / test / remove -----
    async def add(self, name: str, url: str, transport: str, headers: dict[str, str], layer: int, source: str | None) -> dict:
        cid = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "connector"
        if cid in self.conns:
            cid = f"{cid}-{uuid.uuid4().hex[:4]}"
        conn = {
            "id": cid, "name": name[:60], "url": url, "transport": transport, "layer": layer,
            "source": source, "tools": [], "status": "checking", "error": None, "checked_at": None,
            "added_at": int(time.time()), "uses": 0,
        }
        self.conns[cid] = conn
        self._vault[cid] = {k: v for k, v in headers.items() if k and v}
        await self.test(cid)
        return self.view(conn)

    async def test(self, cid: str) -> dict:
        conn = self.conns[cid]
        old_lanes = {t["name"]: t.get("lane", "ask") for t in conn.get("tools", [])}
        try:
            tools = await asyncio.wait_for(self._list_tools(conn), timeout=30)
            conn["tools"] = [
                {
                    "name": t.name,
                    "description": (t.description or "")[:400],
                    "schema": t.inputSchema,
                    "lane": old_lanes.get(t.name, "ask"),
                }
                for t in tools
            ]
            conn["status"], conn["error"] = "ok", None
        except Exception as exc:
            conn["status"], conn["error"] = "failed", _short_error(exc)
        conn["checked_at"] = int(time.time())
        self.save()
        return self.view(conn)

    def remove(self, cid: str) -> None:
        self.conns.pop(cid, None)
        self._vault.pop(cid, None)
        self.save()

    def set_lane(self, cid: str, tool: str, lane: str) -> None:
        if lane not in LANES:
            raise ValueError("lane must be go, ask or you")
        for t in self.conns[cid]["tools"]:
            if t["name"] == tool:
                t["lane"] = lane
        self.save()

    async def call(self, cid: str, tool: str, args: dict[str, Any]) -> dict:
        conn = self.conns[cid]
        async with self._session(conn) as s:
            res = await asyncio.wait_for(s.call_tool(tool, args), timeout=60)
        conn["uses"] = conn.get("uses", 0) + 1
        self.save()
        parts = []
        for c in res.content:
            if getattr(c, "type", "") == "text":
                parts.append({"type": "text", "text": c.text})
            else:
                parts.append({"type": getattr(c, "type", "other")})
        return {"is_error": bool(res.isError), "content": parts, "structured": getattr(res, "structuredContent", None)}

    async def health_loop(self, every: int = 3600) -> None:
        while True:
            await asyncio.sleep(every)
            for cid in list(self.conns):
                try:
                    await self.test(cid)
                except Exception:
                    pass

    # ----- MCP plumbing -----
    @asynccontextmanager
    async def _session(self, conn: dict):
        from mcp import ClientSession

        headers = self._vault.get(conn["id"]) or None
        if conn["transport"] == "sse":
            from mcp.client.sse import sse_client
            ctx = sse_client(conn["url"], headers=headers)
        else:
            from mcp.client.streamable_http import streamablehttp_client
            ctx = streamablehttp_client(conn["url"], headers=headers)
        async with ctx as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                yield session

    async def _list_tools(self, conn: dict):
        async with self._session(conn) as s:
            return (await s.list_tools()).tools


def _guess_url(q: str) -> str:
    q = q.strip()
    if re.match(r"^https?://", q):
        return q
    if re.match(r"^[\w-]+(\.[\w-]+)+(/.*)?$", q):
        return "https://" + q
    return "https://duckduckgo.com/?q=" + quote_plus(q)


def _short_error(exc: BaseException) -> str:
    # anyio wraps errors in groups; show the innermost one
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    msg = str(exc) or type(exc).__name__
    return f"{type(exc).__name__}: {msg}"[:300]


hub = Hub()
