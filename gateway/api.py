"""Mnx Hive gateway: serves the web app and streams Hive events over WebSocket.

Run:  uvicorn gateway.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hive import queen
from hive.core import Task, now

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DATA = ROOT / "data"
STATE_FILE = DATA / "hive.json"

DEFAULT_BEES = [
    {"id": "chat", "name": "Chat", "skill": "Talks with you and answers questions", "tools": [], "approval": False, "builtin": True},
    {"id": "memory", "name": "Memory", "skill": "Saves facts and recalls them before answering", "tools": ["memory"], "approval": False, "builtin": True},
]
DEFAULT_TOOLS = [
    {"id": "web_search", "name": "Web search", "description": "Search the web and read pages"},
    {"id": "google_drive", "name": "Google Drive", "description": "Read and save your files"},
    {"id": "gmail", "name": "Gmail", "description": "Read and draft email"},
    {"id": "google_calendar", "name": "Google Calendar", "description": "See and add events"},
    {"id": "telegram", "name": "Telegram", "description": "Chat with the Hive from Telegram"},
    {"id": "phone_drone", "name": "Phone Drone", "description": "Termux agent on your phone"},
]


# ---------- state (bees + tools persisted to data/hive.json) ----------
def load_state() -> dict:
    try:
        state = json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    bees = state.get("bees") or [dict(b) for b in DEFAULT_BEES]
    connected = set(state.get("connected_tools", []))
    return {"bees": bees, "connected_tools": connected}


def save_state() -> None:
    DATA.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps({
        "bees": STATE["bees"],
        "connected_tools": sorted(STATE["connected_tools"]),
    }, indent=2))


STATE = load_state()
BEE_STATUS: dict[str, str] = {}  # bee id → busy | failed (default idle / scheduled)
EVENT_LOG: deque[dict] = deque(maxlen=500)
CLIENTS: set[WebSocket] = set()
TASKS: dict[str, tuple[Task, asyncio.Task]] = {}


def bee_view(b: dict) -> dict:
    status = BEE_STATUS.get(b["id"]) or ("scheduled" if b.get("schedule") else "idle")
    return {**b, "status": status}


async def broadcast(event: dict) -> None:
    for ws in list(CLIENTS):
        try:
            await ws.send_json(event)
        except Exception:
            CLIENTS.discard(ws)


# ---------- HTTP API ----------
app = FastAPI(title="Mnx Hive")


@app.get("/api/health")
def health():
    from hive import brain
    return {"ok": True, "brain": brain.is_connected()}


@app.get("/api/bees")
def list_bees():
    return [bee_view(b) for b in STATE["bees"]]


class NewBee(BaseModel):
    name: str = Field(min_length=1, max_length=24)
    skill: str = Field(min_length=1, max_length=90)
    tools: list[str] = Field(default_factory=list, max_length=20)
    schedule: str | None = Field(default=None, max_length=40)
    approval: bool = True


@app.post("/api/bees")
async def add_bee(bee: NewBee):
    slug = re.sub(r"[^a-z0-9]+", "_", bee.name.lower()).strip("_") or "bee"
    if any(b["id"] == slug for b in STATE["bees"]):
        raise HTTPException(409, "A Bee with that name already exists")
    record = {"id": slug, **bee.model_dump(), "builtin": False}
    STATE["bees"].append(record)
    save_state()
    await broadcast({"type": "bees_changed"})
    return bee_view(record)


@app.delete("/api/bees/{bee_id}")
async def delete_bee(bee_id: str):
    bee = next((b for b in STATE["bees"] if b["id"] == bee_id), None)
    if not bee:
        raise HTTPException(404, "No such Bee")
    if bee.get("builtin"):
        raise HTTPException(400, "Built-in Bees can't be removed")
    STATE["bees"].remove(bee)
    save_state()
    await broadcast({"type": "bees_changed"})
    return {"ok": True}


@app.get("/api/tools")
def list_tools():
    return [{**t, "connected": t["id"] in STATE["connected_tools"]} for t in DEFAULT_TOOLS]


class ToolToggle(BaseModel):
    connected: bool


@app.post("/api/tools/{tool_id}")
async def toggle_tool(tool_id: str, body: ToolToggle):
    if not any(t["id"] == tool_id for t in DEFAULT_TOOLS):
        raise HTTPException(404, "No such tool")
    (STATE["connected_tools"].add if body.connected else STATE["connected_tools"].discard)(tool_id)
    save_state()
    await broadcast({"type": "tools_changed"})
    return {"ok": True}


@app.get("/api/events")
def recent_events(limit: int = 50):
    limit = max(1, min(limit, 500))
    return list(EVENT_LOG)[-limit:]


# ---------- WebSocket ----------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    CLIENTS.add(ws)

    async def emit(event: dict) -> None:
        EVENT_LOG.append(event)
        try:
            await ws.send_json(event)
        except Exception:
            pass

    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")

            if kind == "user_message":
                text = str(msg.get("text", ""))[:8000].strip()
                if not text:
                    continue
                task_id = str(msg.get("task") or f"t_{uuid.uuid4().hex[:10]}")[:40]
                task = Task(task_id, str(msg.get("chat", ""))[:40], text, emit)
                TASKS[task_id] = (task, asyncio.create_task(run_task(task)))

            elif kind == "action":
                entry = TASKS.get(str(msg.get("task")))
                if entry:
                    values = msg.get("values") if isinstance(msg.get("values"), dict) else {}
                    entry[0].resolve(str(msg.get("ref")), str(msg.get("action")), values)

            elif kind == "stop":
                entry = TASKS.pop(str(msg.get("task")), None)
                if entry:
                    entry[1].cancel()
    except (WebSocketDisconnect, json.JSONDecodeError, RuntimeError):
        pass
    finally:
        CLIENTS.discard(ws)


async def run_task(task: Task) -> None:
    try:
        await queen.handle(task)
        await task.emit("task_done")
    except asyncio.CancelledError:
        EVENT_LOG.append({"task": task.id, "type": "stopped", "text": "Stopped by you", "at": now()})
    except Exception as exc:  # report, don't crash the gateway
        await task.emit("task_error", f"The Hive hit an error: {exc}")
    finally:
        TASKS.pop(task.id, None)


# ---------- web app ----------
@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
