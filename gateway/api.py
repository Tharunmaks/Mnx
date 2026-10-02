"""Mnx Hive gateway: serves the web app, streams Hive events over WebSocket,
and exposes the Connector Hub, the Browser Bee and the Phone Drone.

Run:  uvicorn gateway.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gateway import auth
from hive import phone, queen
from hive.browser import BrowserError, session as browser
from hive.connectors import LANES, LAYERS, hub
from hive.core import Task, now

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DATA = ROOT / "data"
STATE_FILE = DATA / "hive.json"

DEFAULT_BEES = [
    {"id": "chat", "name": "Chat", "skill": "Talks with you and answers questions", "tools": [], "approval": False, "builtin": True},
    {"id": "memory", "name": "Memory", "skill": "Saves facts and recalls them before answering", "tools": ["memory"], "approval": False, "builtin": True},
    {"id": "browser", "name": "Browser", "skill": "Uses any website like a person, in its own Chromium", "tools": ["browser"], "approval": True, "builtin": True},
    {"id": "phone", "name": "Phone", "skill": "Taps, types and opens apps on your paired phone", "tools": ["phone"], "approval": True, "builtin": True},
]


# ---------- state (bees persisted to data/hive.json) ----------
def load_bees() -> list[dict]:
    try:
        bees = json.loads(STATE_FILE.read_text()).get("bees") or []
    except (FileNotFoundError, json.JSONDecodeError):
        bees = []
    have = {b["id"] for b in bees}
    return [dict(b) for b in DEFAULT_BEES if b["id"] not in have] + bees


def save_bees() -> None:
    DATA.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps({"bees": BEES}, indent=2))


BEES = load_bees()
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


def log(bee: str, type: str, text: str) -> None:
    EVENT_LOG.append({"bee": bee, "type": type, "text": text[:300], "at": now()})


@asynccontextmanager
async def lifespan(app: FastAPI):
    health = asyncio.create_task(hub.health_loop())
    yield
    health.cancel()
    await browser.shutdown()


app = FastAPI(title="Mnx Hive", lifespan=lifespan)
api = APIRouter(prefix="/api", dependencies=[Depends(auth.require)])


@app.get("/api/ping")
def ping():
    """Unauthenticated: lets the web app tell 'server down' from 'wrong token'."""
    return {"ok": True}


@api.get("/health")
def health():
    from hive import brain
    return {"ok": True, "brain": brain.is_connected(), "phones": len(phone.PHONES), "browser": browser.running}


# ---------- Bees ----------
@api.get("/bees")
def list_bees():
    return [bee_view(b) for b in BEES]


class NewBee(BaseModel):
    name: str = Field(min_length=1, max_length=24)
    skill: str = Field(min_length=1, max_length=90)
    tools: list[str] = Field(default_factory=list, max_length=20)
    schedule: str | None = Field(default=None, max_length=40)
    approval: bool = True


@api.post("/bees")
async def add_bee(bee: NewBee):
    slug = re.sub(r"[^a-z0-9]+", "_", bee.name.lower()).strip("_") or "bee"
    if any(b["id"] == slug for b in BEES):
        raise HTTPException(409, "A Bee with that name already exists")
    record = {"id": slug, **bee.model_dump(), "builtin": False}
    BEES.append(record)
    save_bees()
    await broadcast({"type": "bees_changed"})
    return bee_view(record)


@api.delete("/bees/{bee_id}")
async def delete_bee(bee_id: str):
    bee = next((b for b in BEES if b["id"] == bee_id), None)
    if not bee:
        raise HTTPException(404, "No such Bee")
    if bee.get("builtin"):
        raise HTTPException(400, "Built-in Bees can't be removed")
    BEES.remove(bee)
    save_bees()
    await broadcast({"type": "bees_changed"})
    return {"ok": True}


@api.get("/events")
def recent_events(limit: int = 50):
    limit = max(1, min(limit, 500))
    return list(EVENT_LOG)[-limit:]


# ---------- Connector Hub ----------
@api.get("/connectors")
def list_connectors():
    return {"layers": LAYERS, "connected": hub.list(), "phones": len(phone.PHONES)}


@api.get("/connectors/search")
async def search_connectors(q: str = ""):
    return await hub.search(q[:100], phone.all_apps())


class NewConnector(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    url: str = Field(pattern=r"^https?://", max_length=500)
    transport: str = Field(default="streamable-http", pattern=r"^(streamable-http|sse)$")
    headers: dict[str, str] = Field(default_factory=dict)
    layer: int = Field(default=2, ge=1, le=6)
    source: str | None = Field(default=None, max_length=200)


@api.post("/connectors")
async def add_connector(c: NewConnector):
    view = await hub.add(c.name, c.url, c.transport, c.headers, c.layer, c.source)
    log("Hub", "connector_added", f"{c.name}: {view['status']}")
    await broadcast({"type": "tools_changed"})
    return view


def _conn_or_404(cid: str) -> None:
    if cid not in hub.conns:
        raise HTTPException(404, "No such connector")


@api.post("/connectors/{cid}/test")
async def test_connector(cid: str):
    _conn_or_404(cid)
    view = await hub.test(cid)
    await broadcast({"type": "tools_changed"})
    return view


@api.delete("/connectors/{cid}")
async def remove_connector(cid: str):
    _conn_or_404(cid)
    hub.remove(cid)
    await broadcast({"type": "tools_changed"})
    return {"ok": True}


class LaneChange(BaseModel):
    tool: str
    lane: str


@api.post("/connectors/{cid}/lane")
def set_lane(cid: str, body: LaneChange):
    _conn_or_404(cid)
    if body.lane not in LANES:
        raise HTTPException(400, "lane must be go, ask or you")
    hub.set_lane(cid, body.tool, body.lane)
    return {"ok": True}


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


@api.post("/connectors/{cid}/call")
async def call_connector(cid: str, body: ToolCall):
    _conn_or_404(cid)
    try:
        result = await hub.call(cid, body.tool, body.args)
    except Exception as exc:
        raise HTTPException(502, f"Tool call failed: {exc}") from exc
    log("Hub", "tool_call", f"{cid}.{body.tool}")
    return result


# ---------- Browser Bee ----------
class BrowserAction(BaseModel):
    action: str = Field(pattern=r"^(goto|click|type|press|scroll|back|forward|reload|read|elements|screenshot|close)$")
    url: str | None = None
    x: float | None = None
    y: float | None = None
    text: str | None = Field(default=None, max_length=5000)
    selector: str | None = Field(default=None, max_length=300)
    key: str | None = Field(default=None, max_length=40)
    dy: float | None = None


@api.get("/browser")
async def browser_state():
    return await browser.state()


@api.post("/browser")
async def browser_do(body: BrowserAction):
    args = body.model_dump(exclude_none=True)
    action = args.pop("action")
    BEE_STATUS["browser"] = "busy"
    try:
        result = await browser.do(action, **args)
        if action not in ("screenshot", "read", "elements"):
            log("Browser", action, args.get("url") or args.get("text") or args.get("key") or "")
        return result
    except BrowserError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        BEE_STATUS.pop("browser", None)


# ---------- Phone Drone ----------
@api.get("/phones")
def list_phones():
    return [p.view() for p in phone.PHONES.values()]


class PhoneCommand(BaseModel):
    cmd: str = Field(max_length=40)
    args: dict[str, Any] = Field(default_factory=dict)
    confirmed: bool = False


@api.post("/phones/{phone_id}/cmd")
async def phone_cmd(phone_id: str, body: PhoneCommand):
    try:
        p = phone.get(phone_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    if body.cmd in phone.RISKY and not body.confirmed:
        raise HTTPException(428, f"'{body.cmd}' needs your confirmation")
    try:
        result = await p.call(body.cmd, body.args, timeout=60 if body.cmd in ("location", "ui_tree", "tap_text") else 25)
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, "The phone didn't answer in time") from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if body.cmd == "apps" and isinstance(result, list):
        p.apps = [{**a, "phone": p.id} for a in result]
    if body.cmd not in ("screenshot", "ui_tree", "apps", "battery", "screen_size"):
        log("Phone", body.cmd, json.dumps(body.args)[:200] if body.args else "")
    return {"result": result}


app.include_router(api)


# ---------- WebSockets ----------
@app.websocket("/ws/drone")
async def drone_ws(ws: WebSocket):
    await ws.accept()
    if not auth.valid(ws.query_params.get("token")):
        await ws.close(code=4401)
        return
    p = None
    try:
        hello = await asyncio.wait_for(ws.receive_json(), 15)
        if hello.get("type") != "hello":
            await ws.close(code=4400)
            return
        p = phone.Phone(ws, hello)
        old = phone.PHONES.get(p.id)
        if old:
            old.fail_all()
        phone.PHONES[p.id] = p
        log("Phone", "connected", f"{p.info.get('name')} ({p.info.get('backend')})")
        await broadcast({"type": "phones_changed"})
        if "apps" in p.capabilities:
            asyncio.create_task(_prefetch_apps(p))
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "reply":
                p.on_reply(msg)
    except (WebSocketDisconnect, asyncio.TimeoutError, json.JSONDecodeError, RuntimeError):
        pass
    finally:
        if p and phone.PHONES.get(p.id) is p:
            p.fail_all()
            del phone.PHONES[p.id]
            log("Phone", "disconnected", str(p.info.get("name")))
            await broadcast({"type": "phones_changed"})


async def _prefetch_apps(p: phone.Phone) -> None:
    try:
        apps = await p.call("apps", timeout=40)
        p.apps = [{**a, "phone": p.id} for a in apps]
    except Exception:
        pass


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    if not auth.valid(ws.query_params.get("token")):
        await ws.close(code=4401)
        return
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
@app.get("/drone.py")
def drone_script():
    """The Termux agent, so the phone can download it with curl."""
    return FileResponse(ROOT / "drone" / "drone.py", media_type="text/x-python")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
