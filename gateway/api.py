"""Mnx Hive gateway: serves the web app, streams Hive events over WebSocket,
and exposes the Connector Hub, the Browser Bee and the Phone Drone.

Run:  uvicorn gateway.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gateway import auth
from hive import phone, queen
from hive.browser import BrowserError, stop_playwright
from hive.cells import cells
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
    status = BEE_STATUS.get(b["id"]) or cells.busy(b["id"]) or ("scheduled" if b.get("schedule") else "idle")
    cell = cells.cells.get(b["id"])
    return {**b, "status": status, "cell": cell.view() if cell else None}


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
    loop = asyncio.get_running_loop()
    cells.on_change = lambda cell: loop.create_task(broadcast({"type": "bees_changed", "bee": cell.id}))
    await cells.start(BEES)
    yield
    health.cancel()
    await cells.shutdown()
    await stop_playwright()


app = FastAPI(title="Mnx Hive", lifespan=lifespan)
api = APIRouter(prefix="/api", dependencies=[Depends(auth.require)])


@app.get("/api/ping")
def ping():
    """Unauthenticated: lets the web app tell 'server down' from 'wrong token'."""
    return {"ok": True}


@api.get("/health")
def health():
    from hive import brain
    return {"ok": True, "brain": brain.is_connected(), "phones": len(phone.PHONES), "cells": cells.mode}


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
    cell = cells.get(slug, bee.name)
    asyncio.create_task(_boot_cell(cell))
    log(bee.name, "cell_created", f"{cells.mode} Cell")
    await broadcast({"type": "bees_changed"})
    return bee_view(record)


async def _boot_cell(cell) -> None:
    try:
        await cell.ensure_up()
    except Exception as exc:
        log(cell.name, "cell_error", str(exc))
    await broadcast({"type": "bees_changed", "bee": cell.id})


@api.delete("/bees/{bee_id}")
async def delete_bee(bee_id: str):
    bee = next((b for b in BEES if b["id"] == bee_id), None)
    if not bee:
        raise HTTPException(404, "No such Bee")
    if bee.get("builtin"):
        raise HTTPException(400, "Built-in Bees can't be removed")
    BEES.remove(bee)
    save_bees()
    await cells.remove(bee_id)
    log(bee["name"], "cell_deleted", "container, files, browser storage and jobs removed")
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


def _bee(bee_id: str) -> dict:
    bee = next((b for b in BEES if b["id"] == bee_id), None)
    if not bee:
        raise HTTPException(404, "No such Bee")
    return bee


def _cell(bee_id: str):
    bee = _bee(bee_id)
    return cells.get(bee["id"], bee["name"])


@api.get("/browser")
async def browser_state(bee: str = "browser"):
    return await _cell(bee).browser.state()


@api.post("/browser")
async def browser_do(body: BrowserAction, bee: str = "browser"):
    """Drive a Bee's own browser (default: the Browser Bee)."""
    cell = _cell(bee)
    args = body.model_dump(exclude_none=True)
    action = args.pop("action")
    BEE_STATUS[cell.id] = "busy"
    try:
        result = await cell.browser.do(action, **args)
        if action not in ("screenshot", "read", "elements"):
            log(cell.name, action, args.get("url") or args.get("text") or args.get("key") or "")
        return result
    except BrowserError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        BEE_STATUS.pop(cell.id, None)


# ---------- Cells: each Bee's own always-on workspace ----------
@api.get("/cells/{bee_id}")
async def cell_info(bee_id: str):
    cell = _cell(bee_id)
    await cell.container_status()
    return {"cell": cell.view(), "jobs": cell.job_views(), "disk": cell.disk_usage(), "bee": _bee(bee_id)}


@api.post("/cells/{bee_id}/start")
async def cell_start(bee_id: str):
    cell = _cell(bee_id)
    try:
        await cell.ensure_up()
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    return cell.view()


class ExecBody(BaseModel):
    command: str = Field(min_length=1, max_length=4000)
    timeout: float = Field(default=60, ge=1, le=600)


@api.post("/cells/{bee_id}/exec")
async def cell_exec(bee_id: str, body: ExecBody):
    cell = _cell(bee_id)
    code, out = await cell.exec(body.command, body.timeout)
    log(cell.name, "exec", body.command[:120])
    return {"exit": code, "output": out[-100_000:]}


class NewJob(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    command: str = Field(min_length=1, max_length=4000)
    schedule: str = Field(default="@manual", max_length=60)


@api.post("/cells/{bee_id}/jobs")
async def job_add(bee_id: str, body: NewJob):
    cell = _cell(bee_id)
    try:
        job = cell.add_job(body.name, body.command, body.schedule.strip())
    except ValueError as exc:
        raise HTTPException(400, f"Bad schedule: {exc}") from exc
    log(cell.name, "job_added", f"{body.name} ({body.schedule})")
    await broadcast({"type": "bees_changed", "bee": cell.id})
    return job


def _job(cell, job_id: str) -> dict:
    if job_id not in cell.jobs:
        raise HTTPException(404, "No such job")
    return cell.jobs[job_id]


@api.post("/cells/{bee_id}/jobs/{job_id}/run")
async def job_run(bee_id: str, job_id: str):
    cell = _cell(bee_id)
    job = _job(cell, job_id)
    if job_id in cell.running:
        raise HTTPException(409, "Already running")
    if job["schedule"] == "@always":
        await cell.set_enabled(job_id, True)
    else:
        asyncio.create_task(cell.run_job(job_id))
    log(cell.name, "job_run", job["name"])
    return {"ok": True}


@api.post("/cells/{bee_id}/jobs/{job_id}/stop")
async def job_stop(bee_id: str, job_id: str):
    cell = _cell(bee_id)
    job = _job(cell, job_id)
    if job["schedule"] == "@always":
        await cell.set_enabled(job_id, False)
    else:
        await cell.stop_job(job_id)
    return {"ok": True}


class Toggle(BaseModel):
    enabled: bool


@api.post("/cells/{bee_id}/jobs/{job_id}/enabled")
async def job_enable(bee_id: str, job_id: str, body: Toggle):
    cell = _cell(bee_id)
    _job(cell, job_id)
    await cell.set_enabled(job_id, body.enabled)
    await broadcast({"type": "bees_changed", "bee": cell.id})
    return {"ok": True}


@api.delete("/cells/{bee_id}/jobs/{job_id}")
async def job_delete(bee_id: str, job_id: str):
    cell = _cell(bee_id)
    _job(cell, job_id)
    await cell.delete_job(job_id)
    await broadcast({"type": "bees_changed", "bee": cell.id})
    return {"ok": True}


@api.get("/cells/{bee_id}/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(bee_id: str, job_id: str, tail: int = 64_000):
    cell = _cell(bee_id)
    _job(cell, job_id)
    try:
        data = cell.log_path(job_id).read_bytes()
    except FileNotFoundError:
        return ""
    return data[-max(1, min(tail, 1_000_000)):].decode(errors="replace")


def _path(cell, path: str):
    try:
        return cell.safe_path(path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@api.get("/cells/{bee_id}/files")
def files_list(bee_id: str, path: str = ""):
    cell = _cell(bee_id)
    p = _path(cell, path)
    if not p.is_dir():
        raise HTTPException(404, "Not a folder")
    return cell.list_files(path)


@api.get("/cells/{bee_id}/file")
def file_get(bee_id: str, path: str):
    cell = _cell(bee_id)
    p = _path(cell, path)
    if not p.is_file():
        raise HTTPException(404, "No such file")
    return FileResponse(p, filename=p.name)


@api.put("/cells/{bee_id}/file")
async def file_put(bee_id: str, path: str, request: Request):
    cell = _cell(bee_id)
    p = _path(cell, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with open(p, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > 200 * 1024 * 1024:
                    raise HTTPException(413, "Files up to 200 MB")
                f.write(chunk)
    except PermissionError as exc:
        raise HTTPException(403, "The Bee owns that file; change it from the terminal") from exc
    except HTTPException:
        p.unlink(missing_ok=True)
        raise
    return {"ok": True, "size": size}


@api.delete("/cells/{bee_id}/file")
async def file_delete(bee_id: str, path: str):
    cell = _cell(bee_id)
    p = _path(cell, path)
    if p == cell.work.resolve():
        raise HTTPException(400, "Can't delete the Cell's root folder")
    rel = str(p.relative_to(cell.work.resolve()))
    # The container may own the file, so delete through the Cell.
    code, out = await cell.exec(f"rm -rf -- {shlex.quote(rel)}", 60)
    if code != 0:
        raise HTTPException(400, out.strip()[:300] or "Couldn't delete")
    return {"ok": True}


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
@app.websocket("/ws/term/{bee_id}")
async def term_ws(ws: WebSocket, bee_id: str):
    """A Bee's terminal. The shell keeps running after you disconnect."""
    await ws.accept()
    if not auth.valid(ws.query_params.get("token")):
        await ws.close(code=4401)
        return
    bee = next((b for b in BEES if b["id"] == bee_id), None)
    if not bee:
        await ws.close(code=4404)
        return
    try:
        term = await cells.get(bee_id, bee["name"]).terminal_session()
    except Exception as exc:
        await ws.send_text(json.dumps({"t": "error", "d": str(exc)}))
        await ws.close()
        return
    queue: asyncio.Queue = asyncio.Queue()
    term.listeners.add(queue)
    await ws.send_bytes(bytes(term.buffer))

    async def pump():
        while True:
            data = await queue.get()
            while not queue.empty() and len(data) < 65536:
                data += queue.get_nowait()
            await ws.send_bytes(data)

    sender = asyncio.create_task(pump())
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("t") == "in":
                term.write(str(msg.get("d", "")).encode())
            elif msg.get("t") == "resize":
                term.resize(int(msg.get("cols", 80)), int(msg.get("rows", 24)))
    except (WebSocketDisconnect, json.JSONDecodeError, RuntimeError, ValueError):
        pass
    finally:
        sender.cancel()
        term.listeners.discard(queue)


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
