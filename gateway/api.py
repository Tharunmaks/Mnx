"""Mnx Hive gateway: serves the web app, streams Hive events over WebSocket,
and exposes the Connector Hub, the Browser Bee and the Phone Drone.

Run:  uvicorn gateway.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import shutil
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel, Field

from gateway import auth
from hive import effort, phone, queen, status
from hive.browser import BrowserError, stop_playwright
from hive.cells import cells
from hive.connectors import LANES, LAYERS, hub
from hive.core import Task, now
from hive.lab import lab
from hive import system

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DATA = ROOT / "data"
STATE_FILE = DATA / "hive.json"

DEFAULT_BEES = [
    {"id": "chat", "name": "Chat", "skill": "Talks with you and answers questions", "tools": [], "approval": False, "builtin": True},
    {"id": "memory", "name": "Memory", "skill": "Saves facts and recalls them before answering", "tools": ["memory"], "approval": False, "builtin": True},
    {"id": "browser", "name": "Browser", "skill": "Uses any website like a person, in its own Chromium", "tools": ["browser"], "approval": True, "builtin": True},
    {"id": "phone", "name": "Phone", "skill": "Taps, types and opens apps on your paired phone", "tools": ["phone"], "approval": True, "builtin": True},
    {"id": "cloud", "name": "Cloud", "skill": "Trains AI models, keeps them and runs them as APIs, here or on any GPU server", "tools": ["lab"], "approval": True, "builtin": True},
    {"id": "reader", "name": "Reader", "skill": "Reads your request and works out what you want", "tools": [], "approval": False, "builtin": True},
    {"id": "architect", "name": "Architect", "skill": "Designs transformers for any size and estimates what training takes", "tools": ["lab"], "approval": False, "builtin": True},
    {"id": "data", "name": "Data", "skill": "Gathers training text: datasets, your files, web pages", "tools": ["browser", "lab"], "approval": False, "builtin": True},
    {"id": "gpu", "name": "GPU", "skill": "Opens cloud Docker and GPUs: your servers, clusters, or rented machines", "tools": ["lab"], "approval": True, "builtin": True},
    {"id": "coding", "name": "Coding", "skill": "Writes the files for a training run (train.py, serve.py, config)", "tools": ["lab"], "approval": False, "builtin": True},
    {"id": "eval", "name": "Eval", "skill": "Tests finished models with held-out prompts and reports back", "tools": ["lab"], "approval": False, "builtin": True},
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
    training = b["id"] == "cloud" and any(r["status"] in ("queued", "preparing", "running") for r in lab.runs.values())
    st = BEE_STATUS.get(b["id"]) or status.busy.get(b["id"]) or ("busy" if training else None) or cells.busy(b["id"]) or ("scheduled" if b.get("schedule") else "idle")
    cell = cells.cells.get(b["id"])
    return {**b, "status": st, "cell": cell.view() if cell else None}


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
    lab.on_change = lambda kind: loop.create_task(broadcast({"type": "lab_changed"}))
    lab.on_finish = lambda run: loop.create_task(_report(run))
    status.on_change = lambda: loop.create_task(broadcast({"type": "bees_changed"}))
    await lab.start()
    prepull = asyncio.create_task(_prepull_images())
    yield
    health.cancel()
    prepull.cancel()
    await lab.shutdown()
    await cells.shutdown()
    await stop_playwright()


async def _report(run: dict) -> None:
    """When a run started from the chat ends, the Eval Bee posts the final report into that chat."""
    chat = run.get("chat") or {}
    status.clear("cloud", "eval")
    if not chat.get("task"):
        return
    parent = chat.get("eval_parent")
    task_id = chat["task"]
    st = run.get("stages") or {}
    fmt = __import__("hive.architect", fromlist=["fmt_time"]).fmt_time

    def took(a: str, b: str) -> str | None:
        if a in st and b in st and st[b] >= st[a]:
            return fmt(st[b] - st[a])
        return None

    events: list[dict] = []

    def ev(type: str, text: str | None = None, bee: str | None = None, **f):
        e = {"task": task_id, "id": f"rp{uuid.uuid4().hex[:8]}", "type": type, "at": now(), **f}
        if text is not None:
            e["text"] = text
        if bee:
            e["bee"] = bee
        if parent:
            e["parent"] = parent
        events.append(e)

    res = run.get("result") or {}
    if run["status"] == "succeeded":
        d = took("data", "training")
        t = took("training", "saving") or took("training", "testing") or took("training", "end")
        x = took("testing", "end")
        if d:
            ev("done", f"Adding data to the model took {d}", bee="Data", status="done")
        if t:
            ev("done", f"Training took {t} · {res.get('tokens', 0):,} tokens", bee="Cloud", status="done")
        if res.get("test_prompts"):
            ev("done", f"Tested with {res['test_prompts']:,} prompts" + (f" in {x}" if x else "")
               + f": next-token accuracy {res.get('test_accuracy')}%, perplexity {res.get('perplexity')}", bee="Eval", status="done")
        details = [["Parameters", f"{res.get('params_text', '?')} ({res.get('params', 0):,})"],
                   ["Tokens trained", f"{res.get('tokens', 0):,}"], ["Training time", t or "—"]]
        if res.get("test_prompts"):
            details += [["Test prompts", f"{res['test_prompts']:,}"], ["Accuracy", f"{res.get('test_accuracy')}%"],
                        ["Perplexity", str(res.get("perplexity"))]]
        if run.get("nodes"):
            details.append(["Trained on", f"a cluster of {len(run['nodes'])} machines"])
        ev("result", service="Eval", title=f"Your model is ready: {res.get('params_text', '?')} parameters", badge="Ready",
           details=details, text="Press Run it on the training card above to talk to it; it also appears in the Lab's Models tab.")
        ev("done", "Stabilizing network · turning off Bees", bee="Queen", status="done")
    else:
        ev("error", f"Training {run['status']}: {run.get('error') or 'see the log in the Lab'}", bee="Cloud", status="error")
        ev("done", "Turning off Bees", bee="Queen", status="done")
    for e in events:
        EVENT_LOG.append(e)
        await broadcast(e)
    await broadcast({"type": "bees_changed"})


async def _prepull_images() -> None:
    """Download container images in the background so the first Cell or training run starts fast."""
    from hive.cells import IMAGE
    images = {IMAGE} | {r.get("image", "python:3.12-slim") for r in lab.recipes.values()}
    for image in sorted(images):
        proc = await asyncio.create_subprocess_exec("docker", "image", "inspect", image,
                                                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        if await proc.wait() != 0:
            proc = await asyncio.create_subprocess_exec("docker", "pull", "-q", image,
                                                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            await proc.wait()


app = FastAPI(title="Mnx Hive", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def cache_headers(request: Request, call_next):
    """Libraries and images are cached for a week; the app's own files are re-checked
    (cheap 304s via ETag) so an update shows up on the next page load."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    elif path.startswith(("/vendor/", "/img/")):
        response.headers["Cache-Control"] = "public, max-age=604800"
    else:
        response.headers["Cache-Control"] = "no-cache"
    return response

api = APIRouter(prefix="/api", dependencies=[Depends(auth.require)])


@app.get("/api/ping")
def ping():
    """Unauthenticated: lets the web app tell 'server down' from 'wrong token'."""
    return {"ok": True}


@api.get("/health")
def health():
    from hive import brain
    return {"ok": True, "brain": brain.is_connected(), "phones": len(phone.PHONES), "cells": cells.mode}


@api.get("/system")
async def system_stats():
    data = await asyncio.to_thread(system.stats)
    data["containers"] = {
        "cells": sum(1 for c in cells.cells.values() if c.container_state == "running"),
        "training": sum(1 for r in lab.runs.values() if r["status"] in ("preparing", "running")),
        "models": sum(1 for d in lab.deployments.values() if d["status"] in ("starting", "running")),
    }
    data["cell_mode"] = cells.mode
    data["gpu_docker"] = lab.gpu_local
    return data


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
    # Container state comes from the manager's periodic refresh (one `docker ps` for all Cells).
    return {"cell": cell.view(), "jobs": cell.job_views(), "disk": await cell.disk_usage_cached(), "bee": _bee(bee_id)}


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


# ---------- Cloud Bee: Model Lab ----------
UPLOADS = DATA / "lab" / "uploads"


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._")[:120]
    if not name:
        raise HTTPException(400, "Bad file name")
    return name


def _upload_path(upload_id: str) -> Path:
    if not re.match(r"^[a-f0-9]{16}$", upload_id or ""):
        raise HTTPException(400, "Bad upload id")
    folder = UPLOADS / upload_id
    files = list(folder.iterdir()) if folder.is_dir() else []
    if not files:
        raise HTTPException(404, "Upload not found (it may have been used already)")
    return files[0]


@api.get("/lab")
async def lab_overview():
    runs = sorted(lab.runs.values(), key=lambda r: r["created"], reverse=True)
    return {
        "recipes": list(lab.recipes.values()), "targets": lab.target_views(),
        "runs": [lab.run_view(r) for r in runs], "models": sorted(lab.models.values(), key=lambda m: m["created"], reverse=True),
        "deployments": lab.deployments, "secrets": lab.secret_names(), "hive_key": await lab.hive_key(),
        "gpu_local": lab.gpu_local,
    }


@api.put("/lab/uploads/{filename}")
async def lab_upload(filename: str, request: Request):
    """Stream one file to the server. Returns an id to use in a run or a model import."""
    name = _safe_name(filename)
    upload_id = uuid.uuid4().hex[:16]
    folder = UPLOADS / upload_id
    folder.mkdir(parents=True)
    size = 0
    try:
        with open(folder / name, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > 20 * 1024**3:
                    raise HTTPException(413, "Files up to 20 GB")
                f.write(chunk)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return {"id": upload_id, "name": name, "size": size}


class NewRun(BaseModel):
    recipe: str = Field(max_length=64)
    name: str = Field(default="", max_length=80)
    params: dict[str, Any] = Field(default_factory=dict)
    target: str = Field(default="local", max_length=64)
    nodes: list[str] | None = Field(default=None, max_length=64)
    uploads: list[str] = Field(default_factory=list, max_length=50)
    use_sample: bool = False
    gpu: bool = False
    cpus: float | None = Field(default=None, ge=0.5, le=256)
    memory_gb: float | None = Field(default=None, ge=0.5, le=2048)


@api.post("/lab/runs")
async def lab_new_run(body: NewRun):
    files = [_upload_path(u) for u in body.uploads]
    try:
        run = lab.create_run(body.recipe, body.name, body.params, body.target, files,
                             body.use_sample, body.gpu, body.cpus, body.memory_gb, nodes=body.nodes)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        for u in body.uploads:
            shutil.rmtree(UPLOADS / u, ignore_errors=True)
    log("Cloud", "training", f"{run['name']} ({run['recipe']}) on {run['target']}")
    return lab.run_view(run)


def _run_or_404(rid: str) -> dict:
    run = lab.runs.get(rid)
    if not run:
        raise HTTPException(404, "No such run")
    return run


@api.get("/lab/runs/{rid}")
def lab_run(rid: str):
    return lab.run_view(_run_or_404(rid), full=True)


@api.get("/lab/runs/{rid}/log")
def lab_run_log(rid: str, since: int = 0):
    _run_or_404(rid)
    data, size = lab.log_tail(rid, since)
    return Response(data, media_type="text/plain; charset=utf-8", headers={"X-Log-Size": str(size)})


@api.get("/lab/runs/{rid}/files")
def lab_run_files(rid: str):
    _run_or_404(rid)
    return lab.list_job_files(rid)


@api.get("/lab/runs/{rid}/file", response_class=PlainTextResponse)
def lab_run_file(rid: str, path: str):
    _run_or_404(rid)
    try:
        return lab.read_job_file(rid, path)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@api.post("/lab/runs/{rid}/stop")
async def lab_stop_run(rid: str):
    _run_or_404(rid)
    await lab.stop_run(rid)
    return {"ok": True}


@api.delete("/lab/runs/{rid}")
async def lab_delete_run(rid: str):
    _run_or_404(rid)
    try:
        await lab.delete_run(rid)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True}


def _model_or_404(mid: str) -> dict:
    model = lab.models.get(mid)
    if not model:
        raise HTTPException(404, "No such model")
    return model


class ImportModel(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    upload: str


@api.post("/lab/models/import")
async def lab_import(body: ImportModel):
    path = _upload_path(body.upload)
    try:
        model = await lab.import_model(body.name, "gguf", path, path.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        shutil.rmtree(UPLOADS / body.upload, ignore_errors=True)
    return model


@api.get("/lab/models/{mid}")
def lab_model(mid: str):
    return {"model": _model_or_404(mid), "deployment": lab.deployments.get(mid), "gpu_local": lab.gpu_local}


@api.delete("/lab/models/{mid}")
async def lab_delete_model(mid: str):
    _model_or_404(mid)
    await lab.delete_model(mid)
    return {"ok": True}


@api.get("/lab/models/{mid}/download")
async def lab_download_model(mid: str):
    model = _model_or_404(mid)
    from hive.lab import MODELS
    proc = await asyncio.create_subprocess_exec("tar", "-C", str(MODELS / mid), "-czf", "-", ".",
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)

    async def stream():
        try:
            while chunk := await proc.stdout.read(256 * 1024):
                yield chunk
        finally:
            if proc.returncode is None:
                proc.kill()

    fname = re.sub(r"[^A-Za-z0-9._-]+", "-", model["name"]).strip("-") or mid
    return StreamingResponse(stream(), media_type="application/gzip",
                             headers={"Content-Disposition": f'attachment; filename="{fname}.tar.gz"'})


class DeployBody(BaseModel):
    gpu: bool = False


@api.post("/lab/models/{mid}/deploy")
async def lab_deploy(mid: str, body: DeployBody):
    _model_or_404(mid)
    try:
        dep = await lab.deploy(mid, body.gpu)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    log("Cloud", "model_started", lab.models[mid]["name"])
    return dep


@api.post("/lab/models/{mid}/undeploy")
async def lab_undeploy(mid: str):
    _model_or_404(mid)
    await lab.undeploy(mid)
    return {"ok": True}


@api.get("/lab/models/{mid}/logs", response_class=PlainTextResponse)
async def lab_model_logs(mid: str):
    _model_or_404(mid)
    return await lab.deployment_logs(mid)


@api.api_route("/lab/serve/{mid}/{path:path}", methods=["GET", "POST"])
async def lab_serve(mid: str, path: str, request: Request):
    """Your running model's own API, behind your access token (OpenAI-compatible for chat models)."""
    _model_or_404(mid)
    try:
        r = await lab.proxy(mid, request.method, path, await request.body(), request.headers.get("content-type"))
    except LookupError as exc:
        raise HTTPException(409, str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(503, "The model isn't answering yet (still starting?)") from exc
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


class NewTarget(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(default=22, ge=1, le=65535)
    user: str = Field(min_length=1, max_length=64)
    workdir: str = Field(default="mnx-runs", max_length=200)
    private_key: str | None = Field(default=None, max_length=20000)


@api.post("/lab/targets")
async def lab_add_target(body: NewTarget):
    try:
        t = lab.add_target(body.name, body.host.strip(), body.port, body.user.strip(), body.workdir.strip() or "mnx-runs", body.private_key)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return await lab.test_target(t["id"])


@api.post("/lab/targets/{tid}/test")
async def lab_test_target(tid: str):
    if tid not in lab.targets:
        raise HTTPException(404, "No such server")
    return await lab.test_target(tid)


@api.delete("/lab/targets/{tid}")
def lab_remove_target(tid: str):
    if tid not in lab.targets:
        raise HTTPException(404, "No such server")
    try:
        lab.remove_target(tid)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True}


# ----- rented cloud GPUs (RunPod) -----
@api.get("/lab/providers")
async def lab_providers():
    from hive import providers
    key = lab._secrets.get("RUNPOD_API_KEY")
    out = {"runpod": {"configured": bool(key), "pods": list(lab.pods.values()), "gpus": [], "error": None}}
    if key:
        try:
            out["runpod"]["gpus"] = await providers.runpod_gpu_types(key)
        except providers.ProviderError as exc:
            out["runpod"]["error"] = str(exc)
        except httpx.HTTPError as exc:
            out["runpod"]["error"] = f"RunPod unreachable ({type(exc).__name__})"
    return out


class RentBody(BaseModel):
    gpu_type: str = Field(max_length=80)
    count: int = Field(default=1, ge=1, le=8)
    disk_gb: int = Field(default=100, ge=20, le=2000)


@api.post("/lab/providers/runpod/rent")
async def lab_rent(body: RentBody):
    try:
        pod = await lab.rent_runpod(body.gpu_type, body.count, body.disk_gb)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    log("Cloud", "rented", f"{body.count}× {body.gpu_type} on RunPod")
    asyncio.create_task(_attach(pod["id"]))
    return pod


async def _attach(pod_id: str) -> None:
    try:
        await lab.attach_pod(pod_id)
    except Exception as exc:
        rec = lab.pods.get(pod_id)
        if rec:
            rec["status"] = "failed"
            rec["error"] = str(exc)[:300]
            lab._save_pods()
            await broadcast({"type": "lab_changed"})


@api.post("/lab/providers/runpod/pods/{pod_id}/stop")
async def lab_stop_pod(pod_id: str):
    if pod_id not in lab.pods:
        raise HTTPException(404, "No such rented machine")
    try:
        await lab.stop_pod(pod_id)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    log("Cloud", "rental_stopped", pod_id)
    return {"ok": True}


class SecretBody(BaseModel):
    name: str = Field(max_length=64)
    value: str = Field(default="", max_length=10000)


@api.post("/lab/secrets")
def lab_secret(body: SecretBody):
    try:
        lab.set_secret(body.name.strip(), body.value)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"secrets": lab.secret_names()}


@api.get("/effort")
def effort_levels():
    """The five effort levels (low, med, high, ultra, maxxxx) and their token budgets."""
    return {"levels": effort.listing(), "default": effort.DEFAULT, "maxxxx_bonus": effort.MAXXXX_BONUS}


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
                task = Task(task_id, str(msg.get("chat", ""))[:40], text, emit, effort=str(msg.get("effort") or "")[:12])
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
        await task.emit("tokens", **task.usage())  # token reading for this turn
        await task.emit("task_done")
    except asyncio.CancelledError:
        EVENT_LOG.append({"task": task.id, "type": "stopped", "text": "Stopped by you", "at": now()})
    except Exception as exc:  # report, don't crash the gateway
        await task.emit("tokens", **task.usage())
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
