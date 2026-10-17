"""Mnx Hive gateway: serves the web app, streams Hive events over WebSocket,
and exposes the Connector Hub, the Browser Bee and the Phone Drone.

Run:  uvicorn gateway.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import shlex
import shutil
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel, Field

from gateway import auth
from hive import effort, phone, queen, status
from hive.browser import BrowserError, stop_playwright
from hive.bees import SIZES, BeeError, registry
from hive.cells import cells, machine
import os
from hive.connectors import LANES, LAYERS, hub
from hive.core import Task, now
from hive.lab import lab
from hive import bots, cloud, system, usage
from hive.gpu_pool import GpuError, pool as gpu
from hive.plan import plan
from hive import users
from hive.billing import BillingError, billing
from hive.calls import calls, twilio_signature_ok
from hive.voice import VoiceError, voice

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DATA = ROOT / "data"
STATE_FILE = DATA / "hive.json"

# ---------- Bees: each user's own, and Bees made by Bees (hive/bees.py) ----------
BEES = registry.bees  # the same list: the registry adds and removes in place
BEE_STATUS: dict[str, str] = {}  # bee id → busy | failed (default idle / scheduled)
EVENT_LOG: deque[dict] = deque(maxlen=500)
CLIENTS: set[WebSocket] = set()
CLIENT_USER: dict = {}  # an open page → who it belongs to (hive/users.py)
TASKS: dict[str, tuple[Task, asyncio.Task]] = {}


def bee_view(b: dict) -> dict:
    st = BEE_STATUS.get(b["id"]) or status.busy.get(b["id"]) or cells.busy(b["id"]) or ("scheduled" if b.get("schedule") else "idle")
    cell = cells.cells.get(b["id"])
    return {**{k: v for k, v in registry.view(b).items() if k != "owner"}, "status": st, "cell": cell.view() if cell else None}


# ---------- each Bee's computer: here, or on a cloud server the Lab reaches over SSH ----------
PORT = os.getenv("MNX_PORT", "8000")


def _remote_for(bee_id: str, where: str) -> dict | None:
    t = lab.targets.get(where)
    if not t:
        return None
    return {"ssh": lab._ssh_base(t), "dir": f"{t['workdir'].rstrip('/')}/cells/{bee_id}", "name": t["name"]}


def _env_for(bee_id: str, mode: str) -> dict:
    """What a Cell needs to reach its Hive: the URL (as seen from inside it) and the Bee's own key."""
    public = os.getenv("MNX_PUBLIC_URL", "").rstrip("/")
    url = public or {"docker": f"http://host.docker.internal:{PORT}", "local": f"http://127.0.0.1:{PORT}"}.get(mode, "")
    env = {"MNX_BEE_KEY": registry.key_for(bee_id)}
    if url:
        env["MNX_HIVE"] = url
    return env


def _server_name(where: str) -> str | None:
    if where == "here":
        return "this server"
    t = lab.targets.get(where)
    return t["name"] if t else f"{where} (disconnected)"


async def _bee_created(b: dict) -> None:
    try:
        cell = cells.get(b["id"], b["name"], b["computer"])
        asyncio.create_task(_boot_cell(cell))
    except RuntimeError as exc:
        log(b["name"], "cell_error", str(exc))
    who = "you" if b.get("created_by") == "you" else (registry.get(b["created_by"]) or {}).get("name", b.get("created_by"))
    log(b["name"], "bee_created", f"by {who} · {registry.view(b)['computer_text']}")
    await broadcast({"type": "bees_changed"})


async def _bee_deleted(b: dict) -> None:
    await cells.remove(b["id"])
    if b.get("bot"):
        bots.store.remove(b["id"])  # its knowledge and memories
    try:
        await cloud.machines.release_bee(b["id"])  # its own cloud server goes too: billing stops
    except cloud.CloudError as exc:
        log(b["name"], "cloud_error", f"its cloud server wasn't deleted: {exc}")
    log(b["name"], "cell_deleted", "computer, files, browser storage and jobs removed")
    await broadcast({"type": "bees_changed"})


async def _bee_computer(b: dict, old: dict) -> None:
    await cells.reconfigure(b["id"], b["name"], b["computer"], old)
    if old.get("cloud") and b["computer"].get("where") != old.get("where"):  # moved off its own cloud server
        b["computer"].pop("cloud", None)
        registry.save()
        try:
            await cloud.machines.release(old["cloud"])
        except cloud.CloudError as exc:
            log(b["name"], "cloud_error", f"its old cloud server wasn't deleted: {exc}")
    log(b["name"], "computer", registry.view(b)["computer_text"])
    await broadcast({"type": "bees_changed", "bee": b["id"]})


registry.on_created, registry.on_deleted, registry.on_computer = _bee_created, _bee_deleted, _bee_computer
registry.server_name = _server_name
cells.remote_for, cells.env_for = _remote_for, _env_for


async def broadcast(event: dict) -> None:
    """To every open page; an event that carries a "user" (a call) only to that user's pages."""
    only = event.get("user")
    for ws in list(CLIENTS):
        if only and (CLIENT_USER.get(ws) if ws in CLIENT_USER else getattr(ws, "user", None) if isinstance(ws, SseClient) else None) != only:
            continue
        try:
            await ws.send_json(event)
        except Exception:
            CLIENTS.discard(ws)


def log(bee: str, type: str, text: str) -> None:
    EVENT_LOG.append({"bee": bee, "type": type, "text": text[:300], "at": now()})


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not _ws_supported():
        print("[mnx] No WebSocket library here: the page uses its streaming fallback (chat works). For terminals and the "
              "phone drone, run: pip install websockets", flush=True)
    health = asyncio.create_task(hub.health_loop())
    loop = asyncio.get_running_loop()
    cells.on_change = lambda cell: loop.create_task(broadcast({"type": "bees_changed", "bee": cell.id}))
    await cells.start(BEES)
    lab.on_change = lambda kind: loop.create_task(broadcast({"type": "lab_changed"}))
    lab.on_finish = lambda run: loop.create_task(_report(run))
    status.on_change = lambda: loop.create_task(broadcast({"type": "bees_changed"}))
    await lab.start()
    import hive.cells as cells_mod
    cells_mod.GPU_DOCKER = lab.gpu_local
    prepull = asyncio.create_task(_prepull_images())
    cloud.machines.on_change = lambda: loop.create_task(broadcast({"type": "bees_changed"}))
    calls.broadcast, calls.run, calls.notify_phones = broadcast, run_task, _ring_phones
    calls.start_loop()
    gpu.deliver = _deliver_gpu_output
    gpu.on_change = lambda: loop.create_task(broadcast({"type": "gpu_changed"}))
    gpu.resume()
    from hive.models import models as _mnx_models
    _mnx_models.on_change = lambda: loop.create_task(broadcast({"type": "models_changed"}))
    plans_loop = asyncio.create_task(_plan_loop())
    yield
    plans_loop.cancel()
    from hive.sites import chroma as _chroma
    await _chroma.stop()
    await _mnx_models.stop_all()
    calls.stop_loop()
    health.cancel()
    prepull.cancel()
    await lab.shutdown()
    await cells.shutdown()
    await stop_playwright()


async def _plan_loop() -> None:
    """Every minute: users whose Pro ran out get Free computers again; users given Pro (owner CLI) get Pro ones."""
    from hive import plans
    while True:
        try:
            for uid in plan.expired_users():
                await plans.apply(uid)
                plan.forget_expired(uid)
            for uid in plan.pro_users():
                if any(b["computer"].get("size") != "pro" for b in registry.of(uid)):
                    await plans.apply(uid)
        except Exception as exc:  # never let the loop die
            log("Queen", "plan", f"computers didn't change: {exc}")
        await asyncio.sleep(60)


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
async def who_is_asking(request: Request, call_next):
    """The caller's id for this request (hive/users.py), from their token: their Bees, their plan."""
    uid = auth.user_of(_token_of(request))
    token = users.current.set(uid or "anonymous")
    try:
        return await call_next(request)
    finally:
        users.current.reset(token)


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


def _ws_supported() -> bool:
    import importlib.util
    return any(importlib.util.find_spec(m) for m in ("websockets", "wsproto"))


@app.get("/api/ping")
def ping():
    """Unauthenticated: lets the web app tell 'server down' from 'wrong token', and whether this server can do
    WebSockets (a plain uvicorn install can't: the page then uses the streaming fallback, /api/live)."""
    return {"ok": True, "ws": _ws_supported()}


@api.get("/health")
def health():
    from hive import brain
    from hive.models import models as _m
    return {"ok": True, "brain": brain.is_connected(), "phones": len(phone.PHONES), "cells": cells.mode, "engine": _m.engine()}


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


# ---------- Usage: what was actually used, next to the limits ----------
_disk_cache: dict[str, tuple[float, int]] = {}


async def _cached_size(key: str, path: Path, max_age: float = 120) -> int:
    hit = _disk_cache.get(key)
    if hit and time.monotonic() - hit[0] < max_age:
        return hit[1]
    size = await asyncio.to_thread(usage.dir_size, path) if path.exists() else 0
    _disk_cache[key] = (time.monotonic(), size)
    return size


async def _docker_stats() -> dict:
    if cells.mode != "docker":
        return {}
    try:
        proc = await asyncio.create_subprocess_exec("docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}",
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), 20)
        return usage.parse_docker_stats(out.decode(errors="replace"))
    except (OSError, asyncio.TimeoutError):
        return {}


@api.get("/usage")
async def usage_report(request: Request):
    from hive.bees import MAX_CHILDREN, MAX_DEPTH, SIZES
    from hive.lab import MODELS
    stats = system.stats()
    docker = await _docker_stats()
    rows = []
    mine = registry.of(users.who())
    for b in mine:
        cell = cells.cells.get(b["id"])
        view = registry.view(b)
        c = b["computer"]
        measured = docker.get(cell.container) if cell else None
        disk = await cell.disk_usage_cached() if cell and cell.mode != "remote" else None
        rows.append({
            "id": b["id"], "name": b["name"], "parent": view["parent_name"], "computer": view["computer_text"],
            "where": c.get("where", "here"), "cpus_limit": c.get("cpus"), "memory_limit_gb": c.get("memory_gb"),
            "limits": cell.view()["limits"] if cell else None, "container": cell.container_state if cell else "no Cell yet",
            "cpu_percent": measured["cpu"] if measured else None, "memory_bytes": measured["mem"] if measured else None,
            "memory_limit_bytes": measured["mem_limit"] if measured else None, "disk_bytes": disk,
            "jobs": len(cell.jobs) if cell else 0, "running": len(cell.running) if cell else 0,
            "measured": bool(measured), "status": bee_view(b)["status"]})
    runs = list(lab.runs.values())
    by_status: dict[str, int] = {}
    for r in runs:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    ts = time.time()
    pods = [{**usage.rental_now(p, ts), "id": p["id"], "status": p.get("status")} for p in lab.pods.values()]
    past = usage.ledger.rentals
    efforts = {k: {"label": v["label"], "tokens": v["tokens"], "reply": v["reply"]} for k, v in effort.LEVELS.items()}
    data_dir = await _cached_size("data", DATA)
    return {
        "at": int(ts),
        "tokens": usage.token_summary(usage.ledger),
        "server": {"cpus": stats["cpus"], "load": stats["load"], "cpu_percent": stats["cpu_percent"], "memory": stats["memory"], "disk": stats["disk"],
                   "gpus": stats["gpus"], "uptime": stats["uptime"], "data_bytes": data_dir, "cell_mode": cells.mode},
        "bees": {"count": len(mine), "limit": plan.max_bees(), "hive_count": len(BEES) if users.is_owner() else None,
                 "yours": sum(1 for b in mine if b.get("created_by") == "you"),
                 "by_bees": sum(1 for b in mine if b.get("created_by") not in (None, "you")),
                 "deepest": max((registry.depth(b["id"]) for b in mine), default=0), "max_depth": MAX_DEPTH, "max_children": MAX_CHILDREN,
                 "rows": rows, "docker_measured": bool(docker)},
        "lab": {"runs": len(runs), "by_status": by_status, "training_seconds": round(sum(usage.run_seconds(r, ts) for r in runs)),
                "running": sum(1 for r in runs if r["status"] in ("preparing", "running")),
                "models": len(lab.models), "models_bytes": await _cached_size("models", MODELS),
                "deployments_running": sum(1 for d in lab.deployments.values() if d.get("status") in ("starting", "running")),
                "servers": len(lab.targets)},
        "rentals": {"active": pods, "active_cost": round(sum(p["cost"] or 0 for p in pods), 2),
                    "past_count": len(past), "past_hours": round(sum(x["hours"] for x in past), 2),
                    "past_cost": round(sum(x["cost"] or 0 for x in past), 2),
                    "unpriced": sum(1 for p in pods if p["cost"] is None) + sum(1 for x in past if x["cost"] is None),
                    "recent": past[-8:][::-1],
                    "note": "Cost is hours x the price RunPod reported for that machine; the invoice on runpod.io is the authority."},
        "limits": {"efforts": efforts, "sizes": SIZES, "upload_mb": 200, "file_mb": 200},
        "plan": plan.view(),
        "cloud": _cloud_view() if _is_owner(request) else {"providers": [], "machines": [], "month": 0, "currencies": [], "note": ""},
    }


def _cloud_view() -> dict:
    ms = cloud.machines.views()
    return {"providers": [{"id": k, "label": v.label, "secret": v.secret, "connected": k in cloud.machines.configured()}
                          for k, v in cloud.PROVIDERS.items()],
            "machines": ms, "month": round(sum(m["price_month"] for m in ms), 2),
            "currencies": sorted({m["currency"] for m in ms}),
            "note": "Cost is hours x the price the provider listed; its invoice is the authority."}


# ---------- plans (Free / Pro) and paying with Razorpay ----------
def _token_of(request: Request) -> str | None:
    h = request.headers.get("authorization", "")
    return h[7:].strip() if h.lower().startswith("bearer ") else request.query_params.get("token")


def _is_owner(request: Request) -> bool:
    return auth.role(_token_of(request)) == "owner"


@api.get("/whoami")
def whoami(request: Request):
    """"owner" or "user": the pages hide setup and credentials from users."""
    return {"role": auth.role(_token_of(request)), "plan": plan.of()}


@api.get("/cloud/status")
def cloud_public_status():
    """What users may see of the cloud: their plan and whether the GPU is ready. No keys, no addresses."""
    g = gpu.view()
    return {"plan": plan.view(), "gpu_ready": bool(g["source"]), "hours_week": g["hours_week"], "weekly_limit": g["weekly_limit"]}


@api.get("/plan")
def get_plan():
    return {**plan.view(), "billing": billing.view()}


@api.get("/billing")
def billing_status():
    return {**billing.view(), "plan": plan.view()}


@api.post("/billing/order")
async def billing_order():
    """Step 1 of paying for Pro: a Razorpay order for this user, at today's price."""
    if users.is_owner():
        raise HTTPException(400, "You own this Hive: you're always on Pro")
    try:
        return await billing.create_order(users.who())
    except BillingError as exc:
        raise HTTPException(400, str(exc)) from exc


class PaidBody(BaseModel):
    razorpay_order_id: str = Field(max_length=64)
    razorpay_payment_id: str = Field(max_length=64)
    razorpay_signature: str = Field(max_length=256)


async def _pro_started(uid: str) -> list[str]:
    from hive import plans
    try:
        changed = await plans.apply(uid)
    except Exception as exc:
        log("Queen", "plan", f"some computers didn't change: {exc}")
        changed = []
    await broadcast({"type": "plan_changed"})
    await broadcast({"type": "bees_changed"})
    return changed


@api.post("/billing/confirm")
async def billing_confirm(body: PaidBody):
    """Step 3: Checkout says it's paid. The signature proves it; then this user is on Pro."""
    uid = users.who()
    try:
        r = billing.confirm(uid, body.razorpay_order_id, body.razorpay_payment_id, body.razorpay_signature)
    except BillingError as exc:
        raise HTTPException(400, str(exc)) from exc
    changed = await _pro_started(uid)
    return {**plan.view(uid), "changed": changed, "new": r["new"]}


@app.post("/razorpay/webhook")
async def razorpay_webhook(request: Request):
    """Razorpay tells the Hive a payment went through (in case the page closed before step 3)."""
    body = await request.body()
    try:
        r = billing.webhook(body, request.headers.get("x-razorpay-signature", ""))
    except BillingError as exc:
        raise HTTPException(400, str(exc)) from exc
    if r.get("user"):
        await _pro_started(r["user"])
    return {"ok": True}


@api.get("/cloud", dependencies=[Depends(auth.require_owner)])
def cloud_status():
    return _cloud_view()


@api.get("/cloud/quote", dependencies=[Depends(auth.require_owner)])
async def cloud_quote(provider: str | None = None):
    try:
        q = await cloud.machines.quote(provider or None)
    except cloud.CloudError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {**q, "text": cloud.describe_quote(q), "inr_month": cloud.to_rupees(q["price_month"], q["currency"])}


# ---------- Bees ----------
@api.get("/bees")
def list_bees():
    """Your Bees (each user sees only their own)."""
    return [bee_view(b) for b in registry.of(users.who())]


class NewBee(BaseModel):
    name: str = Field(min_length=1, max_length=24)
    skill: str = Field(min_length=1, max_length=90)
    created_by: str = Field(default="you", max_length=40)


@api.post("/bees")
async def add_bee(bee: NewBee):
    if bee.created_by != "you" and not registry.mine(bee.created_by, users.who()):
        raise HTTPException(404, "No such Bee")
    try:
        record = await registry.create(bee.name, bee.skill, bee.created_by, owner=users.who())
    except BeeError as exc:
        raise HTTPException(400, str(exc)) from exc
    return bee_view(record)


@api.get("/computers")
def computer_options():
    """What a Bee's computer is: nobody picks it, the plan decides (Free or Pro)."""
    return {"sizes": SIZES, "plan": plan.view(), "cell_mode": cells.mode,
            "limits": "enforced" if cells.mode == "docker" else "not enforced on this server (no Docker)"}


async def _boot_cell(cell) -> None:
    try:
        await cell.ensure_up()
    except Exception as exc:
        log(cell.name, "cell_error", str(exc))
    await broadcast({"type": "bees_changed", "bee": cell.id})


@api.delete("/bees/{bee_id}")
async def delete_bee(bee_id: str):
    _bee(bee_id)
    try:
        adopted = await registry.delete(bee_id)
    except BeeError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "adopted": [b["id"] for b in adopted]}


@api.get("/events", dependencies=[Depends(auth.require_owner)])
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


FUNCTION_CELLS = {"browser": "Browser"}  # the Hive's own browser: a function, not a Bee, and the owner's


def _bee(bee_id: str) -> dict:
    """One of the caller's Bees (the owner can reach every Bee)."""
    bee = registry.mine(bee_id, users.who())
    if not bee:
        raise HTTPException(404, "No such Bee")
    return bee


def _cell(bee_id: str):
    if bee_id in FUNCTION_CELLS and not registry.get(bee_id):
        if not users.is_owner():
            raise HTTPException(403, "Only the Hive's owner can use the Hive's own browser; each of your Bees has its own")
        return cells.get(bee_id, FUNCTION_CELLS[bee_id])
    bee = _bee(bee_id)
    try:
        return cells.get(bee["id"], bee["name"], bee.get("computer"))
    except RuntimeError as exc:
        raise HTTPException(409, f"{exc}. Move this Bee's computer to another server.") from exc


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


# ---------- each Bee's desktop: Vortex OS on its computer ----------
@api.get("/cells/{bee_id}/desktop")
def desktop_status(bee_id: str):
    from hive.vortex import desktops
    cell = _cell(bee_id)
    return {**desktops.status(cell), "os_error": cell.os_error}


@api.post("/cells/{bee_id}/desktop/start")
async def desktop_start(bee_id: str):
    from hive.vortex import desktops
    cell = _cell(bee_id)
    try:
        st = await desktops.start(cell)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    log(cell.name, "desktop", f"Vortex OS on port {st['port']}")
    await broadcast({"type": "bees_changed", "bee": cell.id})
    return {**st, "os_error": cell.os_error}


@api.post("/cells/{bee_id}/desktop/stop")
async def desktop_stop(bee_id: str):
    from hive.vortex import desktops
    cell = _cell(bee_id)
    if cell.mode == "docker":
        raise HTTPException(400, "In Docker the desktop is the Bee's computer itself; it runs while the computer runs")
    await desktops.stop(cell)
    return desktops.status(cell)


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
async def files_list(bee_id: str, path: str = ""):
    cell = _cell(bee_id)
    if cell.mode == "remote":
        try:
            await cell.ensure_up()
            return await cell.rlist(path)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(404, "Not a folder") from exc
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from exc
    p = _path(cell, path)
    if not p.is_dir():
        raise HTTPException(404, "Not a folder")
    return cell.list_files(path)


@api.get("/cells/{bee_id}/file")
async def file_get(bee_id: str, path: str):
    cell = _cell(bee_id)
    if cell.mode == "remote":
        try:
            data = await cell.rread(path)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(404, "No such file") from exc
        name = path.rstrip("/").rsplit("/", 1)[-1] or "file"
        return Response(data, media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{name}"'})
    p = _path(cell, path)
    if not p.is_file():
        raise HTTPException(404, "No such file")
    return FileResponse(p, filename=p.name)


@api.put("/cells/{bee_id}/file")
async def file_put(bee_id: str, path: str, request: Request):
    cell = _cell(bee_id)
    if cell.mode == "remote":
        data = bytearray()
        async for chunk in request.stream():
            data += chunk
            if len(data) > 200 * 1024 * 1024:
                raise HTTPException(413, "Files up to 200 MB")
        try:
            await cell.ensure_up()
            await cell.rwrite(path, bytes(data))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from exc
        return {"ok": True, "size": len(data)}
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
    if cell.mode == "remote":
        from hive.cells import rpath
        try:
            target = rpath(cell.remote["dir"], path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if target == rpath(cell.remote["dir"]):
            raise HTTPException(400, "Can't delete the Cell's root folder")
        code, out = await cell.exec(f"rm -rf -- {target}", 60)
        if code != 0:
            raise HTTPException(400, out.strip()[:300] or "Couldn't delete")
        return {"ok": True}
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
@api.get("/phones", dependencies=[Depends(auth.require_owner)])
def list_phones():
    return [p.view() for p in phone.PHONES.values()]


class PhoneCommand(BaseModel):
    cmd: str = Field(max_length=40)
    args: dict[str, Any] = Field(default_factory=dict)
    confirmed: bool = False


@api.post("/phones/{phone_id}/cmd", dependencies=[Depends(auth.require_owner)])
async def phone_cmd(phone_id: str, body: PhoneCommand):
    try:
        p = phone.get(phone_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    if body.cmd in phone.RISKY and not body.confirmed:
        raise HTTPException(428, f"'{body.cmd}' needs your confirmation")
    try:
        result = await p.call(body.cmd, body.args, timeout=phone.SLOW.get(body.cmd, 25))
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, "The phone didn't answer in time") from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if body.cmd == "apps" and isinstance(result, list):
        p.apps = [{**a, "phone": p.id} for a in result]
    if body.cmd not in ("screenshot", "ui_tree", "apps", "battery", "screen_size", "stream_status"):
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
async def lab_overview(request: Request):
    runs = sorted(lab.runs.values(), key=lambda r: r["created"], reverse=True)
    return {
        "recipes": list(lab.recipes.values()), "targets": lab.target_views(),
        "runs": [lab.run_view(r) for r in runs], "models": sorted(lab.models.values(), key=lambda m: m["created"], reverse=True),
        "deployments": lab.deployments, "secrets": lab.secret_names() if _is_owner(request) else [],
        "hive_key": await lab.hive_key() if _is_owner(request) else "",
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


@api.post("/lab/targets", dependencies=[Depends(auth.require_owner)])
async def lab_add_target(body: NewTarget):
    try:
        t = lab.add_target(body.name, body.host.strip(), body.port, body.user.strip(), body.workdir.strip() or "mnx-runs", body.private_key)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return await lab.test_target(t["id"])


@api.post("/lab/targets/{tid}/test", dependencies=[Depends(auth.require_owner)])
async def lab_test_target(tid: str):
    if tid not in lab.targets:
        raise HTTPException(404, "No such server")
    return await lab.test_target(tid)


@api.delete("/lab/targets/{tid}", dependencies=[Depends(auth.require_owner)])
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


@api.post("/lab/providers/runpod/rent", dependencies=[Depends(auth.require_owner)])
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


@api.post("/lab/providers/runpod/pods/{pod_id}/stop", dependencies=[Depends(auth.require_owner)])
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


@api.post("/lab/secrets", dependencies=[Depends(auth.require_owner)])
def lab_secret(body: SecretBody):
    try:
        lab.set_secret(body.name.strip(), body.value)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"secrets": lab.secret_names()}


# ---------- layer-by-layer streaming: a model served one layer at a time ----------
def _model_dirs(mid: str) -> tuple[dict, Path, Path]:
    from hive.lab import MODELS
    m = lab.models.get(mid)
    if not m:
        raise HTTPException(404, "No such model")
    model_dir = MODELS / mid / "files" / "model"
    if not model_dir.is_dir() or not list(model_dir.glob("*.safetensors")):
        raise HTTPException(400, "This model has no saved weights yet (a blueprint must be created and trained first)")
    return m, model_dir, MODELS / mid / "layers"


@api.get("/lab/models/{mid}/layers")
async def layers_manifest(mid: str):
    """Split the model into one file per layer (first call) and describe them."""
    from hive import layers
    m, model_dir, layers_dir = _model_dirs(mid)
    try:
        manifest = await asyncio.to_thread(layers.split, model_dir, layers_dir)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {**manifest, "model": mid, "name": m["name"]}


@api.get("/lab/models/{mid}/layers/plan")
def layers_plan(mid: str, ram_gb: float = 8, storage_free_gb: float | None = None, mb_s: float = 40, cores: int = 8):
    """What running / training this model layer by layer costs on a device with that much memory."""
    from hive import architect, layers
    m = lab.models.get(mid)
    if not m:
        raise HTTPException(404, "No such model")
    arch = m.get("arch") or {}
    res = m.get("result") or {}
    params = res.get("params") or arch.get("params") or 0
    n_layers = arch.get("layers") or 0
    hidden = arch.get("hidden") or 0
    if not (n_layers and hidden):
        from hive.lab import MODELS
        cfg = MODELS / mid / "files" / "model" / "config.json"
        if cfg.exists():
            c = json.loads(cfg.read_text())
            n_layers, hidden = c.get("num_hidden_layers", n_layers), c.get("hidden_size", hidden)
    if not params:
        raise HTTPException(400, "Unknown model size")
    p = layers.plan(params, n_layers or 1, hidden or 1, ram_gb=ram_gb, storage_free_gb=storage_free_gb, link_mb_s=mb_s, cores=cores)
    return {**p, "model": mid, "name": m["name"], "params": params, "params_text": architect.fmt_params(params), "text": layers.plan_text(p, m["name"])}


@api.get("/lab/models/{mid}/lpu/plan")
def lpu_plan(mid: str, chips: int = 1, seq: int = 64, train: bool = False):
    """What running (or one training step of) this model on `chips` virtual LPUs costs: exact compiled cycles, SRAM, resident or streamed weights."""
    from hive import architect, lpu
    m, model_dir, _ = _model_dirs(mid)
    if not (model_dir / "config.json").exists():
        raise HTTPException(400, "This model has no config.json to compile from")
    cfg = json.loads((model_dir / "config.json").read_text())
    params = (m.get("result") or {}).get("params") or (m.get("arch") or {}).get("params") or 0
    p = lpu.plan(cfg, params, seq=max(8, min(int(seq), 2048)), chips=max(1, min(int(chips), 65536)), train=train,
                 spec=lpu.ChipSpec(word_bytes=4) if train else None, lr=3e-2)
    return {**p, "model": mid, "name": m["name"], "params": params, "params_text": architect.fmt_params(params), "text": lpu.plan_text(p, m["name"])}


@api.get("/lab/models/{mid}/layers/{name}")
def layers_file(mid: str, name: str):
    from hive import layers
    _, _, layers_dir = _model_dirs(mid)
    if not (layers_dir / "manifest.json").exists():
        raise HTTPException(409, "Ask for the manifest first")
    if name in layers.SMALL_FILES and (layers_dir / name).exists():
        return FileResponse(layers_dir / name)
    if re.fullmatch(r"(layer_\d{4}|shared)", name) and (layers_dir / f"{name}.safetensors").exists():
        return FileResponse(layers_dir / f"{name}.safetensors", media_type="application/octet-stream")
    raise HTTPException(404, "No such layer")


@api.put("/lab/models/{mid}/layers/{name}")
async def layers_put(mid: str, name: str, request: Request):
    """A trained layer comes back from the phone and replaces the Hive's copy."""
    from hive import layers
    _, _, layers_dir = _model_dirs(mid)
    if not re.fullmatch(r"(layer_\d{4}|shared)", name) or not (layers_dir / f"{name}.safetensors").exists():
        raise HTTPException(404, "No such layer")
    tmp = layers_dir / f"{name}.upload"
    size = 0
    try:
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > 64 * 1024**3:
                    raise HTTPException(413, "Layer too big")
                f.write(chunk)
        await asyncio.to_thread(layers.check_layer_file, tmp, layers.expected_tensors(layers_dir, name))
        tmp.replace(layers_dir / f"{name}.safetensors")
    except ValueError as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {"ok": True, "bytes": size}


class MergeBody(BaseModel):
    steps: int = 0
    loss: float | None = None


@api.post("/lab/models/{mid}/layers/merge")
async def layers_merge(mid: str, body: MergeBody):
    """Write the (retrained) layers back into model.safetensors so Run it uses them."""
    m, _, layers_dir = _model_dirs(mid)
    if not (layers_dir / "manifest.json").exists():
        raise HTTPException(409, "Nothing to merge")
    res = await asyncio.to_thread(lab.merge_layers, mid, body.steps, body.loss)
    log("Phone", "merged", f"{m['name']}: {body.steps} layer-streamed steps")
    return res


@api.get("/lab/stream/sample", response_class=PlainTextResponse)
def stream_sample():
    """Demo text a phone can train on when it has no file of its own."""
    from hive.lab import RECIPES
    return (RECIPES / "llm-pretrain" / "sample.txt").read_text()


@api.get("/effort")
def effort_levels():
    """The five effort levels (low, med, high, ultra, maxxxx) and their token budgets."""
    return {"levels": effort.listing(), "default": effort.DEFAULT, "maxxxx_bonus": effort.MAXXXX_BONUS}




# ---------- WebSockets ----------
@app.websocket("/ws/term/{bee_id}")
async def term_ws(ws: WebSocket, bee_id: str):
    """A Bee's terminal. The shell keeps running after you disconnect."""
    await ws.accept()
    uid = auth.user_of(ws.query_params.get("token"))
    if not uid:
        await ws.close(code=4401)
        return
    bee = registry.mine(bee_id, uid)
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
    if auth.role(ws.query_params.get("token")) != "owner":  # a phone pairs with the owner's Hive only
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


async def handle_client_message(msg: dict, emit, user: str) -> None:
    """One message from the web app (over a WebSocket or the streaming fallback): a chat message, a tapped card, a stop.
    `user` is who sent it: a card or a stop only reaches that user's own messages."""
    kind = msg.get("type")
    if kind == "user_message":
        text = str(msg.get("text", ""))[:8000].strip()
        if not text:
            return
        task_id = str(msg.get("task") or f"t_{uuid.uuid4().hex[:10]}")[:40]
        if task_id in TASKS:
            return
        task = Task(task_id, str(msg.get("chat", ""))[:40], text, emit, effort=str(msg.get("effort") or "")[:12],
                    bee=str(msg.get("bee") or "")[:40] or None, mode=str(msg.get("mode") or "")[:8], user=user)
        hist = msg.get("history") if isinstance(msg.get("history"), list) else []
        task.history = [{"role": "bot" if h.get("role") == "bot" else "user", "text": str(h.get("text", ""))[:4000]}
                        for h in hist[-24:] if isinstance(h, dict)]
        task.user_name = str(msg.get("name") or "")[:40]
        from hive.models import models as _models_
        picked = str(msg.get("model") or "")[:60]
        task.model = picked if picked and _models_.usable(picked) else None
        TASKS[task_id] = (task, asyncio.create_task(run_task(task)))
    elif kind == "action":
        entry = TASKS.get(str(msg.get("task")))
        if entry and entry[0].user == user:
            values = msg.get("values") if isinstance(msg.get("values"), dict) else {}
            entry[0].resolve(str(msg.get("ref")), str(msg.get("action")), values)
    elif kind == "stop":
        entry = TASKS.get(str(msg.get("task")))
        if entry and entry[0].user == user:
            TASKS.pop(str(msg.get("task")), None)
            entry[1].cancel()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    uid = auth.user_of(ws.query_params.get("token"))
    if not uid:
        await ws.close(code=4401)
        return
    CLIENTS.add(ws)
    CLIENT_USER[ws] = uid

    async def emit(event: dict) -> None:
        EVENT_LOG.append(event)
        try:
            await ws.send_json(event)
        except Exception:
            pass

    try:
        while True:
            await handle_client_message(await ws.receive_json(), emit, uid)
    except (WebSocketDisconnect, json.JSONDecodeError, RuntimeError):
        pass
    finally:
        CLIENTS.discard(ws)
        CLIENT_USER.pop(ws, None)


# ---------- the streaming fallback: the same conversation without WebSockets ----------
class SseClient:
    """A browser reading /api/live. It looks like a WebSocket to the broadcast code: send_json puts the event in its queue."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=2000)

    async def send_json(self, event: dict) -> None:
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:  # a reader that stopped reading: drop the oldest so the Hive never blocks on it
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass


LIVE: dict[str, SseClient] = {}


@api.get("/live")
async def live_stream(request: Request, sid: str = Query(min_length=6, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")):
    """Server-sent events: every Hive event for this browser, kept open (a comment every 15 s keeps proxies awake)."""
    client = SseClient()
    client.user = users.who()
    LIVE[sid] = client
    CLIENTS.add(client)

    async def gen():
        try:
            yield "retry: 2000\n\n: connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(client.queue.get(), 15)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            CLIENTS.discard(client)
            if LIVE.get(sid) is client:
                LIVE.pop(sid, None)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


@api.post("/live/send")
async def live_send(request: Request, sid: str = Query(min_length=6, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")):
    client = LIVE.get(sid)
    if not client or getattr(client, "user", None) != users.who():
        raise HTTPException(409, "Not connected: reload the page")
    try:
        msg = await request.json()
    except ValueError as exc:
        raise HTTPException(400, "Bad message") from exc

    async def emit(event: dict) -> None:
        EVENT_LOG.append(event)
        await client.send_json(event)
    if isinstance(msg, dict):
        await handle_client_message(msg, emit, users.who())
    return {"ok": True}


async def run_task(task: Task) -> None:
    from hive.models import selected as _selected_model
    _selected_model.set(getattr(task, "model", None))  # this message is answered by the model the user picked
    users.current.set(task.user)  # their Bees, their plan
    try:
        await _run_task(task)
    finally:
        usage.ledger.record_task(task.usage())  # every message counts, finished, failed or stopped


async def _run_task(task: Task) -> None:
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


# ---------- the brain, bots you make, the Hive's free GPU and free cloud ----------
@api.get("/brain")
def brain_settings(request: Request):
    from hive.brain import brain
    v = brain.view()
    if _is_owner(request):
        return v
    return {"connected": v["connected"], "model": v["model"]}  # users learn whether bots can talk, never where or how


@api.put("/brain", dependencies=[Depends(auth.require_owner)])
def brain_update(body: dict):
    from hive.brain import brain
    try:
        return brain.update(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@api.post("/brain/test", dependencies=[Depends(auth.require_owner)])
async def brain_test():
    from hive.brain import BrainNotConnected, brain
    try:
        reply = await brain.chat([{"role": "user", "content": "Reply with just: OK"}], max_tokens=20)
    except (BrainNotConnected, httpx.HTTPError) as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"ok": True, "reply": reply[:200]}


def _bot(bot_id: str) -> dict:
    b = registry.mine(bot_id, users.who())
    if not bots.is_bot(b):
        raise HTTPException(404, "No such Bee")
    return b


def _bot_view(b: dict) -> dict:
    return {**bee_view(b), "knowledge": bots.store.files(b["id"]), "memories": len(bots.store.memories(b["id"]))}


@api.get("/bots")
def list_bots():
    """Your Bees, with their profiles (the Bees page)."""
    return [_bot_view(b) for b in registry.of(users.who())]


@api.get("/bots/shared")
def shared_bots():
    """Bees their owners shared with everyone on this Hive: you can remix them into your own (not use theirs)."""
    me = users.who()
    return [{"id": b["id"], "name": b["name"], "skill": b["skill"], "tone": b["bot"].get("tone"),
             "starters": b["bot"].get("starters") or [], "files": len(bots.store.files(b["id"]))}
            for b in BEES if bots.is_bot(b) and b["bot"].get("visibility") == "hive" and b.get("owner", "owner") != me]


@api.get("/bots/templates")
def bot_templates():
    return {"templates": [], "tones": bots.TONES, "knowledge_types": sorted(bots.KNOWLEDGE_TYPES)}


class BotBody(BaseModel):
    name: str | None = Field(default=None, max_length=24)
    tagline: str = Field(default="", max_length=90)
    profile: dict = Field(default_factory=dict)
    knowledge: dict[str, str] = Field(default_factory=dict)
    remix_of: str | None = Field(default=None, max_length=40)


@api.post("/bots")
async def new_bot(body: BotBody):
    if not body.name:
        raise HTTPException(400, "A bot needs a name")
    src = registry.get(body.remix_of) if body.remix_of else None
    if src and not (registry.mine(src["id"], users.who()) or (src.get("bot") or {}).get("visibility") == "hive"):
        src = None  # someone else's private Bee: nothing to copy
    try:
        b = await bots.create_bot(body.name, body.tagline, body.profile, knowledge=body.knowledge, owner=users.who())
        if src and bots.is_bot(src):
            bots.store.copy(src["id"], b["id"])
    except (BeeError, bots.BotError) as exc:
        raise HTTPException(400, str(exc)) from exc
    await broadcast({"type": "bees_changed"})
    return _bot_view(b)


@api.put("/bots/{bot_id}")
async def edit_bot(bot_id: str, body: BotBody):
    try:
        b = bots.update_bot(_bot(bot_id), {"profile": body.profile, "tagline": body.tagline, "name": body.name})
    except bots.BotError as exc:
        raise HTTPException(400, str(exc)) from exc
    await broadcast({"type": "bees_changed"})
    return _bot_view(b)


@api.post("/bots/{bot_id}/knowledge")
async def bot_knowledge(bot_id: str, request: Request, name: str):
    _bot(bot_id)
    data = await request.body()
    try:
        return bots.store.add_file(bot_id, name, data)
    except bots.BotError as exc:
        raise HTTPException(400, str(exc)) from exc


@api.delete("/bots/{bot_id}/knowledge/{name}")
def bot_knowledge_remove(bot_id: str, name: str):
    _bot(bot_id)
    bots.store.remove_file(bot_id, name)
    return {"ok": True}


@api.get("/bots/{bot_id}/memory")
def bot_memory(bot_id: str):
    _bot(bot_id)
    return bots.store.memories(bot_id)


@api.delete("/bots/{bot_id}/memory")
def bot_forget(bot_id: str):
    _bot(bot_id)
    return {"forgot": bots.store.forget(bot_id)}


@api.get("/bots/{bot_id}/export")
def bot_export(bot_id: str):
    return bots.export(_bot(bot_id))


@api.post("/bots/import")
async def bot_import(body: dict):
    if body.get("mnx_bot") != 1 or not body.get("name"):
        raise HTTPException(400, "That isn't an exported Mnx bot")
    name = str(body["name"])[:24]
    while registry.find(name, users.who()):
        name = (name[:20] + " 2") if not name.endswith(" 2") else name[:-2] + " 3"
    try:
        b = await bots.create_bot(name, str(body.get("tagline") or ""), body.get("profile") or {},
                                  knowledge={k: str(v) for k, v in (body.get("knowledge") or {}).items()}, owner=users.who())
    except (BeeError, bots.BotError) as exc:
        raise HTTPException(400, str(exc)) from exc
    await broadcast({"type": "bees_changed"})
    return _bot_view(b)


async def _deliver_gpu_output(bee_id: str, rel: str, data: bytes) -> None:
    """A GPU job's output files go back into the Bee's computer (gpu-out/<job>/…)."""
    b = registry.get(bee_id)
    if not b:
        return
    cell = cells.get(b["id"], b["name"], b["computer"])
    if cell.mode == "remote":
        await cell.rwrite(rel, data)
        return
    p = cell.safe_path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


@api.get("/apps/shot/{sid}")
def app_screenshot(sid: str):
    """A screenshot a Bee took while using a website or app for you (only yours; kept in memory, newest 300)."""
    from hive.apps import shot_for
    hit = shot_for(sid, users.who())
    if not hit:
        raise HTTPException(404, "That screenshot is gone")
    return Response(hit[0], media_type=hit[1], headers={"Cache-Control": "private, max-age=86400"})


@api.get("/gpu")
def gpu_status():
    v = gpu.view()
    if not users.is_owner():
        own = {b["id"] for b in registry.of(users.who())}
        v = {**v, "jobs": [j for j in v.get("jobs", []) if j.get("bee") in own]}
    return v


class GpuJob(BaseModel):
    bee: str = Field(max_length=40)
    command: str = Field(min_length=1, max_length=500)
    files: list[str] = Field(default_factory=list, max_length=50)


@api.post("/gpu/jobs")
async def gpu_job(body: GpuJob):
    from hive.gpu_chat import read_bee_file
    b = _bee(body.bee)
    try:
        files = {rel: await read_bee_file(b, rel) for rel in body.files}
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(400, f"Not in the Bee's computer: {exc}") from exc
    try:
        return await gpu.submit(b["id"], body.command, files)
    except GpuError as exc:
        raise HTTPException(400, str(exc)) from exc


@api.get("/gpu/jobs/{job_id}/files/{name:path}")
def gpu_file(job_id: str, name: str):
    job = gpu.jobs.get(job_id)
    if not job or not registry.mine(job["bee"], users.who()):
        raise HTTPException(404, "No such output")
    try:
        return FileResponse(gpu.output_path(job_id, name))
    except FileNotFoundError as exc:
        raise HTTPException(404, "No such output") from exc


@api.get("/cloud/free", dependencies=[Depends(auth.require_owner)])
async def free_cloud():
    """The Hive's free cloud: what's set up (free GPU, free servers) and the Hive's SSH key to paste."""
    return {"gpu": {k: v for k, v in gpu.view().items() if k != "jobs"}, "kaggle_user": lab._secrets.get("KAGGLE_USERNAME", ""),
            "pool": cloud.pool_view(), "hive_key": await lab.hive_key(), "plan": plan.view()}


class KaggleBody(BaseModel):
    username: str = Field(min_length=1, max_length=60, pattern=r"^[A-Za-z0-9_-]+$")
    token: str = Field(default="", max_length=200)


@api.post("/cloud/kaggle", dependencies=[Depends(auth.require_owner)])
async def free_gpu_setup(body: KaggleBody):
    lab.set_secret("KAGGLE_USERNAME", body.username)
    if body.token:
        lab.set_secret("KAGGLE_API_TOKEN" if not re.fullmatch(r"[0-9a-f]{32}", body.token) else "KAGGLE_KEY", body.token.strip())
    try:
        await gpu.check()
    except GpuError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


class PoolBody(BaseModel):
    host: str = Field(min_length=3, max_length=200)
    user: str = Field(default="ubuntu", max_length=32)


POOL_JOBS: dict[str, asyncio.Task] = {}


@api.post("/cloud/pool", dependencies=[Depends(auth.require_owner)])
async def free_server_add(body: PoolBody):
    if body.host in POOL_JOBS:
        raise HTTPException(409, "That server is already being set up")

    async def go():
        try:
            t = await cloud.add_pool_server(body.host.strip(), body.user.strip() or "ubuntu")
            log("GPU", "cloud", f"Hive cloud server ready: {t['name']} · {t.get('setup')}")
            await broadcast({"type": "cloud_ready", "text": f"{t['name']} is ready: {t.get('setup')}"})
        except Exception as exc:
            await broadcast({"type": "cloud_error", "text": str(exc)[:300]})
        finally:
            POOL_JOBS.pop(body.host, None)
    POOL_JOBS[body.host] = asyncio.create_task(go())
    return {"ok": True, "text": "Setting it up: checking SSH, installing Docker and Node. This takes a few minutes."}


# ----- Iotamine (iotamine.com): your VPSes as the Hive's cloud -----
class IotamineKey(BaseModel):
    key: str = Field(min_length=8, max_length=200)


@api.post("/cloud/iotamine/key", dependencies=[Depends(auth.require_owner)])
async def iotamine_key(body: IotamineKey):
    from hive.iotamine import IotamineError, iotamine
    lab.set_secret("IOTAMINE_API_KEY", body.key.strip())
    try:
        return {"ok": True, "servers": await iotamine.servers()}
    except IotamineError as exc:
        return {"ok": False, "error": str(exc)}


@api.get("/cloud/iotamine", dependencies=[Depends(auth.require_owner)])
async def iotamine_servers():
    from hive.iotamine import IotamineError, iotamine
    if not lab._secrets.get("IOTAMINE_API_KEY"):
        return {"connected": False, "servers": []}
    try:
        used = {t.get("host") for t in cloud.pool_servers()}
        return {"connected": True, "servers": [{**v, "in_hive": v["ip"] in used} for v in await iotamine.servers()]}
    except IotamineError as exc:
        return {"connected": True, "servers": [], "error": str(exc)}


class IotamineDeploy(BaseModel):
    cores: int = Field(default=2, ge=1, le=64)
    ram_gb: int = Field(default=8, ge=1, le=512)
    disk_gb: int = Field(default=40, ge=10, le=4000)
    pop: int | None = None
    confirm: bool = False


@api.post("/cloud/iotamine/deploy", dependencies=[Depends(auth.require_owner)])
async def iotamine_deploy(body: IotamineDeploy):
    """Without confirm: the price. With confirm: deploys (real money), then adds it to the Hive's cloud."""
    from hive.iotamine import IotamineError, iotamine
    try:
        q = await iotamine.quote(body.cores, body.ram_gb, body.disk_gb, body.pop)
    except IotamineError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not body.confirm:
        return {"quote": q}

    async def go():
        try:
            v = await iotamine.deploy(f"mnx-hive-{uuid.uuid4().hex[:6]}", body.cores, body.ram_gb, body.disk_gb, q["pop"])
            await broadcast({"type": "cloud_ready", "text": f"Iotamine VPS {v['name']} is up at {v['ip']}; setting it up for the Hive"})
            t = await cloud.add_pool_server(v["ip"], "root")
            await broadcast({"type": "cloud_ready", "text": f"{t['name']} is in the Hive's cloud: {t.get('setup')}"})
        except Exception as exc:
            await broadcast({"type": "cloud_error", "text": f"Iotamine: {exc}"[:300]})
    POOL_JOBS[f"iotamine-{time.time()}"] = asyncio.create_task(go())
    return {"quote": q, "deploying": True}


@api.delete("/cloud/pool/{tid}", dependencies=[Depends(auth.require_owner)])
def free_server_remove(tid: str):
    t = lab.targets.get(tid)
    if not t or not t.get("pool"):
        raise HTTPException(404, "No such Hive cloud server")
    if any(b["computer"].get("where") == tid for b in BEES):
        raise HTTPException(409, "Bees still run on it; move them home first (“bring the bees back home”)")
    lab.remove_target(tid)
    return {"ok": True}


# ---------- your Mnx models (Google Drive → this machine → the brain) ----------
def _models():
    from hive.models import ModelError, models
    return models, ModelError


@api.get("/models")
def list_models(request: Request):
    """Users: just the names of the models they can pick. The owner: everything, for setting them up."""
    m, _ = _models()
    if _is_owner(request) and request.query_params.get("setup"):
        return m.view(owner=True)
    return m.choices()


class ModelLink(BaseModel):
    link: str = Field(min_length=10, max_length=500)


@api.put("/models/{model_id}/source", dependencies=[Depends(auth.require_owner)])
def model_source(model_id: str, body: ModelLink):
    m, ModelError = _models()
    try:
        m.set_source(model_id, body.link)
    except ModelError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True}


class ModelCommand(BaseModel):
    command: str = Field(min_length=1, max_length=1000)
    port: int = Field(default=8000, ge=1, le=65535)


@api.put("/models/{model_id}/command", dependencies=[Depends(auth.require_owner)])
def model_command(model_id: str, body: ModelCommand):
    m, ModelError = _models()
    try:
        m.set_command(model_id, body.command, body.port)
    except ModelError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True}


@api.post("/models/{model_id}/{action}", dependencies=[Depends(auth.require_owner)])
async def model_action(model_id: str, action: str):
    m, ModelError = _models()
    try:
        if action == "download":
            m.download(model_id)
        elif action == "start":
            await m.start(model_id, use=True)
        elif action == "stop":
            await m.stop(model_id)
        elif action == "delete":
            m.delete(model_id)
        else:
            raise HTTPException(404, "Unknown action")
    except ModelError as exc:
        raise HTTPException(400, str(exc)) from exc
    await broadcast({"type": "models_changed"})
    return {"ok": True, "models": m.view(owner=True)}


# ---------- voice (Fish Audio) and calls ----------
async def _ring_phones(call) -> None:
    """The owner's paired phone (drone.py) vibrates and shows who's calling; it opens the Hive when MNX_PUBLIC_URL is set."""
    if getattr(call, "user", "owner") != "owner":
        return
    for p in list(phone.PHONES.values()):
        for cmd, args in (("vibrate", {"ms": 1200}), ("toast", {"text": f"{call.bee} is calling you. Open the Hive to answer."})):
            if cmd in p.capabilities:
                try:
                    await p.call(cmd, args, timeout=10)
                except Exception:
                    pass


@api.get("/voice")
def voice_settings(request: Request):
    if _is_owner(request):
        return voice.view()
    return {"fish_ready": voice.fish_ready, "voice_name": voice.s.get("voice_name", "")}


@api.put("/voice", dependencies=[Depends(auth.require_owner)])
async def voice_update(body: dict):
    try:
        return voice.update(body)
    except (VoiceError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


class SpeakBody(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    bee: str | None = Field(default=None, max_length=40)  # a bot speaks with its own voice


@api.post("/voice/tts")
async def voice_tts(body: SpeakBody):
    """Read text aloud with the Fish Audio voice (MP3). 409 when no Fish Audio key is set: the browser speaks instead."""
    if not voice.fish_ready:
        raise HTTPException(409, "No Fish Audio key yet (Settings → Voice)")
    try:
        from hive.voice import voice_for
        audio = await voice.tts(body.text, voice_for(body.bee))
    except (VoiceError, httpx.HTTPError) as exc:
        raise HTTPException(502, str(exc)) from exc
    return Response(audio, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


@api.post("/voice/asr")
async def voice_asr(request: Request):
    """Speech to text with Fish Audio. Body: the recording (webm / ogg / wav / mp3)."""
    if not voice.fish_ready:
        raise HTTPException(409, "No Fish Audio key yet (Settings → Voice)")
    audio = await request.body()
    try:
        return {"text": await voice.asr(audio, request.query_params.get("language"))}
    except (VoiceError, httpx.HTTPError) as exc:
        raise HTTPException(502, str(exc)) from exc


@api.get("/voice/voices")
async def voice_list(q: str = ""):
    try:
        return await voice.voices(q)
    except (VoiceError, httpx.HTTPError) as exc:
        raise HTTPException(502, str(exc)) from exc


class NewCall(BaseModel):
    bee: str | None = Field(default=None, max_length=40)
    reason: str = Field(default="", max_length=500)
    request: str = Field(default="", max_length=500)
    in_minutes: float | None = Field(default=None, ge=0, le=60 * 24 * 30)
    channel: str = Field(default="auto", max_length=8)
    answer: bool = False  # Talk mode: you start it in the app, so it's picked up already


@api.get("/calls")
def list_calls():
    return calls.views(users.who())


@api.post("/calls")
async def new_call(body: NewCall):
    if body.bee and not registry.mine(body.bee, users.who()):
        raise HTTPException(404, "No such Bee")
    at = time.time() + body.in_minutes * 60 if body.in_minutes else None
    call = await calls.start(body.bee, body.reason, at, body.channel, answer=body.answer, talk=body.answer, request=body.request)
    return call.view()


def _call(call_id: str):
    try:
        c = calls.get(call_id)
    except KeyError as exc:
        raise HTTPException(404, "No such call") from exc
    if c.user != users.who():
        raise HTTPException(404, "No such call")
    return c


@api.get("/calls/{call_id}")
def get_call(call_id: str):
    return _call(call_id).view()


@api.post("/calls/{call_id}/{what}")
async def call_action(call_id: str, what: str, request: Request):
    """answer · decline · hangup · say {text} · listen (body: a recording) · card {action, values}"""
    call = _call(call_id)
    try:
        if what == "answer":
            await calls.answer(call_id)
        elif what == "decline":
            await calls.decline(call_id)
        elif what == "hangup":
            await calls.hangup(call_id)
        elif what == "say":
            body = await request.json()
            await calls.heard(call_id, str(body.get("text") or "")[:2000])
        elif what == "listen":
            text = await voice.asr(await request.body(), request.query_params.get("language"))
            if text:
                await calls.heard(call_id, text)
            return {"text": text}
        elif what == "card":
            body = await request.json()
            await calls.tap(call_id, str(body.get("action") or ""), body.get("values") if isinstance(body.get("values"), dict) else {})
        else:
            raise HTTPException(404, "Unknown call action")
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (VoiceError, httpx.HTTPError) as exc:
        raise HTTPException(502, str(exc)) from exc
    return call.view()


@api.get("/calls/{call_id}/audio/{n}")
def call_audio(call_id: str, n: int):
    _call(call_id)
    try:
        return Response(calls.audio_for(call_id, n), media_type="audio/mpeg", headers={"Cache-Control": "private, max-age=3600"})
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


# Twilio reaches these from the internet (a real phone call): no Hive token, but each call's own secret in the path,
# and Twilio's signature on every webhook.
@app.post("/voice/twilio/{call_id}/{secret}/{step}")
async def twilio_hook(call_id: str, secret: str, step: str, request: Request):
    try:
        call = calls.check(call_id, secret)
    except KeyError as exc:
        raise HTTPException(404, "No such call") from exc
    if step not in ("start", "heard", "wait", "status"):
        raise HTTPException(404, "Unknown step")
    from urllib.parse import parse_qsl
    form = dict(parse_qsl((await request.body()).decode(errors="replace"), keep_blank_values=True))
    url = f"{os.getenv('MNX_PUBLIC_URL', '').rstrip('/')}{request.url.path}"
    if not twilio_signature_ok(url, form, request.headers.get("x-twilio-signature", ""), voice.s["twilio_token"]):
        raise HTTPException(403, "Bad Twilio signature")
    return Response(await calls.twilio(call, step, form), media_type="application/xml")


@app.get("/voice/twilio/{call_id}/{secret}/audio/{n}.mp3")
def twilio_audio(call_id: str, secret: str, n: int):
    try:
        calls.check(call_id, secret)
        return Response(calls.audio_for(call_id, n), media_type="audio/mpeg")
    except KeyError as exc:
        raise HTTPException(404, "No such audio") from exc


# ---------- web app ----------
# ---------- for Bees themselves: create and list their own children with their Bee key ----------
class ChildBee(BaseModel):
    name: str = Field(min_length=1, max_length=24)
    skill: str = Field(min_length=1, max_length=90)
    tools: list[str] = Field(default_factory=list, max_length=20)


def _calling_bee(request: Request) -> dict:
    b = registry.by_key(request.headers.get("x-bee-key", ""))
    if not b:
        raise HTTPException(401, "Unknown Bee key")
    return b


@app.get("/bee-api/me")
def bee_me(request: Request):
    return bee_view(_calling_bee(request))


@app.get("/bee-api/children")
def bee_children(request: Request):
    return [bee_view(c) for c in registry.children(_calling_bee(request)["id"])]


@app.post("/bee-api/children")
async def bee_new_child(body: ChildBee, request: Request):
    parent = _calling_bee(request)
    try:
        child = await registry.create(body.name, body.skill, parent["id"], body.tools, None, True)
    except BeeError as exc:
        raise HTTPException(400, str(exc)) from exc
    return bee_view(child)


class BeeGpuJob(BaseModel):
    command: str = Field(min_length=1, max_length=500)
    files: dict[str, str] = Field(default_factory=dict)  # name → base64


@app.post("/bee-api/gpu")
async def bee_gpu(body: BeeGpuJob, request: Request):
    """A Bee sends a job to the Hive's GPU from its own computer (mnx-bee gpu …)."""
    import base64
    b = _calling_bee(request)
    try:
        files = {k: base64.b64decode(v) for k, v in body.files.items()}
    except ValueError as exc:
        raise HTTPException(400, "Files must be base64") from exc
    try:
        return await gpu.submit(b["id"], body.command, files)
    except GpuError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/bee-api/gpu/{job_id}")
def bee_gpu_job(job_id: str, request: Request):
    b = _calling_bee(request)
    job = gpu.jobs.get(job_id)
    if not job or job["bee"] != b["id"]:
        raise HTTPException(404, "No such job")
    return job


# ---------- sites the Bees build with Chroma (hive/sites.py): Chroma's own paths, passed through to it ----------
async def _chroma_pass(request: Request, path: str):
    from hive.sites import chroma
    if not await chroma.healthy():
        if not request.url.path.startswith(("/p/", "/d/")):
            raise HTTPException(404, "Not found")
        try:
            await chroma.ensure()  # the first visit after a restart wakes it up
        except Exception as exc:
            raise HTTPException(503, f"The sites server isn't running: {exc}") from exc
    url = chroma.url + request.url.path + (f"?{request.url.query}" if request.url.query else "")
    headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "accept", "last-event-id", "if-none-match", "range")}
    headers["X-Forwarded-For"] = request.client.host if request.client else ""
    client = httpx.AsyncClient(timeout=None, transport=chroma.transport)
    req = client.build_request(request.method, url, headers=headers, content=await request.body())
    try:
        r = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(502, f"The sites server didn't answer: {exc}") from exc
    keep = {k: v for k, v in r.headers.items() if k.lower() not in ("content-length", "transfer-encoding", "connection", "content-encoding")}

    async def body():
        try:
            async for chunk in r.aiter_bytes():  # decoded: Chroma may gzip, and the Hive compresses again itself
                yield chunk
        finally:
            await r.aclose()
            await client.aclose()
    return StreamingResponse(body(), status_code=r.status_code, headers=keep)


for _prefix in ("/p", "/d", "/assets", "/vendor/three", "/vendor/pyre", "/api/db", "/api/fn", "/api/a"):
    app.add_api_route(_prefix + "/{path:path}", _chroma_pass, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
                      include_in_schema=False)


@app.get("/drone.py")
def drone_script():
    """The Termux agent, so the phone can download it with curl."""
    return FileResponse(ROOT / "drone" / "drone.py", media_type="text/x-python")


@app.get("/stream.py")
def stream_script():
    """The layer-by-layer runner the phone fetches before a streaming job."""
    return FileResponse(ROOT / "hive" / "recipes" / "llm-pretrain" / "stream.py", media_type="text/x-python")


@app.get("/lpu.py")
def lpu_script():
    """The virtual LPU (simulator + compiler), so a phone can run streamed layers on it: stream.py run --engine lpu."""
    return FileResponse(ROOT / "hive" / "lpu.py", media_type="text/x-python")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.include_router(api)  # last, so every /api route above is in it
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
