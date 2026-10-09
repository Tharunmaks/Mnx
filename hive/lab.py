"""Model Lab: the Cloud Bee's engine. Train models, keep them in a registry and run them as APIs.

Where training runs ("targets"):
  local   a fresh Docker container on this server per run (GPU passed through when available)
  ssh     any machine you can SSH into: a cloud GPU VM, your own PC, a rented pod.
          Files go over with tar, the run keeps going with nohup if the connection drops,
          and the output comes back to this server when it finishes.

A recipe (hive/recipes/<id>/) is train.py + requirements.txt (+ serve.py). It reads
params.json and data/, writes output/, and prints "MNX_METRIC {json}" lines for the
live chart and one "MNX_RESULT {json}" line at the end.

Trained models are served on this server, each in its own always-on container,
reachable through the gateway at /api/lab/serve/<model>/... with your access token.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import socket
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
RECIPES = Path(__file__).resolve().parent / "recipes"
LAB = ROOT / "data" / "lab"
RUNS, MODELS, KEYS = LAB / "runs", LAB / "models", LAB / "keys"
UPLOADS = LAB / "uploads"
DOCKER_ARGS = shlex.split(os.getenv("MNX_DOCKER_ARGS", ""))
PIP_CACHE = "mnx-pip-cache"
HF_CACHE = "mnx-hf-cache"  # downloaded models and datasets, shared between runs
LLAMA_IMAGE = os.getenv("MNX_LLAMA_IMAGE", "ghcr.io/ggml-org/llama.cpp:server")
MAX_POINTS = 3000
CLUSTER_PORT = int(os.getenv("MNX_CLUSTER_PORT", "29400"))  # torchrun rendezvous port on the master node
ID_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
SSH_HOST_RE = re.compile(r"^[A-Za-z0-9.\-:\[\]]{1,255}$")
SSH_USER_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

RUN_SH = """#!/bin/sh
# Mnx Lab run script: same steps on this server's containers and on SSH servers.
cd "$(dirname "$0")"
if [ -n "$MNX_UID" ]; then trap 'chown -R "$MNX_UID:$MNX_GID" . 2>/dev/null' EXIT; fi
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=python3; command -v python3 >/dev/null 2>&1 || PY=python
if [ "$MNX_VENV" = "1" ]; then
  echo "MNX_STAGE preparing"
  if [ ! -x .venv/bin/python ]; then
    $PY -m venv .venv || { echo "Couldn't make a Python venv: install python3-venv on this server"; exit 1; }
  fi
  PY=.venv/bin/python
fi
if [ -f requirements.txt ]; then
  echo "MNX_STAGE installing"
  # No GPU on an x86 machine: the CPU build of PyTorch is ~200 MB instead of ~3 GB.
  if grep -qi '^torch' requirements.txt && ! command -v nvidia-smi >/dev/null 2>&1 && [ "$(uname -m)" = "x86_64" ]; then
    echo "No GPU here, so installing the small CPU build of PyTorch"
    $PY -m pip install --disable-pip-version-check --progress-bar off torch --index-url https://download.pytorch.org/whl/cpu \
      || echo "(The CPU build wasn't reachable; using the standard one)"
  fi
  $PY -m pip install --disable-pip-version-check --progress-bar off -r requirements.txt || exit 1
fi
mkdir -p output
echo "MNX_STAGE training"
# Run in the background so a stop signal reaches the training process too.
$PY -u train.py &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM INT HUP
wait $child; code=$?
while kill -0 $child 2>/dev/null; do wait $child; code=$?; done
exit $code
"""

# On SSH servers: start in a new session so Stop can end the whole process group.
LAUNCH_SH = """#!/bin/sh
cd "$(dirname "$0")"
echo $$ > .pid
sh run.sh
echo $? > .exitcode
"""


def _read(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write(path: Path, data, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    if private:
        os.chmod(tmp, 0o600)
    tmp.replace(path)


async def _run(argv: list[str], timeout: float = 120, stdin: bytes | None = None) -> tuple[int, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(stdin), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, b"timed out"
    return proc.returncode or 0, out


def _size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def load_recipes() -> dict[str, dict]:
    out = {}
    for d in sorted(RECIPES.iterdir()):
        meta = _read(d / "recipe.json", None)
        if not meta:
            continue
        meta["has_serve"] = (d / "serve.py").exists()
        meta["has_sample"] = bool(meta.get("sample")) and (d / meta["sample"]).exists()
        out[meta["id"]] = meta
    return dict(sorted(out.items(), key=lambda kv: (kv[1].get("order", 99), kv[0])))


def coerce_params(recipe: dict, given: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for p in recipe.get("params", []):
        v = given.get(p["name"], p.get("default"))
        t = p.get("type", "text")
        try:
            if t == "int":
                v = int(float(v))
            elif t == "float":
                v = float(v)
            elif t == "bool":
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            else:
                v = str(v if v is not None else "")[:500]
        except (TypeError, ValueError):
            raise ValueError(f"{p.get('label', p['name'])}: not a valid {t}")
        if t in ("int", "float"):
            if "min" in p and v < p["min"]:
                raise ValueError(f"{p.get('label', p['name'])} must be at least {p['min']}")
            if "max" in p and v > p["max"]:
                raise ValueError(f"{p.get('label', p['name'])} must be at most {p['max']}")
        out[p["name"]] = v
    return out


class Lab:
    def __init__(self) -> None:
        for d in (RUNS, MODELS, KEYS):
            d.mkdir(parents=True, exist_ok=True)
        self.recipes = load_recipes()
        self.runs: dict[str, dict] = {}
        for f in RUNS.glob("*/run.json"):
            r = _read(f, None)
            if r:
                self.runs[r["id"]] = r
        self.models: dict[str, dict] = {}
        for f in MODELS.glob("*/model.json"):
            m = _read(f, None)
            if m:
                self.models[m["id"]] = m
        self.deployments: dict[str, dict] = _read(LAB / "deployments.json", {})
        self.targets: dict[str, dict] = _read(LAB / "targets.json", {})
        self.pods: dict[str, dict] = _read(LAB / "pods.json", {})  # rented cloud GPUs
        self._secrets: dict[str, str] = _read(LAB / "secrets.json", {})
        self.watchers: dict[str, asyncio.Task] = {}
        self._logs: dict[str, Any] = {}
        self._dirty: set[str] = set()
        self.on_change = None
        self.on_finish = None  # called with the run when it ends (the chat posts the final report)
        self.gpu_local = False
        self._tasks: list[asyncio.Task] = []

    # ---------- lifecycle ----------
    async def start(self) -> None:
        self.gpu_local = await self._detect_local_gpu()
        await self.hive_key()
        for run in self.runs.values():
            if run["status"] in ("queued", "preparing", "running"):
                self.watchers[run["id"]] = asyncio.create_task(self._resume(run))
        self._tasks = [asyncio.create_task(self._flush_loop()), asyncio.create_task(self._health_loop())]

    async def shutdown(self) -> None:
        for t in self._tasks + list(self.watchers.values()):
            t.cancel()
        self._flush()
        for fh in self._logs.values():
            fh.close()
        # Run and model containers keep going; they're picked up again on the next start.

    def _changed(self, kind: str = "lab") -> None:
        if self.on_change:
            self.on_change(kind)

    async def _detect_local_gpu(self) -> bool:
        try:
            code, out = await _run(["docker", "info", "--format", "{{json .Runtimes}}"], 20)
        except FileNotFoundError:  # no Docker here (a phone, say): no local GPU, not an error
            return False
        return code == 0 and b"nvidia" in out

    # ---------- uploads ----------
    def take_upload(self, upload_id: str) -> Path:
        """The file behind an upload id (from PUT /api/lab/uploads/<name>)."""
        if not re.fullmatch(r"[a-f0-9]{16}", upload_id or ""):
            raise ValueError("Bad upload id")
        folder = UPLOADS / upload_id
        files = list(folder.iterdir()) if folder.is_dir() else []
        if not files:
            raise ValueError("That upload is gone; please choose the file again")
        return files[0]

    def drop_upload(self, upload_id: str) -> None:
        if re.fullmatch(r"[a-f0-9]{16}", upload_id or ""):
            shutil.rmtree(UPLOADS / upload_id, ignore_errors=True)

    # ---------- secrets (env vars for every run, e.g. HF_TOKEN) ----------
    def secret_names(self) -> list[str]:
        return sorted(self._secrets)

    def set_secret(self, name: str, value: str | None) -> None:
        if not re.match(r"^[A-Z_][A-Z0-9_]{0,63}$", name):
            raise ValueError("Use CAPITAL_LETTERS, digits and _ for the name")
        if value:
            self._secrets[name] = value
        else:
            self._secrets.pop(name, None)
        _write(LAB / "secrets.json", self._secrets, private=True)

    # ---------- targets ----------
    async def hive_key(self) -> str:
        """The Hive's own SSH key; add its public half to a server's ~/.ssh/authorized_keys."""
        key = KEYS / "hive_ed25519"
        if not key.exists():
            code, out = await _run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "mnx-hive", "-f", str(key)], 30)
            if code != 0:
                return ""
            os.chmod(key, 0o600)
        try:
            return (KEYS / "hive_ed25519.pub").read_text().strip()
        except FileNotFoundError:
            return ""

    def target_views(self) -> list[dict]:
        local = {"id": "local", "name": "This server", "kind": "local", "gpu": self.gpu_local, "status": "ok"}
        return [local] + [{k: v for k, v in t.items()} for t in self.targets.values()]

    def add_target(self, name: str, host: str, port: int, user: str, workdir: str, private_key: str | None) -> dict:
        if not SSH_HOST_RE.match(host) or host.startswith("-"):
            raise ValueError("Bad host")
        if not SSH_USER_RE.match(user) or user.startswith("-"):
            raise ValueError("Bad user name")
        if not re.match(r"^[A-Za-z0-9._/~-]{1,200}$", workdir) or ".." in workdir:
            raise ValueError("Bad folder")
        tid = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "server"
        if tid in self.targets or tid == "local":
            tid = f"{tid}-{uuid.uuid4().hex[:4]}"
        if private_key:
            key = private_key.strip() + "\n"
            if "PRIVATE KEY" not in key:
                raise ValueError("That doesn't look like a private key")
            path = KEYS / f"{tid}.key"
            path.write_text(key)
            os.chmod(path, 0o600)
        t = {"id": tid, "name": name[:60], "kind": "ssh", "host": host, "port": int(port), "user": user,
             "workdir": workdir, "own_key": bool(private_key), "status": "unchecked", "info": None,
             "gpu": False, "checked": None}
        self.targets[tid] = t
        self._save_targets()
        return t

    def remove_target(self, tid: str) -> None:
        if any(r["target"] == tid and r["status"] in ("queued", "preparing", "running") for r in self.runs.values()):
            raise ValueError("A run is still using this server")
        self.targets.pop(tid, None)
        (KEYS / f"{tid}.key").unlink(missing_ok=True)
        self._save_targets()

    def _save_targets(self) -> None:
        _write(LAB / "targets.json", self.targets)

    def _ssh_base(self, t: dict) -> list[str]:
        key = KEYS / (f"{t['id']}.key" if t.get("own_key") else "hive_ed25519")
        return ["ssh", "-i", str(key), "-p", str(t["port"]), "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={KEYS / 'known_hosts'}",
                "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3",
                "-o", "IdentitiesOnly=yes", f"{t['user']}@{t['host']}", "--"]

    async def ssh(self, t: dict, command: str, timeout: float = 60, stdin: bytes | None = None) -> tuple[int, bytes]:
        return await _run(self._ssh_base(t) + [command], timeout, stdin)

    async def test_target(self, tid: str) -> dict:
        t = self.targets[tid]
        script = ("echo MNX_OK; uname -sm; nproc; awk '/MemTotal/{print int($2/1024)}' /proc/meminfo; "
                  "python3 --version 2>&1 | head -1; "
                  "(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)")
        code, out = await self.ssh(t, script, 30)
        text = out.decode(errors="replace").strip()
        if code == 0 and "MNX_OK" in text:
            lines = text.split("MNX_OK", 1)[1].strip().splitlines()
            gpus = [l.strip() for l in lines[4:] if l.strip()]
            t["info"] = {"system": lines[0] if lines else "", "cpus": lines[1] if len(lines) > 1 else "",
                         "ram_mb": lines[2] if len(lines) > 2 else "", "python": lines[3] if len(lines) > 3 else "",
                         "gpus": gpus}
            t["gpu"], t["status"], t["error"] = bool(gpus), "ok", None
        else:
            t["status"], t["error"] = "failed", text[-400:] or f"ssh exit {code}"
        t["checked"] = int(time.time())
        self._save_targets()
        return t

    # ---------- runs ----------
    def run_dir(self, rid: str) -> Path:
        return RUNS / rid

    def create_run(self, recipe_id: str, name: str, params: dict, target: str, files: list[Path],
                   use_sample: bool, gpu: bool, cpus: float | None, memory_gb: float | None,
                   base_model_id: str | None = None, nodes: list[str] | None = None,
                   chat: dict | None = None, extra_files: dict[str, str] | None = None,
                   blueprint: str | None = None) -> dict:
        """Start a training run. `nodes` = several SSH servers trained together as one cluster."""
        recipe = self.recipes.get(recipe_id)
        if not recipe:
            raise ValueError("Unknown recipe")
        if nodes:
            if len(nodes) < 2 or any(n not in self.targets for n in nodes):
                raise ValueError("A cluster needs two or more connected SSH servers")
            target = nodes[0]
        if target != "local" and target not in self.targets:
            raise ValueError("Unknown server")
        params = coerce_params(recipe, params)
        rid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        job = self.run_dir(rid) / "job"
        (job / "data").mkdir(parents=True)
        src = RECIPES / recipe_id
        for f in ("train.py", "serve.py", "requirements.txt"):
            if (src / f).exists():
                shutil.copy(src / f, job / f)
        for helper in recipe.get("helpers", []):  # shared Python files a recipe imports (e.g. architect.py)
            hp = Path(__file__).resolve().parent / helper
            if hp.is_file():
                shutil.copy(hp, job / hp.name)
        mount_models = False
        if base_model_id:
            base = self.models.get(base_model_id)
            if not base:
                raise ValueError("Unknown base model")
            model_dir = MODELS / base_model_id / "files" / "model"
            if not model_dir.is_dir():
                raise ValueError("That model has no saved weights folder to continue from")
            if target == "local":
                mount_models = True
                params["init_from"] = f"/models/{base_model_id}/files/model"
            else:
                shutil.copytree(model_dir, job / "base")  # ships with the run to the other server
                params["init_from"] = "base"
        for f in files:
            dest = job / f.name if f.name in ("train.py", "serve.py", "requirements.txt") else job / "data" / f.name
            shutil.move(str(f), dest)
        if use_sample and recipe.get("has_sample"):
            shutil.copy(src / recipe["sample"], job / "data" / recipe["sample"])
        for need in recipe.get("needs_files", []):
            if not (job / need).exists():
                shutil.rmtree(self.run_dir(rid), ignore_errors=True)
                raise ValueError(f"This recipe needs you to upload {need}")
        if recipe["dataset"].get("required") and not any((job / "data").iterdir()):
            shutil.rmtree(self.run_dir(rid), ignore_errors=True)
            raise ValueError("Add a dataset file (or tick 'use sample data')")
        for fname, content in (extra_files or {}).items():
            fname = re.sub(r"[^A-Za-z0-9._-]", "_", Path(fname).name)[:80]
            if fname and not fname.startswith("."):
                (job / fname).write_text(content)
        (job / "params.json").write_text(json.dumps(params, indent=2))
        (job / "run.sh").write_text(RUN_SH)
        (job / "launch.sh").write_text(LAUNCH_SH)
        if self._secrets:
            env = "".join(f"{k}={shlex.quote(v)}\n" for k, v in self._secrets.items())
            (job / ".env").write_text(env)
            os.chmod(job / ".env", 0o600)
        run = {
            "id": rid, "name": (name or recipe["title"])[:80], "recipe": recipe_id, "params": params,
            "target": target, "gpu": bool(gpu), "cpus": cpus, "memory_gb": memory_gb,
            "status": "queued", "stage": None, "created": int(time.time()), "started": None, "ended": None,
            "exit": None, "metrics": [], "result": None, "model": None, "error": None,
            "data_files": sorted(p.name for p in (job / "data").iterdir()), "remote_offset": 0,
            "base_model": base_model_id, "mount_models": mount_models, "nodes": nodes or None,
            "chat": chat, "stages": {}, "blueprint": blueprint,
        }
        self.runs[rid] = run
        self._save_run(run)
        self.watchers[rid] = asyncio.create_task(self._launch(run))
        self._changed()
        return run

    def _save_run(self, run: dict) -> None:
        _write(self.run_dir(run["id"]) / "run.json", run)

    def _mark(self, run: dict) -> None:
        self._dirty.add(run["id"])

    def _flush(self) -> None:
        for rid in list(self._dirty):
            run = self.runs.get(rid)
            if run:
                self._save_run(run)
        if self._dirty:
            self._dirty.clear()
            self._changed()

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(1.5)
            self._flush()

    def _log_reset(self, run: dict) -> None:
        fh = self._logs.pop(run["id"], None)
        if fh:
            fh.close()
        (self.run_dir(run["id"]) / "log.txt").write_bytes(b"")
        run["metrics"], run["result"] = [], None

    def _line(self, run: dict, line: str) -> None:
        fh = self._logs.get(run["id"])
        if fh is None:
            fh = self._logs[run["id"]] = open(self.run_dir(run["id"]) / "log.txt", "a", encoding="utf-8")
        fh.write(line if line.endswith("\n") else line + "\n")
        fh.flush()
        s = line.strip()
        if s.startswith("MNX_METRIC "):
            try:
                point = json.loads(s[11:])
                if isinstance(point, dict):
                    run["metrics"].append(point)
                    if len(run["metrics"]) > MAX_POINTS:  # keep the curve's shape, drop every other old point
                        run["metrics"] = run["metrics"][: MAX_POINTS // 2: 2] + run["metrics"][MAX_POINTS // 2:]
            except json.JSONDecodeError:
                pass
        elif s.startswith("MNX_RESULT "):
            try:
                run["result"] = json.loads(s[11:])
            except json.JSONDecodeError:
                pass
        elif s and not s.startswith("MNX_") and not s.startswith("==="):
            run["last_line"] = s[:200]
        if s.startswith("MNX_STAGE "):
            run["stage"] = s[10:].strip()
            run.setdefault("stages", {})[run["stage"]] = time.time()  # run.sh says "training" early; train.py repeats it when the loop starts
            if run["status"] == "preparing" and run["stage"] == "training":
                run["status"] = "running"
        self._mark(run)

    def list_job_files(self, rid: str) -> list[dict]:
        """The files the Coding Bee wrote for a run (no secrets, no shipped base weights)."""
        job = self.run_dir(rid) / "job"
        out = []
        for p in sorted(job.rglob("*")):
            rel = p.relative_to(job)
            if not p.is_file() or p.name.startswith(".") or rel.parts[0] in ("base", ".venv"):
                continue
            out.append({"path": str(rel), "size": p.stat().st_size})
            if len(out) >= 200:
                break
        return out

    def read_job_file(self, rid: str, rel: str) -> str:
        job = (self.run_dir(rid) / "job").resolve()
        p = (job / rel).resolve()
        if job not in p.parents or p.name.startswith(".") or not p.is_file():
            raise ValueError("No such file")
        data = p.read_bytes()
        if b"\0" in data[:4000]:
            return f"(binary file, {p.stat().st_size:,} bytes)"
        text = data[:300_000].decode(errors="replace")
        return text + ("\n… (truncated)" if len(data) > 300_000 else "")

    def log_tail(self, rid: str, since: int = 0) -> tuple[bytes, int]:
        path = self.run_dir(rid) / "log.txt"
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return b"", 0
        since = max(0, min(since, size))
        if size - since > 512_000:  # first load of a huge log: just the end
            since = size - 512_000
        with open(path, "rb") as f:
            f.seek(since)
            return f.read(), size

    async def _launch(self, run: dict) -> None:
        run["status"], run["started"] = "preparing", int(time.time())
        self._mark(run)
        try:
            if run["target"] == "local":
                await self._start_local(run)
                await self._follow_local(run)
            else:
                await self._start_ssh(run)
                await self._follow_ssh(run)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._finish(run, -1, str(exc))

    async def _resume(self, run: dict) -> None:
        try:
            if run["target"] == "local":
                code, out = await _run(["docker", "inspect", "-f", "{{.State.Status}}", f"mnx-run-{run['id']}"], 20)
                if code != 0:
                    if run["status"] == "queued":
                        return await self._launch(run)
                    return self._finish(run, -1, "The run's container is gone (server restarted?)")
                await self._follow_local(run)
            else:
                if run["status"] == "queued":
                    return await self._launch(run)
                await self._follow_ssh(run)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._finish(run, -1, str(exc))

    # ----- local docker -----
    async def _start_local(self, run: dict) -> None:
        job = (self.run_dir(run["id"]) / "job").resolve()
        image = self.recipes.get(run["recipe"], {}).get("image", "python:3.12-slim")
        args = ["docker", "run", "-d", "--name", f"mnx-run-{run['id']}", "--label", f"mnx.run={run['id']}",
                "-v", f"{job}:/job", "-w", "/job", "-v", f"{PIP_CACHE}:/root/.cache/pip",
                "-v", f"{HF_CACHE}:/root/.cache/huggingface",
                "-e", "PYTHONUNBUFFERED=1", "-e", "PIP_ROOT_USER_ACTION=ignore", "-e", f"MNX_UID={os.getuid()}", "-e", f"MNX_GID={os.getgid()}",
                "--shm-size", "8g" if run.get("gpu") else "1g", "--security-opt", "no-new-privileges", "--init"]
        if run.get("mount_models"):
            args += ["-v", f"{MODELS.resolve()}:/models:ro"]
        if run.get("gpu") and self.gpu_local:
            args += ["--gpus", "all"]
        if run.get("cpus"):
            args += ["--cpus", str(run["cpus"])]
        if run.get("memory_gb"):
            args += ["--memory", f"{run['memory_gb']}g"]
        args += DOCKER_ARGS + [image, "sh", "/job/run.sh"]
        code, out = await _run(args, 900)
        if code != 0:
            raise RuntimeError(f"Couldn't start the training container: {out.decode(errors='replace').strip()[-300:]}")

    async def _follow_local(self, run: dict) -> None:
        name = f"mnx-run-{run['id']}"
        self._log_reset(run)  # docker keeps the whole log, so replay it from the start
        proc = await asyncio.create_subprocess_exec("docker", "logs", "-f", name,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                self._line(run, line.decode(errors="replace"))
        finally:
            if proc.returncode is None:
                proc.kill()
        code, out = await _run(["docker", "wait", name], 60)
        exit_code = int(out.strip() or -1) if code == 0 and out.strip().lstrip(b"-").isdigit() else -1
        await _run(["docker", "rm", "-f", name], 60)
        self._finish(run, exit_code)

    # ----- ssh -----
    def _remote_dir(self, run: dict, t: dict) -> str:
        wd = t.get("workdir") or "mnx-runs"
        return f"{wd.rstrip('/')}/{run['id']}"

    @staticmethod
    def _q(rdir: str) -> str:
        return shlex.quote(rdir) if not rdir.startswith("~/") else "~/" + shlex.quote(rdir[2:])

    async def _start_ssh(self, run: dict) -> None:
        """Upload and launch on one SSH server, or on every server of a cluster (node 0 is the master)."""
        nodes = run.get("nodes") or [run["target"]]
        job = self.run_dir(run["id"]) / "job"
        self._log_reset(run)
        tar = await _run(["tar", "-C", str(job), "-czf", "-", "."], 300)
        if tar[0] != 0:
            raise RuntimeError("Couldn't pack the run files")
        master = self.targets[nodes[0]]
        run["node_dirs"] = {}
        for i, tid in enumerate(nodes):
            t = self.targets[tid]
            rdir = self._remote_dir(run, t)
            q = self._q(rdir)
            self._line(run, f"Copying files to {t['name']} ({t['user']}@{t['host']}) …")
            code, out = await self.ssh(t, f"mkdir -p {q} && tar -C {q} -xzf -", 900, tar[1])
            if code != 0:
                raise RuntimeError(f"Upload to {t['name']} failed: {out.decode(errors='replace').strip()[-300:]}")
            run["node_dirs"][tid] = rdir
            env = "MNX_VENV=1"
            if len(nodes) > 1:
                env += (f" MNX_NNODES={len(nodes)} MNX_NODE_RANK={i} MNX_MASTER_ADDR={shlex.quote(master['host'])} "
                        f"MNX_MASTER_PORT={CLUSTER_PORT}")
            # The ( … &) subshell exits at once, so ssh doesn't wait on the run's output pipes.
            launch = f"cd {q} && ({env} setsid nohup sh launch.sh > log.txt 2>&1 < /dev/null &) && sleep 1 && cat .pid"
            code, out = await self.ssh(t, launch, 60)
            if code != 0:
                raise RuntimeError(f"Couldn't start the run on {t['name']}: {out.decode(errors='replace').strip()[-300:]}")
            self._line(run, f"Started on {t['name']} in {rdir}" + (f" as node {i} of {len(nodes)}" if len(nodes) > 1 else "")
                       + "; it keeps running if the connection drops")
        run["remote_dir"], run["remote_offset"] = run["node_dirs"][nodes[0]], 0
        self._mark(run)

    async def _follow_ssh(self, run: dict) -> None:
        t = self.targets.get(run["target"])
        if not t:
            return self._finish(run, -1, "The server was removed")
        rdir = run.get("remote_dir") or self._remote_dir(run, t)
        q = shlex.quote(rdir) if not rdir.startswith("~/") else "~/" + shlex.quote(rdir[2:])
        partial, failures = "", 0
        while True:
            poll = (f"cd {q} && tail -c +{int(run['remote_offset']) + 1} log.txt 2>/dev/null; "
                    "printf '\\n__MNX__ %s %s\\n' \"$(cat .exitcode 2>/dev/null || echo -)\" "
                    "\"$(kill -0 $(cat .pid 2>/dev/null) 2>/dev/null && echo alive || echo gone)\"")
            code, out = await self.ssh(t, poll, 60)
            marker = out.rfind(b"\n__MNX__ ")
            if code != 0 or marker < 0:
                failures += 1
                if failures in (1, 5) or failures % 20 == 0:
                    self._line(run, f"[Lost contact with {t['name']}, retrying … ({out.decode(errors='replace').strip()[-120:]})]")
                if failures > 360:  # ~30 minutes
                    return self._finish(run, -1, f"Lost contact with {t['name']}")
                await asyncio.sleep(5)
                continue
            failures = 0
            data, tail = out[:marker], out[marker + 9:].decode().split()
            run["remote_offset"] += len(data)
            text = partial + data.decode(errors="replace")
            lines = text.split("\n")
            partial = lines.pop()
            for line in lines:
                self._line(run, line)
            exit_s, alive = (tail + ["-", "gone"])[:2]
            if exit_s != "-" or alive == "gone":
                if partial:
                    self._line(run, partial)
                exit_code = int(exit_s) if exit_s.lstrip("-").isdigit() else -1
                if exit_code != 0 and len(run.get("nodes") or []) > 1:
                    await self._other_node_tails(run)
                if not run.get("stop_requested"):
                    await self._fetch_output(run, t, q)
                return self._finish(run, exit_code, None if exit_s != "-" else "The run stopped without an exit code")
            await asyncio.sleep(3)

    async def _other_node_tails(self, run: dict) -> None:
        """When a cluster run fails, show what the other machines said."""
        for tid in (run.get("nodes") or [])[1:]:
            t = self.targets.get(tid)
            rdir = (run.get("node_dirs") or {}).get(tid)
            if not t or not rdir:
                continue
            code, out = await self.ssh(t, f"tail -n 15 {self._q(rdir)}/log.txt 2>/dev/null", 30)
            self._line(run, f"--- last lines from {t['name']} ---")
            for line in out.decode(errors="replace").splitlines():
                self._line(run, "  " + line)

    async def _fetch_output(self, run: dict, t: dict, q: str) -> None:
        job = self.run_dir(run["id"]) / "job"
        code, out = await self.ssh(t, f"cd {q} && tar -czf - output 2>/dev/null", 1800)
        if code == 0 and out:
            (job / "output").mkdir(exist_ok=True)
            res = await _run(["tar", "-C", str(job), "-xzf", "-"], 600, out)
            self._line(run, "Fetched the output back to this server" if res[0] == 0 else "Couldn't unpack the output")

    async def stop_run(self, rid: str) -> None:
        run = self.runs[rid]
        if run["status"] not in ("queued", "preparing", "running"):
            return
        run["stop_requested"] = True
        if run["target"] == "local":
            await _run(["docker", "stop", "-t", "15", f"mnx-run-{rid}"], 60)
        else:
            dirs = run.get("node_dirs") or {run["target"]: run.get("remote_dir")}
            for tid, rdir in dirs.items():
                t = self.targets.get(tid)
                if t and rdir:
                    await self.ssh(t, f"kill -TERM -$(cat {self._q(rdir)}/.pid) 2>/dev/null; true", 30)
        self._mark(run)

    def _finish(self, run: dict, exit_code: int, error: str | None = None) -> None:
        if run["status"] in ("succeeded", "failed", "stopped"):
            return
        run["exit"], run["ended"] = exit_code, int(time.time())
        output = self.run_dir(run["id"]) / "job" / "output"
        has_output = output.exists() and any(output.iterdir())
        if run.get("stop_requested"):
            run["status"] = "stopped"
        elif exit_code == 0 and has_output:
            run["status"] = "succeeded"
        else:
            run["status"] = "failed"
            last = run.get("last_line") or ""
            run["error"] = error or ((last if last else f"Exited with code {exit_code}") if exit_code
                                     else "Finished but wrote nothing to output/")
        if error:
            self._line(run, f"[{error}]")
        self._line(run, f"=== {run['status']} (exit {exit_code})")
        run.setdefault("stages", {})["end"] = time.time()
        if run["status"] == "succeeded":
            try:
                model = self.register_model(run)
                bp = self.models.get(run.get("blueprint") or "")
                if bp and bp.get("kind") == "blueprint":
                    bp["trained_model"] = model["id"]
                    _write(MODELS / bp["id"] / "model.json", bp)
            except Exception as exc:
                self._line(run, f"[Couldn't add the model to the registry: {exc}]")
        fh = self._logs.pop(run["id"], None)
        if fh:
            fh.close()
        self.watchers.pop(run["id"], None)
        self._save_run(run)
        self._changed()
        if self.on_finish:
            try:
                self.on_finish(run)
            except Exception:
                pass

    async def delete_run(self, rid: str) -> None:
        run = self.runs[rid]
        if run["status"] in ("queued", "preparing", "running"):
            raise ValueError("Stop the run first")
        self.runs.pop(rid)
        await _rmtree(self.run_dir(rid))
        self._changed()

    # ---------- models ----------
    def register_model(self, run: dict) -> dict:
        recipe = self.recipes.get(run["recipe"], {})
        job = self.run_dir(run["id"]) / "job"
        mid = "m-" + run["id"]
        mdir = MODELS / mid
        mdir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(job / "output"), mdir / "files")
        for f in ("serve.py", "requirements.txt"):
            if (job / f).exists():
                shutil.copy(job / f, mdir / f)
        model = {
            "id": mid, "name": run["name"], "recipe": run["recipe"], "kind": recipe.get("kind", "custom"),
            "playground": recipe.get("playground", "json"), "run": run["id"], "created": int(time.time()),
            "result": run.get("result"), "size": _size(mdir / "files"), "servable": (mdir / "serve.py").exists(),
            "image": recipe.get("image", "python:3.12-slim"), "params": run.get("params"),
        }
        _write(mdir / "model.json", model)
        self.models[mid] = model
        run["model"] = mid
        return model

    def save_blueprint(self, name: str, arch: dict, estimates: dict, purpose: str = "") -> dict:
        """Keep a designed-but-not-trained model so it can be trained later."""
        mid = "m-blueprint-" + uuid.uuid4().hex[:8]
        (MODELS / mid).mkdir(parents=True, exist_ok=True)
        model = {"id": mid, "name": name[:80], "recipe": "llm-pretrain", "kind": "blueprint", "playground": None,
                 "run": None, "created": int(time.time()), "result": None, "size": 0, "servable": False,
                 "arch": arch, "estimates": {k: v for k, v in estimates.items() if k != "flops"}, "purpose": purpose[:200]}
        _write(MODELS / mid / "model.json", model)
        self.models[mid] = model
        self._changed()
        return model

    # ---------- layer-by-layer training: trained layer files go back into the model ----------
    def merge_layers(self, mid: str, steps: int, loss: float | None) -> dict:
        """Write the (retrained) layer files back into model.safetensors and record it, so Run it uses them."""
        from . import layers
        m = self.models[mid]
        model_dir, layers_dir = MODELS / mid / "files" / "model", MODELS / mid / "layers"
        size = layers.merge(layers_dir, model_dir)
        manifest = json.loads((layers_dir / "manifest.json").read_text())
        manifest["trained_steps"] = int(manifest.get("trained_steps") or 0) + max(0, steps)
        manifest["last_loss"] = loss
        (layers_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        m["stream_trained"] = {"steps": manifest["trained_steps"], "loss": loss, "at": int(time.time())}
        _write(MODELS / mid / "model.json", m)
        self._changed()
        return {"ok": True, "bytes": size, "trained_steps": manifest["trained_steps"]}

    # ---------- rented cloud GPUs ----------
    def _save_pods(self) -> None:
        _write(LAB / "pods.json", self.pods)

    async def rent_runpod(self, gpu_type: str, count: int, disk_gb: int = 100) -> dict:
        from . import providers
        key = self._secrets.get("RUNPOD_API_KEY")
        if not key:
            raise ValueError("Save RUNPOD_API_KEY under Secrets first")
        pod = await providers.runpod_rent(key, gpu_type, count, await self.hive_key(), disk_gb=disk_gb)
        rec = {"id": pod["id"], "provider": "runpod", "name": pod["name"], "gpu_type": gpu_type, "count": count,
               "started": int(time.time()), "target": None, "status": "starting", "cost_per_hour": None}
        self.pods[pod["id"]] = rec
        self._save_pods()
        self._changed()
        return rec

    async def attach_pod(self, pod_id: str) -> dict:
        """Wait for a rented pod's SSH, add it as a training server and test it."""
        from . import providers
        key = self._secrets.get("RUNPOD_API_KEY")
        rec = self.pods[pod_id]
        info = await providers.runpod_wait_ssh(key, pod_id)
        rec["cost_per_hour"] = info.get("cost_per_hour")
        if not rec.get("target"):
            t = self.add_target(f"RunPod {info.get('gpu') or rec['gpu_type']}", info["ssh"]["host"], int(info["ssh"]["port"]),
                                "root", "/workspace/mnx-runs", None)
            rec["target"] = t["id"]
        # The pod may take a moment to accept the key after the port appears.
        for _ in range(12):
            t = await self.test_target(rec["target"])
            if t["status"] == "ok":
                break
            await asyncio.sleep(10)
        rec["status"] = "ready" if t["status"] == "ok" else "ssh-failed"
        self._save_pods()
        self._changed()
        return rec

    async def stop_pod(self, pod_id: str) -> None:
        from . import providers
        key = self._secrets.get("RUNPOD_API_KEY")
        rec = self.pods.get(pod_id)
        if key:
            try:
                await providers.runpod_stop(key, pod_id)
            except providers.ProviderError as exc:
                if "no longer exists" not in str(exc):
                    raise
        if rec and rec.get("target") in self.targets:
            try:
                self.remove_target(rec["target"])
            except ValueError:
                pass
        self.pods.pop(pod_id, None)
        self._save_pods()
        self._changed()

    async def import_model(self, name: str, kind: str, src: Path, filename: str) -> dict:
        if kind != "gguf" or not filename.lower().endswith(".gguf"):
            raise ValueError("Import takes a .gguf file (run it with llama.cpp)")
        mid = "m-import-" + uuid.uuid4().hex[:8]
        mdir = MODELS / mid / "files"
        mdir.mkdir(parents=True)
        shutil.move(str(src), mdir / filename)
        model = {"id": mid, "name": name[:80], "recipe": None, "kind": "gguf", "playground": "chat",
                 "run": None, "created": int(time.time()), "result": None, "size": _size(mdir),
                 "servable": True, "file": filename, "image": LLAMA_IMAGE}
        _write(MODELS / mid / "model.json", model)
        self.models[mid] = model
        self._changed()
        return model

    async def delete_model(self, mid: str) -> None:
        if any(r["status"] in ("queued", "preparing", "running") and r.get("base_model") == mid for r in self.runs.values()):
            raise ValueError("A training run is still using this model as its base")
        if mid in self.deployments:
            await self.undeploy(mid)
        self.models.pop(mid, None)
        for run in self.runs.values():
            if run.get("model") == mid:
                run["model"] = None
                self._save_run(run)
        await _rmtree(MODELS / mid)
        self._changed()

    # ---------- deployments (always-on model servers on this server) ----------
    async def deploy(self, mid: str, gpu: bool = False) -> dict:
        model = self.models[mid]
        if not model.get("servable"):
            raise ValueError("This model has no serve.py, so it can't run as an API")
        if mid in self.deployments:
            await self.undeploy(mid)
        port = _free_port()
        name = f"mnx-serve-{mid}"
        mdir = (MODELS / mid).resolve()
        args = ["docker", "run", "-d", "--name", name, "--label", f"mnx.serve={mid}", "--restart", "unless-stopped",
                "-p", f"127.0.0.1:{port}:{port}", "-e", f"PORT={port}", "-e", "PYTHONUNBUFFERED=1", "-e", "PIP_ROOT_USER_ACTION=ignore",
                "-e", "MODEL_DIR=/model/files", "-e", f"MODEL_NAME={re.sub(r'[^A-Za-z0-9._-]', '-', model['name'])[:40] or 'mnx'}",
                "-v", f"{mdir}:/model:ro", "--security-opt", "no-new-privileges", "--init"]
        if gpu and self.gpu_local:
            args += ["--gpus", "all"]
        if model["kind"] == "gguf":
            args += DOCKER_ARGS + [LLAMA_IMAGE, "-m", f"/model/files/{model['file']}", "--host", "0.0.0.0",
                                   "--port", str(port), "-c", "4096"]
        else:
            args += ["-v", f"{PIP_CACHE}:/root/.cache/pip", "-w", "/model"] + DOCKER_ARGS + [
                model.get("image", "python:3.12-slim"), "sh", "-c",
                "if [ -f requirements.txt ]; then "
                "if grep -qi '^torch' requirements.txt && ! command -v nvidia-smi >/dev/null 2>&1 && [ \"$(uname -m)\" = x86_64 ]; then "
                "pip install --disable-pip-version-check --progress-bar off torch --index-url https://download.pytorch.org/whl/cpu "
                "|| echo 'CPU build not reachable; using the standard one'; fi; "
                "pip install --disable-pip-version-check --progress-bar off -r requirements.txt || exit 1; fi; "
                "exec python -u serve.py"]
        code, out = await _run(args, 900)
        if code != 0:
            raise RuntimeError(f"Couldn't start the model: {out.decode(errors='replace').strip()[-300:]}")
        dep = {"id": mid, "port": port, "container": name, "status": "starting", "started": int(time.time()),
               "gpu": bool(gpu and self.gpu_local), "error": None}
        self.deployments[mid] = dep
        self._save_deployments()
        self._changed()
        return dep

    async def undeploy(self, mid: str) -> None:
        dep = self.deployments.pop(mid, None)
        self._save_deployments()
        if dep:
            await _run(["docker", "rm", "-f", dep["container"]], 60)
        self._changed()

    def _save_deployments(self) -> None:
        _write(LAB / "deployments.json", self.deployments)

    async def deployment_logs(self, mid: str) -> str:
        dep = self.deployments.get(mid)
        if not dep:
            return ""
        code, out = await _run(["docker", "logs", "--tail", "200", dep["container"]], 30)
        return out.decode(errors="replace")

    async def _health_loop(self) -> None:
        async with httpx.AsyncClient(timeout=4) as client:
            while True:
                changed = False
                for mid, dep in list(self.deployments.items()):
                    old = dep["status"]
                    try:
                        r = await client.get(f"http://127.0.0.1:{dep['port']}/health")
                        dep["status"], dep["error"] = ("running", None) if r.status_code == 200 else ("starting", None)
                    except httpx.HTTPError:
                        code, out = await _run(["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", dep["container"]], 20)
                        state = out.decode().split() if code == 0 else ["missing"]
                        if state[0] == "running":
                            # Still installing packages or loading weights; give it time.
                            dep["status"] = "starting"
                        else:
                            dep["status"] = "failed"
                            tail = (await self.deployment_logs(mid)).strip().splitlines()[-3:]
                            dep["error"] = " · ".join(tail)[-300:] or f"Container {state[0]}"
                    changed |= dep["status"] != old
                if changed:
                    self._save_deployments()
                    self._changed()
                await asyncio.sleep(5)

    async def proxy(self, mid: str, method: str, path: str, body: bytes, content_type: str | None) -> httpx.Response:
        dep = self.deployments.get(mid)
        if not dep:
            raise LookupError("This model isn't running")
        headers = {"Content-Type": content_type} if content_type else {}
        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=5)) as client:
            return await client.request(method, f"http://127.0.0.1:{dep['port']}/{path.lstrip('/')}", content=body, headers=headers)

    # ---------- views ----------
    def run_view(self, run: dict, full: bool = False) -> dict:
        v = {k: run[k] for k in run if k not in ("metrics",)}
        pts = run["metrics"]
        v["metric_count"] = len(pts)
        v["last_metric"] = pts[-1] if pts else None
        if full:
            v["metrics"] = pts
        return v


async def _rmtree(path: Path) -> None:
    """Remove a folder; files written as root inside a container are removed through Docker."""
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        await _run(["docker", "run", "--rm", "-v", f"{path.parent.resolve()}:/p", "python:3.12-slim",
                    "rm", "-rf", f"/p/{path.name}"], 120)
        shutil.rmtree(path, ignore_errors=True)


lab = Lab()
