"""Cells: every Bee gets its own always-on workspace on the Hive server.

A Cell has
  - its own container (Docker) with a persistent /work folder, or a plain folder in local mode
  - its own browser profile, so cookies, logins and site storage survive restarts
  - a terminal that keeps running when you close the app (reconnect and you see the scrollback)
  - jobs that run 24/7: always-on (restarted if they stop), on a cron schedule, or by hand

Cells live in the gateway process, not the web page, so closing the app changes nothing.
Run the gateway as a service (services/mnx-hive.service) so it also survives reboots.

Modes (MNX_CELL_MODE): docker (isolated, recommended), local (no isolation: commands run
as the gateway's user), auto (docker if available, else local).
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import termios
import time
import uuid
from datetime import datetime
from pathlib import Path

from .browser import BrowserSession

DATA = Path(__file__).resolve().parent.parent / "data"
CELLS = DATA / "cells"
IMAGE = os.getenv("MNX_CELL_IMAGE", "python:3.12-slim")
MEMORY = os.getenv("MNX_CELL_MEMORY", "1g")
CPUS = os.getenv("MNX_CELL_CPUS", "1")
SCROLLBACK = 256 * 1024
LOG_MAX = 1024 * 1024
BROWSER_IDLE = int(os.getenv("MNX_BROWSER_IDLE", "900"))  # close idle browsers; storage stays on disk
ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def pick_mode() -> str:
    want = os.getenv("MNX_CELL_MODE", "auto").lower()
    if want == "local":
        return "local"
    if want == "docker" or _docker_ok():
        return "docker"
    return "local"


async def _run(argv: list[str], timeout: float = 60) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, "timed out"
    return proc.returncode or 0, out.decode(errors="replace")


# ---------- cron ----------
def _cron_field(field: str, lo: int, hi: int) -> set[int]:
    out: set[int] = set()
    for part in field.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/", 1)
            step = int(s)
            if step < 1:
                raise ValueError("bad step")
        if part in ("*", ""):
            a, b = lo, hi
        elif "-" in part:
            a, b = map(int, part.split("-", 1))
        else:
            a = b = int(part)
        if a < lo or b > hi or a > b:
            raise ValueError(f"{part} out of range {lo}-{hi}")
        out.update(range(a, b + 1, step))
    return out


def parse_cron(expr: str) -> list[set[int]]:
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError("cron needs 5 fields: minute hour day month weekday")
    ranges = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]
    fields = [_cron_field(p, lo, hi) for p, (lo, hi) in zip(parts, ranges)]
    if 7 in fields[4]:
        fields[4].add(0)
    return fields


def cron_matches(expr: str, dt: datetime) -> bool:
    m, h, dom, mon, dow = parse_cron(expr)
    return (dt.minute in m and dt.hour in h and dt.month in mon
            and dt.day in dom and (dt.isoweekday() % 7) in dow)


# ---------- terminal ----------
class Terminal:
    """A shell on a pseudo-terminal that outlives the browser tab that opened it."""

    def __init__(self, argv: list[str], cwd: Path, env: dict[str, str]):
        self.argv, self.cwd, self.env = argv, cwd, env
        self.proc: subprocess.Popen | None = None
        self.fd: int | None = None
        self.buffer = bytearray()
        self.listeners: set[asyncio.Queue] = set()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        if self.alive:
            return
        master, slave = os.openpty()
        self.proc = subprocess.Popen(
            self.argv, stdin=slave, stdout=slave, stderr=slave, cwd=self.cwd, env=self.env,
            start_new_session=True, close_fds=True)
        os.close(slave)
        self.fd = master
        os.set_blocking(master, False)
        asyncio.get_running_loop().add_reader(master, self._on_output)
        self.resize(120, 32)

    def _on_output(self) -> None:
        try:
            data = os.read(self.fd, 65536)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            data = b""
        if not data:
            self._closed()
            return
        self.buffer += data
        if len(self.buffer) > SCROLLBACK:
            del self.buffer[: len(self.buffer) - SCROLLBACK]
        for q in list(self.listeners):
            q.put_nowait(data)

    def _closed(self) -> None:
        if self.fd is not None:
            asyncio.get_running_loop().remove_reader(self.fd)
            os.close(self.fd)
            self.fd = None
        msg = b"\r\n[shell ended - press Enter to start a new one]\r\n"
        self.buffer += msg
        for q in list(self.listeners):
            q.put_nowait(msg)

    def write(self, data: bytes) -> None:
        if not self.alive or self.fd is None:
            self.start()
            return
        os.write(self.fd, data)

    def resize(self, cols: int, rows: int) -> None:
        if self.fd is not None:
            fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", max(rows, 2), max(cols, 10), 0, 0))

    def stop(self) -> None:
        if self.alive:
            try:
                os.killpg(self.proc.pid, signal.SIGHUP)
            except ProcessLookupError:
                pass


# ---------- cell ----------
class Cell:
    def __init__(self, bee_id: str, name: str, mode: str):
        if not ID_RE.match(bee_id):
            raise ValueError("bad bee id")
        self.id, self.name, self.mode = bee_id, name, mode
        self.dir = CELLS / bee_id
        self.work = self.dir / "work"
        self.logs = self.dir / "logs"
        for d in (self.work, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        self.container = f"mnx-cell-{bee_id.replace('_', '-')}"
        self.browser = BrowserSession(self.dir / "browser")
        self.terminal: Terminal | None = None
        self.jobs: dict[str, dict] = {j["id"]: j for j in self._read_jobs()}
        self.running: dict[str, asyncio.subprocess.Process] = {}
        self.loops: dict[str, asyncio.Task] = {}
        self.container_state = "local" if mode == "local" else "missing"
        self.on_change = None  # set by the manager
        self._disk = (0.0, 0)
        self._up_lock = asyncio.Lock()

    # ----- container -----
    async def ensure_up(self) -> None:
        if self.mode != "docker":
            return
        # One starter at a time: a new Bee's boot and its first terminal can arrive together,
        # and two `docker run`s with the same name make Docker refuse the second.
        async with self._up_lock:
            await self._ensure_up()

    async def _ensure_up(self) -> None:
        code, out = await _run(["docker", "inspect", "-f", "{{.State.Running}}", self.container], 20)
        if code == 0 and out.strip() == "true":
            self.container_state = "running"
            return
        if code == 0:
            code, out = await _run(["docker", "start", self.container], 60)
        else:
            code, out = await _run([
                "docker", "run", "-d", "--name", self.container, "--hostname", self.id.replace("_", "-"),
                "--restart", "unless-stopped", "--memory", MEMORY, "--cpus", CPUS, "--pids-limit", "512",
                "--security-opt", "no-new-privileges", "--label", "mnx.cell=" + self.id,
                "-v", f"{self.work.resolve()}:/work", "-w", "/work", "-e", "HOME=/work",
                IMAGE, "sleep", "infinity"], 600)
        self.container_state = "running" if code == 0 else "failed"
        if code != 0:
            raise RuntimeError(f"Couldn't start the Cell container: {out.strip()[:300]}")

    async def container_status(self) -> str:
        if self.mode != "docker":
            return "local"
        code, out = await _run(["docker", "inspect", "-f", "{{.State.Status}}", self.container], 20)
        self.container_state = out.strip() if code == 0 else "missing"
        return self.container_state

    def _env(self) -> dict[str, str]:
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "TERM": "xterm-256color", "LANG": "C.UTF-8",
               "HOME": str(self.work), "MNX_BEE": self.id}
        return env

    def shell_argv(self) -> list[str]:
        if self.mode == "docker":
            return ["docker", "exec", "-it", "-w", "/work", "-e", "TERM=xterm-256color", self.container,
                    "sh", "-c", "command -v bash >/dev/null && exec bash -l || exec sh -l"]
        return ["bash", "-l"] if shutil.which("bash") else ["sh", "-l"]

    async def terminal_session(self) -> Terminal:
        await self.ensure_up()
        if not self.terminal:
            self.terminal = Terminal(self.shell_argv(), self.work, self._env())
        if not self.terminal.alive:
            self.terminal.start()
        return self.terminal

    async def exec(self, command: str, timeout: float = 120) -> tuple[int, str]:
        await self.ensure_up()
        if self.mode == "docker":
            return await _run(["docker", "exec", "-w", "/work", self.container, "sh", "-lc", command], timeout)
        proc = await asyncio.create_subprocess_exec(
            "sh", "-lc", command, cwd=self.work, env=self._env(),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            os.killpg(proc.pid, signal.SIGKILL)
            return 124, "timed out"
        return proc.returncode or 0, out.decode(errors="replace")

    # ----- files (inside work/) -----
    def safe_path(self, rel: str) -> Path:
        p = (self.work / rel.lstrip("/")).resolve()
        if p != self.work.resolve() and self.work.resolve() not in p.parents:
            raise ValueError("Path is outside the Cell")
        return p

    def list_files(self, rel: str = "") -> list[dict]:
        base = self.safe_path(rel)
        out = []
        for e in sorted(base.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            st = e.lstat()
            out.append({"name": e.name, "dir": e.is_dir(), "size": st.st_size, "mtime": int(st.st_mtime),
                        "path": str(e.relative_to(self.work))})
        return out

    async def disk_usage_cached(self, max_age: float = 60) -> int:
        """Walking a browser profile is slow; do it in a thread, at most once a minute."""
        if time.monotonic() - self._disk[0] > max_age:
            self._disk = (time.monotonic(), await asyncio.to_thread(self.disk_usage))
        return self._disk[1]

    def disk_usage(self) -> int:
        total = 0
        for root, _, files in os.walk(self.dir):
            for f in files:
                try:
                    total += os.lstat(os.path.join(root, f)).st_size
                except OSError:
                    pass
        return total

    # ----- jobs -----
    def _read_jobs(self) -> list[dict]:
        try:
            return json.loads((self.dir / "jobs.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_jobs(self) -> None:
        tmp = self.dir / "jobs.tmp"
        tmp.write_text(json.dumps(list(self.jobs.values()), indent=2))
        tmp.replace(self.dir / "jobs.json")

    def add_job(self, name: str, command: str, schedule: str) -> dict:
        if schedule not in ("@always", "@manual"):
            parse_cron(schedule)
        job = {"id": uuid.uuid4().hex[:8], "name": name, "command": command, "schedule": schedule,
               "enabled": True, "created": int(time.time()), "runs": 0,
               "last_start": None, "last_end": None, "last_exit": None}
        self.jobs[job["id"]] = job
        self.save_jobs()
        if schedule == "@always":
            self.start_always(job["id"])
        return job

    def log_path(self, job_id: str) -> Path:
        return self.logs / f"{job_id}.log"

    def _log(self, job_id: str, text: str, fh=None) -> None:
        if fh is not None:
            fh.write(text.encode())
            return
        path = self.log_path(job_id)
        with open(path, "ab") as f:
            f.write(text.encode())
        self._trim(job_id)

    def _trim(self, job_id: str) -> None:
        path = self.log_path(job_id)
        try:
            if path.stat().st_size > LOG_MAX:
                data = path.read_bytes()[-LOG_MAX // 2:]
                path.write_bytes(b"[older output trimmed]\n" + data)
        except FileNotFoundError:
            pass

    async def run_job(self, job_id: str) -> int:
        job = self.jobs[job_id]
        if job_id in self.running:
            return -1
        await self.ensure_up()
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log(job_id, f"\n=== {stamp} start: {job['command']}\n")
        job["last_start"], job["runs"] = int(time.time()), job["runs"] + 1
        self.save_jobs()
        pidfile = f"/tmp/mnx-job-{job_id}.pid"
        if self.mode == "docker":
            # A run left over from before a gateway restart keeps going inside the container: end it first.
            await self._kill_in_container(job_id)
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", "-w", "/work", "-e", f"MNX_CMD={job['command']}", self.container,
                "sh", "-c", f'echo $$ > {pidfile}; exec sh -lc "$MNX_CMD"',
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        else:
            proc = await asyncio.create_subprocess_exec(
                "sh", "-lc", job["command"], cwd=self.work, env=self._env(),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        self.running[job_id] = proc
        self._changed()
        written = 0
        fh = open(self.log_path(job_id), "ab", buffering=0)
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                fh.write(line)
                written += len(line)
                if written > 256 * 1024:
                    fh.close()
                    self._trim(job_id)
                    fh = open(self.log_path(job_id), "ab", buffering=0)
                    written = 0
            code = await proc.wait()
        finally:
            fh.close()
            self.running.pop(job_id, None)
        self._log(job_id, f"=== exit {code}\n")
        job["last_end"], job["last_exit"] = int(time.time()), code
        self.save_jobs()
        self._changed()
        return code

    async def stop_job(self, job_id: str) -> None:
        loop = self.loops.pop(job_id, None)
        if loop:
            loop.cancel()
        proc = self.running.get(job_id)
        if not proc:
            return
        if self.mode == "docker":
            await self._kill_in_container(job_id)
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(proc.wait(), 10)
        except asyncio.TimeoutError:
            proc.kill()

    async def _kill_in_container(self, job_id: str) -> None:
        pidfile = f"/tmp/mnx-job-{job_id}.pid"
        # Kill the job's whole process tree (slim images have no pkill, so walk /proc).
        script = (
            f'p=$(cat {pidfile} 2>/dev/null) || exit 0; '
            'kids() { for s in /proc/[0-9]*/stat; do set -- $(cat $s 2>/dev/null); [ "$4" = "$1" ] && continue; '
            '[ "$4" = "$P" ] && echo $1; done; }; '
            'all=$p; todo=$p; while [ -n "$todo" ]; do n=""; for P in $todo; do n="$n $(kids)"; done; todo=$(echo $n); all="$all $todo"; done; '
            f'kill -TERM $all 2>/dev/null; rm -f {pidfile}; true'
        )
        await _run(["docker", "exec", self.container, "sh", "-c", script], 20)

    def start_always(self, job_id: str) -> None:
        if job_id in self.loops and not self.loops[job_id].done():
            return

        async def keep_running():
            delay = 5
            while self.jobs.get(job_id, {}).get("enabled"):
                started = time.monotonic()
                try:
                    await self.run_job(job_id)
                except Exception as exc:
                    self._log(job_id, f"=== couldn't start: {exc}\n")
                if time.monotonic() - started > 300:
                    delay = 5  # it ran for a while: restart quickly
                self._log(job_id, f"=== restarting in {delay}s\n")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 300)

        self.loops[job_id] = asyncio.create_task(keep_running())

    async def set_enabled(self, job_id: str, enabled: bool) -> None:
        job = self.jobs[job_id]
        job["enabled"] = enabled
        self.save_jobs()
        if not enabled:
            await self.stop_job(job_id)
        elif job["schedule"] == "@always":
            self.start_always(job_id)

    async def delete_job(self, job_id: str) -> None:
        await self.stop_job(job_id)
        self.jobs.pop(job_id, None)
        self.save_jobs()
        self.log_path(job_id).unlink(missing_ok=True)

    def _changed(self) -> None:
        if self.on_change:
            self.on_change(self)

    # ----- views -----
    def view(self) -> dict:
        failed = any(j["last_exit"] not in (None, 0, -15, 143) and j["id"] not in self.running for j in self.jobs.values())
        return {
            "bee": self.id, "name": self.name, "mode": self.mode, "container": self.container_state,
            "image": IMAGE if self.mode == "docker" else None,
            "terminal": bool(self.terminal and self.terminal.alive),
            "browser": self.browser.running,
            "jobs": len(self.jobs), "running": len(self.running), "failed": failed,
        }

    def job_views(self) -> list[dict]:
        return [{**j, "running": j["id"] in self.running} for j in self.jobs.values()]

    async def destroy(self) -> None:
        for jid in list(self.jobs):
            await self.stop_job(jid)
        if self.terminal:
            self.terminal.stop()
        await self.browser.shutdown()
        if self.mode == "docker":
            # Files written inside the container belong to its root user; remove them from inside.
            if await self.container_status() == "running":
                await _run(["docker", "exec", self.container, "sh", "-c", "rm -rf /work/* /work/.[!.]* /work/..?*"], 120)
            await _run(["docker", "rm", "-f", self.container], 60)
        shutil.rmtree(self.dir, ignore_errors=True)


# ---------- manager ----------
class CellManager:
    def __init__(self) -> None:
        self.mode = "local"
        self.cells: dict[str, Cell] = {}
        self.on_change = None
        self._tasks: list[asyncio.Task] = []

    async def start(self, bees: list[dict]) -> None:
        self.mode = await asyncio.to_thread(pick_mode)
        print(f"[mnx] Cells run in {self.mode} mode", flush=True)
        for b in bees:
            cell = self.get(b["id"], b["name"])
            for job in cell.jobs.values():
                if job["schedule"] == "@always" and job["enabled"]:
                    cell.start_always(job["id"])
        await self.refresh_states()
        self._tasks = [asyncio.create_task(self._cron_loop()), asyncio.create_task(self._idle_loop())]

    async def refresh_states(self) -> None:
        """One `docker ps` for every Cell's container state."""
        if self.mode != "docker":
            return
        code, out = await _run(["docker", "ps", "-a", "--filter", "label=mnx.cell",
                                "--format", '{{.Label "mnx.cell"}} {{.State}}'], 20)
        if code != 0:
            return
        states = dict(line.split(" ", 1) for line in out.splitlines() if " " in line)
        for cell in self.cells.values():
            cell.container_state = states.get(cell.id, "missing")

    def get(self, bee_id: str, name: str | None = None) -> Cell:
        cell = self.cells.get(bee_id)
        if not cell:
            cell = Cell(bee_id, name or bee_id, self.mode)
            cell.on_change = lambda c: self.on_change and self.on_change(c)
            self.cells[bee_id] = cell
        return cell

    async def remove(self, bee_id: str) -> None:
        cell = self.cells.pop(bee_id, None) or Cell(bee_id, bee_id, self.mode)
        await cell.destroy()

    def busy(self, bee_id: str) -> str | None:
        cell = self.cells.get(bee_id)
        if not cell:
            return None
        if cell.running:
            return "busy"
        if cell.view()["failed"]:
            return "failed"
        if any(j["enabled"] and j["schedule"] != "@manual" for j in cell.jobs.values()):
            return "scheduled"
        return None

    async def _cron_loop(self) -> None:
        last = None
        while True:
            now = datetime.now().replace(second=0, microsecond=0)
            if now != last:
                last = now
                for cell in list(self.cells.values()):
                    for job in list(cell.jobs.values()):
                        sched = job["schedule"]
                        if not job["enabled"] or sched.startswith("@") or job["id"] in cell.running:
                            continue
                        try:
                            if cron_matches(sched, now):
                                asyncio.create_task(cell.run_job(job["id"]))
                        except ValueError:
                            pass
            await asyncio.sleep(max(1, 60 - datetime.now().second))

    async def _idle_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            await self.refresh_states()
            for cell in list(self.cells.values()):
                if cell.browser.running and time.monotonic() - cell.browser.last_used > BROWSER_IDLE:
                    await cell.browser.shutdown()

    async def shutdown(self) -> None:
        for t in self._tasks:
            t.cancel()
        for cell in self.cells.values():
            for loop in cell.loops.values():
                loop.cancel()
            await cell.browser.shutdown()
        # Containers keep running (restart=unless-stopped); jobs restart when the gateway does.


cells = CellManager()
