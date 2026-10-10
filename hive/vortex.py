"""Vortex OS on every Bee's cloud PC.

Vortex OS (https://github.com/Tharunmaks/Vortex-os-) is a Debian web desktop: windows, a real bash terminal,
files, an editor, a from-scratch browser, a system monitor. Each Bee whose computer runs Vortex gets its own
desktop, signed in as the Bee, whose home folder IS the Bee's work folder (the same files its terminal,
jobs and the Cell page see).

Where it runs, by where the Bee's computer is:
  - this server with Docker: the Bee's computer container is the Vortex image itself (hardened as in the
    Vortex compose file: read-only root, no capabilities, its own CPU / memory limits); the image is built
    from the Vortex repository the first time
  - this server without Docker (a phone under Termux): a Vortex process per Bee, from a checkout in
    MNX_VORTEX_DIR (cloned and installed the first time it is opened)
  - a cloud server: Vortex runs there (Docker if it has it, else Node), and the Hive opens an SSH tunnel to it

Each desktop listens on its own port on MNX_VORTEX_BIND (127.0.0.1 by default: open it on the machine running
the Hive, or set 0.0.0.0 to reach it from your network; it always asks for the Bee's password). The Hive
creates the account and keeps the password in data/vortex.json (owner-only).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import string
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPO = os.getenv("MNX_VORTEX_REPO", "https://github.com/Tharunmaks/Vortex-os-.git")
IMAGE = os.getenv("MNX_VORTEX_IMAGE", "mnx/vortex-os:latest")
DIR = Path(os.getenv("MNX_VORTEX_DIR", str(ROOT / "vortex-os")))
BIND = os.getenv("MNX_VORTEX_BIND", "127.0.0.1")
PORT_BASE = int(os.getenv("MNX_VORTEX_PORT_BASE", "8421"))
CREDS_FILE = DATA / "vortex.json"
BEE_UID = 10001  # the `bee` user inside the Vortex image
USER_RE = re.compile(r"^[a-z][a-z0-9_-]{2,31}$")
OS_CHOICES = {"vortex": "Vortex OS (desktop)", "plain": "Plain Linux (terminal only)"}


def username_for(bee_id: str) -> str:
    """Vortex wants 3-32 chars, a-z 0-9 _ -, starting with a letter."""
    u = bee_id.replace("_", "-")
    return u if USER_RE.match(u) else f"bee-{u}"[:32]


def new_password(username: str) -> str:
    """Strong enough for Vortex: 3+ character classes, never containing the user name."""
    alpha = string.ascii_letters + string.digits
    while True:
        pw = "".join(secrets.choice(alpha) for _ in range(20)) + "-" + secrets.choice(string.digits)
        if (username.lower() not in pw.lower() and any(c.islower() for c in pw) and any(c.isupper() for c in pw)
                and any(c.isdigit() for c in pw)):
            return pw


def port_free(port: int, host: str = BIND) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


class Creds:
    """Per-Bee desktop account and port, kept in an owner-only file."""

    def __init__(self, path: Path = CREDS_FILE):
        self.path = path
        try:
            self.data: dict[str, dict] = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    def get(self, bee_id: str, check_free=port_free) -> dict:
        rec = self.data.get(bee_id)
        if rec:
            return rec
        user = username_for(bee_id)
        used = {r["port"] for r in self.data.values()}
        port = PORT_BASE
        while port in used or not check_free(port):
            port += 1
        rec = self.data[bee_id] = {"user": user, "password": new_password(user), "port": port}
        self.save()
        return rec

    def forget(self, bee_id: str) -> None:
        if self.data.pop(bee_id, None) is not None:
            self.save()


creds = Creds()


# ---------------------------------------------------------------- Docker: the Bee's computer container is Vortex
def docker_run_args(container: str, bee_id: str, name: str, work: Path, vdir: Path, limits: list[str], rec: dict,
                    image: str = IMAGE, bind: str = BIND) -> list[str]:
    """`docker run` for a Bee's computer running Vortex OS, hardened like Vortex's own compose file."""
    user = rec["user"]
    return ["docker", "run", "-d", "--name", container, "--hostname", bee_id.replace("_", "-"),
            "--restart", "unless-stopped", *limits, "--pids-limit", "512",
            "--security-opt", "no-new-privileges", "--cap-drop", "ALL", "--read-only",
            "--tmpfs", "/tmp:size=512m,mode=1777", "--tmpfs", f"/home/bee:size=64m,uid={BEE_UID},gid={BEE_UID}",
            "--label", "mnx.cell=" + bee_id, "--label", "mnx.os=vortex", "--add-host", "host.docker.internal:host-gateway",
            "-v", f"{work.resolve()}:/work", "-v", f"{work.resolve()}:/homes/{user}", "-v", f"{vdir.resolve()}:/var/lib/vortex",
            "-e", "HOME=/work", "-e", "VORTEX_HOMES_DIR=/homes", "-e", f"VORTEX_ADMIN_USER={user}",
            "-e", f"VORTEX_ADMIN_PASSWORD={rec['password']}", "-e", f"VORTEX_HIVE_NAME={name} Bee · MNX Hive",
            "-p", f"{bind}:{rec['port']}:8420", image]


async def _run(argv: list[str], timeout: float = 60, stdin: bytes | None = None, cwd: Path | None = None) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=cwd)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(stdin), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, "timed out"
    return proc.returncode or 0, out.decode(errors="replace")


_image_lock = asyncio.Lock()


async def ensure_image(image: str = IMAGE) -> None:
    """Build the Vortex image from its repository the first time (Docker builds straight from the git URL)."""
    async with _image_lock:
        code, _ = await _run(["docker", "image", "inspect", image], 30)
        if code == 0:
            return
        code, out = await _run(["docker", "build", "-t", image, REPO], 1800)
        if code != 0:
            raise RuntimeError(f"Couldn't build Vortex OS ({REPO}): {out.strip()[-400:]}")


async def give_to_bee(paths: list[Path], image: str = IMAGE) -> None:
    """The Vortex container runs as uid 10001: hand it the Bee's folders."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
    if os.geteuid() == 0:
        for p in paths:
            for root, dirs, files in os.walk(p):
                for x in [root, *(os.path.join(root, d) for d in dirs), *(os.path.join(root, f) for f in files)]:
                    try:
                        os.lchown(x, BEE_UID, BEE_UID)
                    except OSError:
                        pass
        return
    mounts = []
    for i, p in enumerate(paths):
        mounts += ["-v", f"{p.resolve()}:/m{i}"]
    await _run(["docker", "run", "--rm", "-u", "0", "--entrypoint", "chown", *mounts, image, "-R", f"{BEE_UID}:{BEE_UID}",
                *[f"/m{i}" for i in range(len(paths))]], 300)


# ---------------------------------------------------------------- no Docker: a Vortex process per Bee
_install_lock = asyncio.Lock()


async def ensure_checkout(directory: Path | None = None) -> None:
    """Clone Vortex and install its packages once (node-pty compiles here; Termux needs python, make, clang)."""
    directory = directory or DIR
    async with _install_lock:
        if (directory / "node_modules" / "node-pty").exists() and (directory / "server" / "index.js").exists():
            return
        if not shutil.which("node") or not shutil.which("npm"):
            raise RuntimeError("Vortex OS needs Node.js 20+ here (Termux: pkg install nodejs python make clang)")
        if not (directory / "server" / "index.js").exists():
            if not shutil.which("git"):
                raise RuntimeError("Vortex OS needs git to download it")
            code, out = await _run(["git", "clone", "--depth", "1", REPO, str(directory)], 600)
            if code != 0:
                raise RuntimeError(f"Couldn't download Vortex OS: {out.strip()[-300:]}")
        npm = ["npm", "ci", "--omit=dev"] if (directory / "package-lock.json").exists() else ["npm", "install", "--omit=dev"]
        code, out = await _run(npm, 1800, cwd=directory)
        if code != 0:
            raise RuntimeError(f"Couldn't install Vortex OS's packages: {out.strip()[-400:]}")


def local_env(cell_dir: Path, work: Path, name: str, rec: dict, bind: str = BIND) -> dict:
    homes = cell_dir / "homes"
    homes.mkdir(parents=True, exist_ok=True)
    home = homes / rec["user"]
    if not home.exists() and not home.is_symlink():
        home.symlink_to(work.resolve(), target_is_directory=True)  # the desktop's home is the Bee's work folder
    env = {**os.environ, "VORTEX_HOST": bind, "VORTEX_PORT": str(rec["port"]), "VORTEX_DATA_DIR": str(cell_dir / "vortex"),
           "VORTEX_HOMES_DIR": str(homes), "VORTEX_ADMIN_USER": rec["user"], "VORTEX_ADMIN_PASSWORD": rec["password"],
           "VORTEX_HIVE_NAME": f"{name} Bee · MNX Hive", "VORTEX_SHELL_PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    bash = shutil.which("bash")
    if bash:
        env["VORTEX_SHELL"] = bash
    return env


# ---------------------------------------------------------------- a cloud server: run it there, tunnel to it
def remote_script(rec: dict, name: str, cell_dir: str, bee_id: str) -> str:
    """Start Vortex on a remote server (Docker if it has it, else Node), serving the Bee's remote Cell folder."""
    user, port = rec["user"], rec["port"]
    env = (f"VORTEX_HOST=127.0.0.1 VORTEX_PORT={port} VORTEX_DATA_DIR={cell_dir}/.vortex VORTEX_HOMES_DIR={cell_dir}/.homes "
           f"VORTEX_ADMIN_USER={user} VORTEX_ADMIN_PASSWORD={shlex.quote(rec['password'])} "
           f"VORTEX_HIVE_NAME={shlex.quote(name + ' Bee · MNX Hive')}")
    container = f"mnx-vortex-{bee_id.replace('_', '-')}"
    return f"""set -e
cd {cell_dir}
if curl -fsS -m 3 http://127.0.0.1:{port}/api/hive >/dev/null 2>&1; then echo MNX_VORTEX_UP; exit 0; fi
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker image inspect {IMAGE} >/dev/null 2>&1 || docker build -t {IMAGE} {shlex.quote(REPO)} >/dev/null
  mkdir -p .vortex
  docker rm -f {container} >/dev/null 2>&1 || true
  docker run -d --name {container} --restart unless-stopped --cap-drop ALL --read-only --security-opt no-new-privileges \\
    --tmpfs /tmp:size=512m,mode=1777 -u "$(id -u):$(id -g)" -e HOME=/homes/{user} \\
    -v "$PWD":/homes/{user} -v "$PWD/.vortex":/var/lib/vortex -e VORTEX_HOMES_DIR=/homes -e VORTEX_ADMIN_USER={user} \\
    -e VORTEX_ADMIN_PASSWORD={shlex.quote(rec['password'])} -e VORTEX_HIVE_NAME={shlex.quote(name + ' Bee · MNX Hive')} \\
    -p 127.0.0.1:{port}:8420 {IMAGE} >/dev/null
elif command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
  V="$HOME/mnx-vortex"
  [ -f "$V/server/index.js" ] || git clone --depth 1 {shlex.quote(REPO)} "$V" >/dev/null 2>&1
  [ -d "$V/node_modules/node-pty" ] || (cd "$V" && npm ci --omit=dev >/dev/null 2>&1)
  mkdir -p .homes .vortex && [ -e .homes/{user} ] || ln -s "$PWD" .homes/{user}
  {env} VORTEX_DATA_DIR="$PWD/.vortex" VORTEX_HOMES_DIR="$PWD/.homes" setsid nohup node "$V/server/index.js" > .vortex.log 2>&1 &
else
  echo "MNX_VORTEX_NEEDS: Docker or Node.js 20+ on this server"; exit 3
fi
for i in $(seq 1 60); do curl -fsS -m 2 http://127.0.0.1:{port}/api/hive >/dev/null 2>&1 && {{ echo MNX_VORTEX_UP; exit 0; }}; sleep 2; done
echo "MNX_VORTEX_NEEDS: it didn't start; see {cell_dir}/.vortex.log"; exit 4
"""


def tunnel_argv(ssh: list[str], local_port: int, remote_port: int, bind: str = BIND) -> list[str]:
    """ssh … user@host -- (the Lab's base) → a port forward: options before the host, no command."""
    host = ssh[-2] if ssh and ssh[-1] == "--" else ssh[-1]
    opts = ssh[:-2] if ssh and ssh[-1] == "--" else ssh[:-1]
    return [*opts, "-N", "-o", "ExitOnForwardFailure=yes", "-L", f"{bind}:{local_port}:127.0.0.1:{remote_port}", host]


# ---------------------------------------------------------------- one desktop per Bee
class Desktops:
    """Starts, watches and stops the Bees' Vortex desktops (local processes and SSH tunnels; containers persist)."""

    def __init__(self) -> None:
        self.procs: dict[str, subprocess.Popen] = {}
        self.tunnels: dict[str, subprocess.Popen] = {}
        self.state: dict[str, dict] = {}  # bee → {"status": starting|running|failed, "error": ...}

    def status(self, cell) -> dict:
        rec = creds.data.get(cell.id)
        os_name = (cell.computer or {}).get("os", "vortex")
        out = {"os": os_name, "os_label": OS_CHOICES.get(os_name, os_name), "where": cell.view()["server"],
               "user": rec["user"] if rec else None, "password": rec["password"] if rec else None, "port": rec["port"] if rec else None,
               "bind": BIND, **self.state.get(cell.id, {"status": "stopped"})}
        if os_name != "vortex":
            out["status"] = "off"
            return out
        if rec and port_open(rec["port"]):
            out["status"] = "running"
            out.pop("error", None)
        elif out["status"] == "running":
            out["status"] = "stopped"
        return out

    async def start(self, cell) -> dict:
        if (cell.computer or {}).get("os", "vortex") != "vortex":
            raise RuntimeError("This Bee's computer runs plain Linux; switch its OS to Vortex OS first")
        rec = creds.get(cell.id)
        if port_open(rec["port"]):
            self.state[cell.id] = {"status": "running"}
            return self.status(cell)
        self.state[cell.id] = {"status": "starting"}
        try:
            if cell.mode == "docker":
                await cell.ensure_up()  # the container is Vortex: starting the computer starts the desktop
            elif cell.mode == "remote":
                await self._start_remote(cell, rec)
            else:
                await self._start_local(cell, rec)
            for _ in range(60):
                if port_open(rec["port"]):
                    break
                await asyncio.sleep(1)
            else:
                raise RuntimeError("Vortex OS didn't answer on its port within a minute" + self._log_tail(cell))
            self.state[cell.id] = {"status": "running"}
        except Exception as exc:
            self.state[cell.id] = {"status": "failed", "error": str(exc)[:500]}
            raise
        return self.status(cell)

    def _log_tail(self, cell) -> str:
        try:
            tail = (cell.dir / "vortex.log").read_text(errors="replace")[-300:].strip()
            return f": {tail}" if tail else ""
        except OSError:
            return ""

    async def _start_local(self, cell, rec: dict) -> None:
        p = self.procs.get(cell.id)
        if p and p.poll() is None:
            return
        await ensure_checkout()
        env = local_env(cell.dir, cell.work, cell.name, rec)
        log = open(cell.dir / "vortex.log", "ab")
        self.procs[cell.id] = subprocess.Popen(["node", str(DIR / "server" / "index.js")], cwd=str(DIR), env=env,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    async def _start_remote(self, cell, rec: dict) -> None:
        await cell.ensure_up()
        from .cells import rpath
        script = remote_script(rec, cell.name, rpath(cell.remote["dir"]), cell.id)
        code, out = await _run(cell.remote["ssh"] + ["bash -s"], 1800, script.encode())
        if "MNX_VORTEX_UP" not in out:
            need = re.search(r"MNX_VORTEX_NEEDS: (.+)", out)
            raise RuntimeError(f"On {cell.remote['name']}: " + (need.group(1) if need else out.strip()[-300:] or f"exit {code}"))
        t = self.tunnels.get(cell.id)
        if not t or t.poll() is not None:
            self.tunnels[cell.id] = subprocess.Popen(tunnel_argv(cell.remote["ssh"], rec["port"], rec["port"]),
                                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

    async def stop(self, cell) -> None:
        for table in (self.procs, self.tunnels):
            p = table.pop(cell.id, None)
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()
        if cell.mode == "remote":
            rec = creds.data.get(cell.id)
            if rec:
                c = f"mnx-vortex-{cell.id.replace('_', '-')}"
                await _run(cell.remote["ssh"] + [f"docker rm -f {c} >/dev/null 2>&1; pkill -f 'VORTEX_PORT={rec['port']}' 2>/dev/null; true"], 60)
        self.state[cell.id] = {"status": "stopped"}

    async def forget(self, cell) -> None:
        await self.stop(cell)
        creds.forget(cell.id)

    def shutdown(self) -> None:
        for table in (self.procs, self.tunnels):
            for p in table.values():
                if p.poll() is None:
                    p.terminate()


desktops = Desktops()

