"""Your own Mnx models: downloaded from your Google Drive, put back together, checked, and run as the Hive's brain.

The models live in Google Drive split into 95 MB parts (name.part00, name.part01 …). The owner pastes each model's
Drive folder link once (Models, on the chat's model button); links are kept in data/models/sources.json, never in the
code (the repo is public). The folder must be shared as "Anyone with the link".

Then the Hive:
  1. lists the folder (gdown, no Google login), downloads every part (resumable), joins them in order,
  2. checks each file against the MD5s in the catalog below and in the folder's own MD5SUMS / manifest files,
  3. runs it: a GGUF with llama.cpp's server (Termux: pkg install llama-cpp; elsewhere the llama-server binary
     or the llama.cpp Docker image), then makes it the brain (Settings → Brain), so every chat and bot thinks with it.
     Other formats run with the owner's own start command (their code), on a port the brain then points at.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import re
import shutil
import signal
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

DATA = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = DATA / "models"
PORT = int(os.getenv("MNX_MODEL_PORT", "8081"))
PART = re.compile(r"^(?P<base>.+)\.part(?P<n>\d+)$")
MD5_LINE = re.compile(r"^\s*(?P<md5>[0-9a-f]{32})\s+(?:(?P<size>\d+)\s+)?\*?(?P<name>\S.*?)\s*$")

# What each model is. MD5s and sizes come from the models' own READMEs. No links here: the repo is public.
CATALOG = [
    {"id": "mnx-1.5-max-coder", "name": "Mnx 1.5 Max Coder",
     "about": "Your coding model, small enough for the phone (Q4_K_M GGUF).",
     "kind": "gguf", "run": "llama.cpp", "ram_gb": 2, "disk_gb": 0.95,
     "main": "mnx-1.5-max-coder-q4_k_m.gguf",
     "md5": {"mnx-1.5-max-coder-q4_k_m.gguf": "b3e450e066fae5e18d2738344b3a6ef8"},
     "sizes": {"mnx-1.5-max-coder-q4_k_m.gguf": 986048128}},
    {"id": "mnx-1-lite", "name": "Mnx 1 Lite",
     "about": "Your 1.3B mixture-of-experts model, trained from scratch (checkpoint at 320M of 1B tokens; still training).",
     "kind": "pytorch", "run": "its own code", "ram_gb": 6, "disk_gb": 2.7,
     "main": "mnx_lite1.pt",
     "md5": {"mnx_lite1.pt": "9d0e6c88eaf867bc499524e6e7b697e6"}, "sizes": {"mnx_lite1.pt": 2674569846}},
    {"id": "mnx-1", "name": "Mnx 1 (7B)",
     "about": "Your 7B model, final checkpoint (training finished Oct 16 2026): table.bin + core.pt.",
     "kind": "custom", "run": "its own code (race7b scripts)", "ram_gb": 16, "disk_gb": 15.8,
     "main": "core.pt",
     "md5": {"table.bin": "9639b782ab2649bd178fe3152b500f61", "core.pt": "2fdf05d2da61aa8fffaee0d9cbe97a2d",
             "core_only.pt": "1ec8721612395dba450dbff26f5cec0e"},
     "sizes": {"table.bin": 13747503104, "core.pt": 1514987917, "core_only.pt": 504997975}},
    {"id": "mnx-lonjer", "name": "Mnx lonjer",
     "about": "Mnx 1 on Qwen3.6-35B-A3B (int4, colibri engine) with your Mnx app: tools, MCP, self-coding.",
     "kind": "colibri", "run": "run_mnx1.sh (colibri)", "ram_gb": 24, "disk_gb": 23,
     "main": "mnx1_app_code.tar.gz",
     "md5": {"mnx1_app_code.tar.gz": "966e8ad63cf3926abac1d408f0baf473"}, "sizes": {}},
]
selected: contextvars.ContextVar[str | None] = contextvars.ContextVar("mnx_selected_model", default=None)
DRIVE_FOLDER = re.compile(r"(?:folders/|[?&]id=)([A-Za-z0-9_-]{20,})")


def threads_for() -> str:
    """Threads for the model server. All cores is slower on a phone: its small efficiency cores make every step wait for the
    slowest thread, so a phone gets 4 (its fast cores). MNX_THREADS overrides."""
    if os.getenv("MNX_THREADS", "").strip().isdigit():
        return os.environ["MNX_THREADS"].strip()
    cores = os.cpu_count() or 2
    phone = bool(os.getenv("TERMUX_VERSION")) or os.path.isdir("/data/data/com.termux")
    return str(max(1, min(4, cores - 1) if phone else cores - 1))


class ModelError(RuntimeError):
    pass


def folder_id(link: str) -> str:
    link = (link or "").strip()
    m = DRIVE_FOLDER.search(link)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", link):
        return link
    raise ModelError("That isn't a Google Drive folder link (it looks like https://drive.google.com/drive/folders/…)")


def plan_joins(names: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Which parts make which file (in numeric order), and which files are whole already."""
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    whole = []
    for n in names:
        m = PART.match(n)
        if m:
            groups[m.group("base")].append((int(m.group("n")), n))
        else:
            whole.append(n)
    joins = {}
    for base, parts in groups.items():
        parts.sort()
        nums = [p[0] for p in parts]
        if nums != list(range(nums[0], nums[0] + len(nums))):
            missing = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
            raise ModelError(f"{base} is missing part{'s' if len(missing) > 1 else ''} {', '.join(map(str, missing[:8]))}")
        joins[base] = [p[1] for p in parts]
    return joins, whole


def md5_of(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_md5_lists(root: Path) -> dict[str, str]:
    """MD5s the folder lists itself (MD5SUMS*.txt, manifest_*.txt, and "md5  name" lines in README files)."""
    out = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.stat().st_size > 2_000_000:
            continue
        low = p.name.lower()
        if not (low.startswith("md5sums") or low.startswith("manifest") or low.startswith("readme")):
            continue
        for line in p.read_text(errors="replace").splitlines():
            m = MD5_LINE.match(line)
            if m:
                out[Path(m.group("name").strip()).name] = m.group("md5")
    return out


class Models:
    def __init__(self, root: Path = MODELS_DIR):
        self.root = root
        self.quiet = True       # the command line turns gdown's progress bar on
        self.lister = None      # tests swap these in: (folder id) → [(file id, relative path)]
        self.fetcher = None     # (file id, dest Path, progress cb) → None
        self.jobs: dict[str, asyncio.Task] = {}
        self.procs: dict[str, asyncio.subprocess.Process] = {}
        self.progress: dict[str, dict] = {}
        self.on_change = None
        self._lock = asyncio.Lock()
        self.inflight: dict[str, int] = defaultdict(int)
        try:
            self.state = json.loads((root / "state.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self.state = {}

    # ----- bookkeeping -----
    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "state.json").write_text(json.dumps(self.state, indent=1))
        self._mtime = (self.root / "state.json").stat().st_mtime_ns
        if self.on_change:
            self.on_change()

    def _sync(self) -> None:
        """The owner may install a model from the command line while the server runs: pick that up."""
        try:
            m = (self.root / "state.json").stat().st_mtime_ns
        except FileNotFoundError:
            return
        if m != getattr(self, "_mtime", None):
            self._mtime = m
            try:
                disk = json.loads((self.root / "state.json").read_text())
            except (json.JSONDecodeError, OSError):
                return
            for mid, st in disk.items():
                mine = self.state.get(mid, {})
                if mine.get("status") in ("running", "starting", "downloading") and mid in self.procs | self.jobs:
                    continue  # this server is busy with it: its own state wins
                self.state[mid] = st

    def _sources(self) -> dict:
        try:
            return json.loads((self.root / "sources.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def set_source(self, model_id: str, link: str) -> None:
        """Owner only (the API checks): the model's Drive folder. Kept in a private file, never in git."""
        self._entry(model_id)
        fid = folder_id(link)
        s = self._sources()
        s[model_id] = fid
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / "sources.tmp"
        tmp.write_text(json.dumps(s, indent=1))
        os.chmod(tmp, 0o600)
        tmp.replace(self.root / "sources.json")

    def set_command(self, model_id: str, command: str, port: int) -> None:
        """Owner only: how to start a model that isn't a GGUF (its own code), and the OpenAI-compatible port it serves."""
        self._entry(model_id)
        st = self.state.setdefault(model_id, {})
        st["command"], st["port"] = command.strip()[:1000], int(port)
        self._save()

    def _entry(self, model_id: str) -> dict:
        e = next((m for m in CATALOG if m["id"] == model_id), None)
        if not e:
            raise ModelError("No such model")
        return e

    def dir(self, model_id: str) -> Path:
        return self.root / model_id

    def engine(self) -> str:
        from .expert import es_serve_path
        want = os.getenv("MNX_ENGINE", "auto").strip().lower()
        return "Expert Stream" if es_serve_path() and want != "llama" else "llama.cpp"

    def view(self, owner: bool = False) -> list[dict]:
        from .brain import brain
        self._sync()
        out = []
        for m in CATALOG:
            st = self.state.get(m["id"], {})
            running = m["id"] in self.procs and self.procs[m["id"]].returncode is None
            v = {k: m[k] for k in ("id", "name", "about", "kind", "run", "ram_gb", "disk_gb")}
            v.update(status=("downloading" if m["id"] in self.jobs and not self.jobs[m["id"]].done() else
                             "running" if running else st.get("status") or "not downloaded"),
                     progress=self.progress.get(m["id"]), error=st.get("error"),
                     in_use=bool(brain.url) and st.get("url") == brain.url and running,
                     can_run=m["kind"] == "gguf" or bool(st.get("command")))
            if owner:
                v.update(has_link=m["id"] in self._sources(), command=st.get("command", ""), port=st.get("port"),
                         path=st.get("path"))
            out.append(v)
        return out

    # ----- picking a model: what users see and use -----
    def usable(self, model_id: str) -> bool:
        """The owner has set it up: users can pick it and it starts by itself when they do."""
        self._sync()
        e = next((m for m in CATALOG if m["id"] == model_id), None)
        st = self.state.get(model_id, {})
        if not e or st.get("status") not in ("ready", "running", "starting", "stopped"):
            return False
        return e["kind"] == "gguf" or bool(st.get("command"))

    def choices(self) -> list[dict]:
        """What a user's model picker shows: names only."""
        return [{"id": m["id"], "name": m["name"]} for m in CATALOG if self.usable(m["id"])]

    def default(self) -> str | None:
        """The model used when none is picked: the one running now, else the first set up."""
        for mid in self.procs:
            if self.procs[mid].returncode is None and self.usable(mid):
                return mid
        return next((m["id"] for m in CATALOG if self.usable(m["id"])), None)

    @asynccontextmanager
    async def using(self, model_id: str):
        """Yields the model's server URL for one answer, starting it (and stopping another) if needed. A model that
        is answering someone isn't stopped for another until it finishes."""
        if not self.usable(model_id):
            raise ModelError("That model isn't available")
        async with self._lock:
            running = model_id in self.procs and self.procs[model_id].returncode is None
            if not running:
                for _ in range(240):  # let answers in progress on another model finish first
                    if not any(n for mid, n in self.inflight.items() if mid != model_id):
                        break
                    await asyncio.sleep(0.5)
                await self.start(model_id, use=False)
            url = self.state[model_id]["url"]
            self.inflight[model_id] += 1
        try:
            yield url
        finally:
            self.inflight[model_id] -= 1

    # ----- downloading -----
    def _list(self, fid: str) -> list[tuple[str, str]]:
        if self.lister:
            return self.lister(fid)
        try:
            import gdown
        except ImportError as exc:
            raise ModelError("The Hive needs gdown to download from Drive: pip install gdown") from exc
        try:
            items = gdown.download_folder(id=fid, skip_download=True, quiet=True)
        except Exception as exc:
            raise ModelError("Google Drive didn't list the folder. Share it as “Anyone with the link” (Viewer) and try "
                             f"again ({str(exc)[:160]})") from exc
        return [(it.id, it.path) for it in items or []]

    def _fetch(self, file_id: str, dest: Path, progress) -> None:
        if self.fetcher:
            return self.fetcher(file_id, dest, progress)
        import gdown
        dest.parent.mkdir(parents=True, exist_ok=True)
        got = gdown.download(id=file_id, output=str(dest), quiet=self.quiet, resume=True, progress=progress, retries=3)
        if not got:
            raise ModelError(f"Drive didn't send {dest.name} (is the folder shared “Anyone with the link”?)")

    def download(self, model_id: str) -> None:
        e = self._entry(model_id)
        if model_id in self.jobs and not self.jobs[model_id].done():
            raise ModelError("Already downloading")
        fid = self._sources().get(model_id)
        if not fid:
            raise ModelError("Paste the model's Google Drive folder link first")
        free = shutil.disk_usage(self.root.parent if self.root.parent.exists() else "/").free / 1e9
        need = e["disk_gb"] * 1.15
        if free < need:
            raise ModelError(f"{e['name']} needs about {need:.1f} GB free; this machine has {free:.1f} GB")
        self.jobs[model_id] = asyncio.get_running_loop().create_task(self._download(model_id, fid))

    async def _download(self, model_id: str, fid: str) -> None:
        loop = asyncio.get_running_loop()
        await asyncio.to_thread(self.fetch_all, model_id, fid, lambda: loop.call_soon_threadsafe(self._changed))

    def fetch_all(self, model_id: str, fid: str, tick=None, say=None) -> bool:
        """The whole job, synchronous: list the Drive folder, fetch every part (skipping parts already fetched), join,
        check the MD5s, tidy up. Runs on the server in a thread, or straight from the command line. True when ready."""
        e = self._entry(model_id)
        st = self.state.setdefault(model_id, {})
        st.update(status="downloading", error=None)
        self._save()
        tick = tick or (lambda: None)
        say = say or (lambda msg: None)
        work = self.dir(model_id) / "download"
        out = self.dir(model_id) / "files"
        ok = False
        try:
            listing = self._list(fid)
            if not listing:
                raise ModelError("The Drive folder is empty")
            by_rel = {rel: fid_ for fid_, rel in listing}
            total = len(by_rel)
            marks = self.dir(model_id) / "fetched.json"  # parts already complete: a re-run skips them
            try:
                fetched = set(json.loads(marks.read_text()))
            except (FileNotFoundError, json.JSONDecodeError):
                fetched = set()
            t0 = time.time()
            for i, (rel, file_id) in enumerate(sorted(by_rel.items())):
                dest = work / rel
                self.progress[model_id] = {"file": Path(rel).name, "files_done": i, "files": total, "stage": "downloading",
                                           "since": t0}
                tick()
                if rel in fetched and dest.exists():
                    continue
                say(f"  [{i + 1}/{total}] {Path(rel).name}")
                self._fetch(file_id, dest, None)
                fetched.add(rel)
                marks.write_text(json.dumps(sorted(fetched)))
            self.progress[model_id] = {"stage": "joining", "files_done": total, "files": total}
            tick()
            say("  joining the parts…")
            joined = self._join(work, out)
            self.progress[model_id] = {"stage": "checking", "files_done": total, "files": total}
            tick()
            say("  checking MD5s…")
            expected = {**read_md5_lists(work), **read_md5_lists(out), **e["md5"]}
            bad = self._verify(out, joined, expected, e.get("sizes") or {})
            if bad:
                raise ModelError("Checksum mismatch, download again: " + ", ".join(bad[:5]))
            shutil.rmtree(work, ignore_errors=True)  # the parts aren't needed once the files check out
            (self.dir(model_id) / "fetched.json").unlink(missing_ok=True)
            main = next(out.rglob(e["main"]), None)
            st.update(status="ready", path=str(main or out), verified=len(joined), ready_at=int(time.time()))
            ok = True
        except Exception as exc:
            st.update(status="failed", error=str(exc)[:400])
            say(f"  failed: {exc}")
        finally:
            self.progress.pop(model_id, None)
            self._save()
            tick()
        return ok

    def _changed(self) -> None:
        if self.on_change:
            self.on_change()

    @staticmethod
    def _join(work: Path, out: Path) -> list[Path]:
        """Join name.partNN files in order (any sub-folder), copy whole files; returns the joined files."""
        out.mkdir(parents=True, exist_ok=True)
        files = [p for p in work.rglob("*") if p.is_file()]
        by_dir: dict[Path, list[str]] = defaultdict(list)
        for p in files:
            by_dir[p.parent].append(p.name)
        joined = []
        for d, names in by_dir.items():
            joins, whole = plan_joins(names)
            rel_dir = d.relative_to(work)
            if rel_dir.name.endswith("_parts"):  # table.bin_parts/table.bin.part000 → table.bin next to the others
                rel_dir = rel_dir.parent
            target = out / rel_dir
            target.mkdir(parents=True, exist_ok=True)
            for base, parts in joins.items():
                dest = target / base
                with open(dest, "wb") as w:
                    for part in parts:
                        with open(d / part, "rb") as r:
                            shutil.copyfileobj(r, w, 16 << 20)
                joined.append(dest)
            for name in whole:
                shutil.copy2(d / name, target / name)
        return joined

    @staticmethod
    def _verify(out: Path, joined: list[Path], expected: dict[str, str], sizes: dict[str, int]) -> list[str]:
        bad = []
        for p in joined:
            if p.name in sizes and p.stat().st_size != sizes[p.name]:
                bad.append(f"{p.name} (size)")
                continue
            if p.name in expected and md5_of(p) != expected[p.name]:
                bad.append(p.name)
        return bad

    # ----- running -----
    def _port_for(self, model_id: str) -> int:
        """Each model has its own port (PORT + its place in the catalog), so a model never collides with another."""
        return PORT + next((i for i, m in enumerate(CATALOG) if m["id"] == model_id), 0)

    def _server_argv(self, path: str, port: int | None = None) -> list[str]:
        CTX = os.getenv("MNX_CTX", "16384")  # tokens the model can hold at once: a whole game or site needs more than 4096
        port = port or PORT
        # Expert Stream (hive/expert.py) when it's built: the same OpenAI API, with prefix reuse, a prompt cache on disk
        # and Turbo decoding. MNX_ENGINE=llama keeps llama.cpp; MNX_ENGINE=expertstream insists on Expert Stream.
        want = os.getenv("MNX_ENGINE", "auto").strip().lower()
        from .expert import es_serve_path
        es = es_serve_path() if want in ("auto", "expertstream", "expert-stream", "es") else None
        if es:
            return [es, "-g", path, "-p", str(port), "-c", CTX]
        if want in ("expertstream", "expert-stream", "es"):
            raise ModelError("Expert Stream isn't built yet: python3 -m hive.admin install-expertstream")
        exe = shutil.which("llama-server") or next((str(p) for p in [Path.home() / "llama.cpp/build/bin/llama-server"] if p.exists()), None)
        threads = threads_for()
        if exe:
            return [exe, "-m", path, "--host", "127.0.0.1", "--port", str(port), "-c", CTX, "-t", threads]
        if shutil.which("docker"):
            from .lab import LLAMA_IMAGE
            d = str(Path(path).parent)
            return ["docker", "run", "--rm", "--name", f"mnx-model-{port}", "-p", f"127.0.0.1:{port}:8080", "-v", f"{d}:/models:ro",
                    LLAMA_IMAGE, "-m", f"/models/{Path(path).name}", "--host", "0.0.0.0", "--port", "8080", "-c", CTX]
        raise ModelError("llama.cpp isn't installed. On the phone: pkg install llama-cpp · on a server: install llama.cpp "
                         "(llama-server) or Docker")

    async def start(self, model_id: str, use: bool = True, wait: float = 180) -> dict:
        e = self._entry(model_id)
        st = self.state.get(model_id, {})
        if st.get("status") not in ("ready", "stopped", "running") and e["kind"] == "gguf":
            raise ModelError("Download it first")
        await self.stop_all()
        if e["kind"] == "gguf":
            port = self._port_for(model_id)
            argv, cwd, env = self._server_argv(st["path"], port), None, None
        elif st.get("command"):
            argv, port = ["sh", "-c", st["command"]], int(st.get("port") or 8000)
            cwd = str(self.dir(model_id) / "files") if (self.dir(model_id) / "files").exists() else None
            env = {**os.environ, "MNX_MODEL_DIR": cwd or ""}
        else:
            raise ModelError(f"{e['name']} runs with {e['run']}: set its start command and port first")
        log = open(self.dir(model_id) / "server.log", "ab") if self.dir(model_id).exists() else None
        if log is None:
            self.dir(model_id).mkdir(parents=True, exist_ok=True)
            log = open(self.dir(model_id) / "server.log", "ab")
        proc = await asyncio.create_subprocess_exec(*argv, cwd=cwd, env=env, stdout=log, stderr=asyncio.subprocess.STDOUT,
                                                    start_new_session=True)
        self.procs[model_id] = proc
        url = f"http://127.0.0.1:{port}"
        st = self.state.setdefault(model_id, {})
        st.update(url=url, status="starting", error=None)
        self._save()
        t0 = time.time()
        async with httpx.AsyncClient(timeout=5) as c:
            while time.time() - t0 < wait:
                if proc.returncode is not None:
                    tail = (self.dir(model_id) / "server.log").read_text(errors="replace")[-400:]
                    st.update(status="failed", error=f"It stopped while starting: {tail}")
                    self._save()
                    raise ModelError(st["error"])
                try:
                    r = await c.get(url + "/health")
                    if r.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
            else:
                st.update(status="failed", error="It didn't answer in time; see server.log")
                self._save()
                raise ModelError(st["error"])
        st["status"] = "running"
        if use:
            from .brain import brain
            brain.update({"url": url, "model": model_id})
        self._save()
        return st

    async def stop(self, model_id: str) -> None:
        proc = self.procs.pop(model_id, None)
        if proc and proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), 15)
            except asyncio.TimeoutError:
                os.killpg(proc.pid, signal.SIGKILL)
        st = self.state.get(model_id)
        if st and st.get("status") in ("running", "starting"):
            st["status"] = "ready" if st.get("path") else "stopped"
            from .brain import brain
            if brain.url and brain.url == st.get("url"):
                brain.update({"url": "-"})
            self._save()

    async def stop_all(self) -> None:
        for mid in list(self.procs):
            await self.stop(mid)

    def delete(self, model_id: str) -> None:
        self._entry(model_id)
        if model_id in self.procs:
            raise ModelError("Stop it first")
        shutil.rmtree(self.dir(model_id), ignore_errors=True)
        self.state.pop(model_id, None)
        self._save()


models = Models()
