"""The Hive's GPU pool: the GPU a Plus Bee gets, given by the Hive (you don't bring one).

Where the GPU comes from, best first:
  this machine   when Docker here has an NVIDIA GPU, a Plus Bee's own computer gets it directly (cells.py)
  Kaggle         a free NVIDIA GPU with 16 GB VRAM (T4 or P100), about 30 GPU hours a week, on the Hive owner's
                 Kaggle account (Free cloud page: Kaggle user name + API token, saved as Hive-only secrets).
                 A job is a script run with the Bee's files; its output files come back into the Bee's
                 computer, in gpu-out/<job>/.

From a Bee's computer:   python3 .mnx/mnx-bee gpu train.py data.csv      (runs "python train.py" on the GPU)
                         python3 .mnx/mnx-bee gpu --cmd "nvidia-smi && python a.py" a.py
In a Bee's own chat:     "run train.py on the gpu"  ·  "gpu status"
Kaggle's own rules apply (no crypto mining, ~30 GPU hours a week, a session runs at most about 12 hours).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import shlex
import shutil
import tarfile
import time
import uuid
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
GPU_DIR = DATA / "gpu"
WEEKLY_HOURS = float(os.getenv("MNX_GPU_WEEKLY_HOURS", "30"))  # Kaggle's free weekly GPU quota
MAX_INPUT = 20 * 1024 * 1024  # the Bee's files travel inside the job script
POLL = 30.0
VRAM_GB = 16
KAGGLE_BIN: str | None = None  # tests point this at a stand-in
DONE = ("complete", "error", "failed", "cancelled")


class GpuError(RuntimeError):
    pass


def runner_script(command: str, files: dict[str, bytes]) -> str:
    """The job as one Python script: unpacks the Bee's files, shows the GPU, runs the command."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), int(time.time()), 0o644
            tar.addfile(info, io.BytesIO(data))
    blob = base64.b64encode(buf.getvalue()).decode()
    return f'''# Mnx Hive GPU job: the Bee's files, then its command, on this GPU.
import base64, io, os, subprocess, sys, tarfile
os.makedirs("/kaggle/working", exist_ok=True)
os.chdir("/kaggle/working")
with tarfile.open(fileobj=io.BytesIO(base64.b64decode("{blob}")), mode="r:gz") as t:
    try:
        t.extractall(".", filter="data")
    except TypeError:
        t.extractall(".")
subprocess.call("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True)
rc = subprocess.call({json.dumps(command)}, shell=True)
print("MNX_EXIT", rc, flush=True)
sys.exit(0)
'''


def _slug(bee_id: str, job_id: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", f"mnx-{bee_id}-{job_id}".lower())[:50].strip("-")


class GpuPool:
    def __init__(self, root: Path = GPU_DIR):
        self.root = root
        try:
            self.jobs: dict[str, dict] = json.loads((root / "jobs.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self.jobs = {}
        self.poll = POLL
        self.on_change = None
        self.deliver = None  # set by the gateway: async (bee_id, rel path, bytes) → write into the Bee's computer
        self._watch: dict[str, asyncio.Task] = {}

    # ----- setup -----
    def _secrets(self) -> dict:
        from .lab import lab
        return lab._secrets

    def kaggle_ready(self) -> bool:
        s = self._secrets()
        return bool(s.get("KAGGLE_USERNAME") and (s.get("KAGGLE_API_TOKEN") or s.get("KAGGLE_KEY")))

    def local_ready(self) -> bool:
        from .lab import lab
        return bool(lab.gpu_local)

    def source(self) -> str | None:
        return "local" if self.local_ready() else "kaggle" if self.kaggle_ready() else None

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / "jobs.tmp"
        tmp.write_text(json.dumps(self.jobs, indent=2))
        tmp.replace(self.root / "jobs.json")
        if self.on_change:
            self.on_change()

    def hours_this_week(self, now: float | None = None) -> float:
        now = now or time.time()
        total = 0.0
        for j in self.jobs.values():
            start = j.get("started")
            if j.get("source") != "kaggle" or not start:
                continue
            end = j.get("ended") or now
            if end >= now - 7 * 86400:
                total += max(0.0, end - max(start, now - 7 * 86400)) / 3600
        return round(total, 2)

    def view(self) -> dict:
        return {"source": self.source(), "kaggle": self.kaggle_ready(), "local": self.local_ready(), "vram_gb": VRAM_GB,
                "hours_week": self.hours_this_week(), "weekly_limit": WEEKLY_HOURS,
                "cli": bool(self._bin()), "jobs": sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)[:30]}

    # ----- the Kaggle CLI -----
    def _bin(self) -> list[str] | None:
        if KAGGLE_BIN:
            return [KAGGLE_BIN]
        b = shutil.which("kaggle")
        return [b] if b else None

    async def _kaggle(self, *args: str, timeout: float = 300) -> tuple[int, str]:
        cli = self._bin()
        if not cli:
            raise GpuError("The Kaggle tool isn't installed on the Hive: run  pip install kaggle")
        s = self._secrets()
        home = self.root / "home"
        home.mkdir(parents=True, exist_ok=True)
        os.chmod(home, 0o700)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home), "KAGGLE_CONFIG_DIR": str(home),
               "KAGGLE_USERNAME": s.get("KAGGLE_USERNAME", ""), "PYTHONIOENCODING": "utf-8"}
        if s.get("KAGGLE_API_TOKEN"):
            env["KAGGLE_API_TOKEN"] = s["KAGGLE_API_TOKEN"]
        if s.get("KAGGLE_KEY"):
            env["KAGGLE_KEY"] = s["KAGGLE_KEY"]
        proc = await asyncio.create_subprocess_exec(*cli, *args, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return 124, "timed out"
        return proc.returncode or 0, out.decode(errors="replace")

    async def check(self) -> dict:
        """Is the Kaggle account usable? Lists your kernels (needs a working token)."""
        code, out = await self._kaggle("kernels", "list", "--mine", "--page-size", "1", timeout=60)
        if code != 0 or "401" in out or "Unauthorized" in out or "Authentication required" in out:
            raise GpuError("Kaggle didn't accept the user name / API token: " + out.strip()[-200:])
        return {"ok": True}

    # ----- jobs -----
    async def submit(self, bee_id: str, command: str, files: dict[str, bytes]) -> dict:
        from .plan import plan
        if not plan.allows_gpu():
            raise GpuError("The GPU comes with Plus, which is free: subscribe on the Usage page or say “subscribe me to plus”")
        if not self.kaggle_ready():
            raise GpuError("The Hive's GPU pool isn't set up yet: open the Free cloud page and add the Kaggle account (free)")
        if self.hours_this_week() >= WEEKLY_HOURS:
            raise GpuError(f"The free GPU's {WEEKLY_HOURS:g} hours this week are used up; it frees up as the week rolls on")
        command = (command or "").strip()
        if not command:
            raise GpuError("Nothing to run")
        size = sum(len(v) for v in files.values())
        if size > MAX_INPUT:
            raise GpuError(f"The files are {size / 1e6:.0f} MB; a GPU job takes up to {MAX_INPUT / 1e6:.0f} MB "
                           "(download big data inside the script instead)")
        for name in files:
            if name.startswith("/") or ".." in Path(name).parts:
                raise GpuError(f"Bad file name {name!r}")
        jid = uuid.uuid4().hex[:8]
        user = self._secrets()["KAGGLE_USERNAME"]
        slug = _slug(bee_id, jid)
        job = {"id": jid, "bee": bee_id, "command": command[:500], "files": sorted(files), "source": "kaggle",
               "kernel": f"{user}/{slug}", "gpu": f"Kaggle NVIDIA GPU · {VRAM_GB} GB VRAM", "status": "queued",
               "created": int(time.time()), "started": None, "ended": None, "exit": None, "outputs": [], "log": "",
               "error": None}
        d = self.root / jid
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.py").write_text(runner_script(command, files))
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": job["kernel"], "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}))
        self.jobs[jid] = job
        self._save()
        code, out = await self._kaggle("kernels", "push", "-p", str(d))
        if code != 0 or re.search(r"error|invalid|unauthori[sz]ed|not found", out, re.I) and "successfully" not in out.lower():
            job.update(status="failed", error=("Kaggle didn't take the job: " + out.strip()[-300:]), ended=int(time.time()))
            self._save()
            raise GpuError(job["error"])
        job.update(status="running", started=int(time.time()))
        self._save()
        self.watch(jid)
        return job

    def watch(self, jid: str) -> None:
        if jid not in self._watch or self._watch[jid].done():
            self._watch[jid] = asyncio.get_running_loop().create_task(self._follow(jid))

    def resume(self) -> None:
        for jid, j in self.jobs.items():
            if j["status"] in ("queued", "running"):
                self.watch(jid)

    async def _follow(self, jid: str) -> None:
        job = self.jobs[jid]
        while job["status"] == "running":
            await asyncio.sleep(self.poll)
            try:
                code, out = await self._kaggle("kernels", "status", job["kernel"], timeout=60)
            except GpuError as exc:
                job.update(status="failed", error=str(exc), ended=int(time.time()))
                break
            m = re.search(r'has status "([^"]+)"', out)
            st = (m.group(1) if m else out).lower()
            if "complete" in st:
                job["status"] = "complete"
            elif "error" in st or "fail" in st:
                job["status"] = "error"
                fm = re.search(r'Failure message: "([^"]*)"', out)
                job["error"] = (fm.group(1) if fm else "the job failed on Kaggle")[:300]
            elif "cancel" in st:
                job["status"] = "cancelled"
        job["ended"] = job.get("ended") or int(time.time())
        await self._collect(job)
        self._save()

    async def _collect(self, job: dict) -> None:
        out_dir = self.root / job["id"] / "out"
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True)
        try:
            await self._kaggle("kernels", "output", job["kernel"], "-p", str(out_dir), "-o", "-q", timeout=900)
        except GpuError as exc:
            job["error"] = job.get("error") or str(exc)
            return
        files = [p for p in sorted(out_dir.rglob("*")) if p.is_file()]
        log = next((p for p in files if p.suffix == ".log"), None)
        if log:
            text = log.read_text(errors="replace")
            try:  # Kaggle's log is JSON lines of {"data": "..."}; fall back to the raw text
                text = "".join(json.loads(x).get("data", "") for x in json.loads(text)) if text.lstrip().startswith("[") else text
            except (ValueError, AttributeError, TypeError):
                pass
            job["log"] = text[-6000:]
            m = re.search(r"MNX_EXIT (-?\d+)", text)
            if m:
                job["exit"] = int(m.group(1))
                if job["exit"] != 0 and job["status"] == "complete":
                    job["status"] = "error"
                    job["error"] = f"the command exited with {job['exit']}"
        job["outputs"] = [str(p.relative_to(out_dir)) for p in files if p is not log]
        if self.deliver:
            for p in files:
                try:
                    await self.deliver(job["bee"], f"gpu-out/{job['id']}/{p.relative_to(out_dir)}", p.read_bytes())
                except Exception as exc:  # the Bee's computer is away; the files stay in the Hive
                    job["error"] = job.get("error") or f"outputs kept in the Hive: {exc}"[:300]
                    break

    def output_path(self, jid: str, name: str) -> Path:
        base = (self.root / jid / "out").resolve()
        p = (base / name).resolve()
        if base not in p.parents or not p.is_file():
            raise FileNotFoundError(name)
        return p


pool = GpuPool()


# ---------------------------------------------------------------- the chat: "run train.py on the gpu"
GPU_RUN = re.compile(r"^\s*(?:run|train|execute)\s+(?P<what>.+?)\s+on\s+(?:the\s+|a\s+|my\s+)?(?:free\s+)?gpu\s*[.!]?\s*$", re.I)
GPU_STATUS = re.compile(r"\bgpu\s+(?:status|jobs?)\b|\bhow(?:'s| is)\s+(?:my\s+)?gpu\s+job\b|\bgpu\s+pool\b", re.I)


def parse(text: str) -> dict | None:
    m = GPU_RUN.match(text or "")
    if m:
        what = m.group("what").strip().strip("`")
        words = shlex.split(what) if what else []
        first = words[0] if words else ""
        if first.endswith(".py"):
            return {"kind": "run", "command": "python " + what, "files": [w for w in words if "." in w and not w.startswith("-")]}
        return {"kind": "run", "command": what, "files": [w for w in words[1:] if "." in w and not w.startswith("-")]}
    if GPU_STATUS.search(text or ""):
        return {"kind": "status"}
    return None
