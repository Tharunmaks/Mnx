"""Server health: CPU, memory, disk, GPU and containers, for the Lab and Hive screens."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
_gpu_cache: tuple[float, list] = (0.0, [])
_cpu_prev: tuple[int, int] | None = None


def _cpu_percent() -> float | None:
    """CPU busy % since the previous call (from /proc/stat)."""
    global _cpu_prev
    try:
        vals = list(map(int, open("/proc/stat").readline().split()[1:]))
    except (OSError, ValueError):
        return None
    idle, total = vals[3] + (vals[4] if len(vals) > 4 else 0), sum(vals)
    prev, _cpu_prev = _cpu_prev, (idle, total)
    if not prev or total == prev[1]:
        return None
    return round(100 * (1 - (idle - prev[0]) / (total - prev[1])), 1)


def _meminfo() -> dict[str, int]:
    out = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            out[k] = int(v.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return out


def gpus() -> list[dict]:
    global _gpu_cache
    if time.time() - _gpu_cache[0] < 30:
        return _gpu_cache[1]
    found = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout
            for line in out.strip().splitlines():
                name, total, used, util = [x.strip() for x in line.split(",")]
                found.append({"name": name, "memory_mb": int(total), "used_mb": int(used), "util": int(util)})
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    _gpu_cache = (time.time(), found)
    return found


def stats() -> dict:
    mem = _meminfo()
    disk = shutil.disk_usage(DATA if DATA.exists() else "/")
    try:
        uptime = float(open("/proc/uptime").read().split()[0])
    except (OSError, ValueError):
        uptime = None
    return {
        "cpus": os.cpu_count(), "load": [round(x, 2) for x in os.getloadavg()], "cpu_percent": _cpu_percent(),
        "memory": {"total": mem.get("MemTotal"), "available": mem.get("MemAvailable")},
        "disk": {"total": disk.total, "free": disk.free}, "uptime": uptime, "gpus": gpus(),
    }
