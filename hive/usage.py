"""Usage: what the Hive has actually used, next to the limits it runs under.

Everything here is measured, not guessed:
  - tokens per message, from the Hive's token reading (the regex counter; about 10-15% off a real tokenizer),
    saved per day in data/usage.json so it survives restarts
  - CPU and memory of each Bee's container, from `docker stats` (only when the Cell runs in Docker)
  - disk, training time, models, and GPU rentals (hours and cost from the real start time and price)
Where a number can't be measured (a Bee's CPU without Docker, say) it is None, and the page says so.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
LEDGER_FILE = DATA / "usage.json"
KEEP_DAYS = 120
KEEP_RENTALS = 200
UNITS = {"b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12, "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3, "tib": 1024 ** 4}


def day_key(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts if ts is not None else time.time()))


class Ledger:
    """Per-day token totals and the history of rented GPU machines, saved to one small file."""

    def __init__(self, path: Path = LEDGER_FILE):
        self.path = path
        try:
            saved = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            saved = {}
        self.days: dict[str, dict] = saved.get("days") or {}
        self.rentals: list[dict] = saved.get("rentals") or []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"days": self.days, "rentals": self.rentals[-KEEP_RENTALS:]}))
        tmp.replace(self.path)

    def record_task(self, usage: dict, when: float | None = None) -> None:
        """One finished (or stopped) message: {"in", "out", "total", "budget", "effort"}."""
        total = int(usage.get("total") or 0)
        if total <= 0:
            return
        d = self.days.setdefault(day_key(when), {"messages": 0, "in": 0, "out": 0, "peak": 0, "over_budget": 0, "efforts": {}})
        d["messages"] += 1
        d["in"] += int(usage.get("in") or 0)
        d["out"] += int(usage.get("out") or 0)
        d["peak"] = max(d["peak"], total)
        budget = int(usage.get("budget") or 0)
        if budget and total > budget:
            d["over_budget"] += 1
        e = d["efforts"].setdefault(str(usage.get("effort") or "med"), {"messages": 0, "tokens": 0, "peak": 0, "budget": budget})
        e["messages"] += 1
        e["tokens"] += total
        e["peak"] = max(e["peak"], total)
        e["budget"] = budget or e["budget"]
        for old in sorted(self.days)[:-KEEP_DAYS]:
            del self.days[old]
        self.save()

    def record_rental(self, rec: dict, ended: float | None = None) -> dict:
        """A rented GPU machine that was just stopped: how long it ran and what that cost."""
        ended = ended if ended is not None else time.time()
        out = rental_now(rec, ended)
        out.update(id=rec.get("id"), ended=int(ended))
        self.rentals.append(out)
        self.save()
        return out


def token_summary(ledger: Ledger, today: str | None = None, days: int = 14) -> dict:
    today = today or day_key()
    t0 = time.mktime(time.strptime(today, "%Y-%m-%d"))
    series = []
    for i in range(days - 1, -1, -1):
        k = day_key(t0 - i * 86400 + 43200)  # noon, so daylight-saving shifts never skip or repeat a day
        d = ledger.days.get(k) or {}
        series.append({"day": k, "tokens": int(d.get("in", 0)) + int(d.get("out", 0)), "in": int(d.get("in", 0)), "out": int(d.get("out", 0)),
                       "messages": int(d.get("messages", 0))})
    last7 = series[-7:]
    all_days = list(ledger.days.values())
    todays = ledger.days.get(today) or {}
    return {
        "today": {"tokens": int(todays.get("in", 0)) + int(todays.get("out", 0)), "in": int(todays.get("in", 0)), "out": int(todays.get("out", 0)),
                  "messages": int(todays.get("messages", 0)), "peak": int(todays.get("peak", 0)), "over_budget": int(todays.get("over_budget", 0))},
        "week": {"tokens": sum(x["tokens"] for x in last7), "messages": sum(x["messages"] for x in last7)},
        "all": {"tokens": sum(int(d.get("in", 0)) + int(d.get("out", 0)) for d in all_days), "messages": sum(int(d.get("messages", 0)) for d in all_days),
                "since": min(ledger.days) if ledger.days else None, "over_budget": sum(int(d.get("over_budget", 0)) for d in all_days)},
        "series": series,
        "efforts_today": todays.get("efforts") or {},
        "estimate": "Counted by the Hive's own token reading, within about 10-15% of a real tokenizer.",
    }


def rental_now(rec: dict, now: float | None = None) -> dict:
    """Hours so far and cost so far of one rented machine (cost is None when RunPod's price wasn't recorded)."""
    now = now if now is not None else time.time()
    started = float(rec.get("started") or now)
    hours = max(0.0, (now - started) / 3600)
    per_hour = rec.get("cost_per_hour")
    return {"gpu": rec.get("gpu_type") or rec.get("gpu"), "count": int(rec.get("count") or 1), "started": int(started),
            "hours": round(hours, 3), "cost_per_hour": per_hour, "cost": round(hours * float(per_hour), 4) if per_hour else None}


def parse_size(text: str) -> int | None:
    """'12.3MiB' → bytes."""
    m = re.fullmatch(r"\s*([\d.]+)\s*([A-Za-z]+)\s*", text or "")
    if not m or m.group(2).lower() not in UNITS:
        return None
    return int(float(m.group(1)) * UNITS[m.group(2).lower()])


def parse_docker_stats(text: str, prefix: str = "mnx-cell-") -> dict[str, dict]:
    """`docker stats --no-stream --format '{{.Name}}\\t{{.CPUPerc}}\\t{{.MemUsage}}'` → {container: {cpu, mem, mem_limit}}."""
    out = {}
    for line in (text or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or not parts[0].startswith(prefix):
            continue
        used, _, limit = parts[2].partition("/")
        try:
            cpu = float(parts[1].strip().rstrip("%"))
        except ValueError:
            cpu = None
        out[parts[0]] = {"cpu": cpu, "mem": parse_size(used), "mem_limit": parse_size(limit)}
    return out


def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def run_seconds(run: dict, now: float | None = None) -> float:
    """How long a training run has actually run: start to end, or start to now while it runs."""
    started = run.get("started")
    if not started:
        return 0.0
    ended = run.get("ended") or (now if now is not None else time.time())
    return max(0.0, float(ended) - float(started))


ledger = Ledger()
