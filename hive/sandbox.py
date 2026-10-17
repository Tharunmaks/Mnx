"""The Sandbox: everything your Bees made (games, websites, 3D models), each on its private Chroma link, listed at
/sandbox.html and run there in a locked-down frame (Chroma serves it with a CSP sandbox: its own opaque origin, so it can
never read the Hive's pages, your token or other sites)."""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "sandbox.json"
KEEP = 200
KINDS = ("game", "site", "model")


def _read() -> dict:
    try:
        return json.loads(FILE.read_text())
    except (OSError, ValueError):
        return {}


def record(uid: str, kind: str, name: str, bee: str, url: str, **extra) -> None:
    if not url or kind not in KINDS:
        return
    data = _read()
    rows = [r for r in data.get(uid, []) if not (r["kind"] == kind and r["name"] == name)]  # the newest version of each
    rows.insert(0, {"kind": kind, "name": name, "bee": bee, "url": url, "when": int(time.time()), **extra})
    data[uid] = rows[:KEEP]
    FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(FILE)


def of(uid: str) -> list[dict]:
    return _read().get(uid, [])
