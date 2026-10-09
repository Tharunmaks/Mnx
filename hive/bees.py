"""The Bee registry: the built-in Bees, the Bees you create, and the Bees that Bees create.

Every Bee has
  - a card: name, one-line skill, tools, schedule, whether it asks before acting
  - a family: who made it ("you" or another Bee); deleting a Bee hands its children to its own parent
  - its own computer (a Cell): a size (CPUs, memory, GPU) and where it runs: this server, or any cloud server
    the Lab can reach over SSH (your own, or a rented GPU machine). An "os" slot is kept for your own
    system image; until it is set, Cells use the default image.
  - a key that lets the Bee, from inside its own computer, create and list its children and nothing else

Limits keep a Bee from filling the Hive: MAX_BEES in total, MAX_DEPTH generations, MAX_CHILDREN per Bee.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from pathlib import Path
from typing import Awaitable, Callable

DATA = Path(__file__).resolve().parent.parent / "data"
STATE_FILE = DATA / "hive.json"
KEYS_FILE = DATA / "bee_keys.json"

MAX_BEES = 64
MAX_DEPTH = 4       # you → Bee → child → grandchild → great-grandchild
MAX_CHILDREN = 12

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

# Computer sizes. "max" sets no limits: the Cell may use the whole machine it runs on.
SIZES = {
    "small": {"cpus": 1, "memory_gb": 1, "label": "Small"},
    "medium": {"cpus": 2, "memory_gb": 4, "label": "Medium"},
    "large": {"cpus": 4, "memory_gb": 16, "label": "Large"},
    "xl": {"cpus": 8, "memory_gb": 32, "label": "XL"},
    "max": {"cpus": None, "memory_gb": None, "label": "Max (the whole machine)"},
}
SIZE_WORDS = {"small": "small", "tiny": "small", "medium": "medium", "normal": "medium", "large": "large", "big": "large",
              "xl": "xl", "huge": "xl", "extra large": "xl", "max": "max", "maximum": "max", "biggest": "max",
              "most powerful": "max", "powerful": "max", "very powerful": "max", "strongest": "max", "full": "max"}
ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")


class BeeError(ValueError):
    """A Bee can't be created or changed that way (a limit, a name clash, a bad computer)."""


def default_computer() -> dict:
    return {"size": "small", "cpus": 1, "memory_gb": 1, "gpu": False, "where": "here", "os": None}


def normalize_computer(c: dict | None, base: dict | None = None) -> dict:
    """A complete computer spec from a partial one (size presets fill CPUs and memory unless given)."""
    out = dict(base or default_computer())
    c = dict(c or {})
    if "size" in c and c["size"]:
        size = str(c["size"]).lower()
        if size not in SIZES:
            raise BeeError(f"Unknown size {c['size']!r}: use one of {', '.join(SIZES)}")
        out["size"] = size
        out["cpus"], out["memory_gb"] = SIZES[size]["cpus"], SIZES[size]["memory_gb"]
    for key, lo, hi in (("cpus", 0.25, 512), ("memory_gb", 0.25, 4096)):
        if c.get(key) is not None:
            v = float(c[key])
            if not lo <= v <= hi:
                raise BeeError(f"{key} must be between {lo} and {hi}")
            out[key] = v
            out["size"] = "custom" if "size" not in c else out["size"]
    if "gpu" in c:
        out["gpu"] = bool(c["gpu"])
    if c.get("where"):
        where = str(c["where"])
        if where != "here" and not re.fullmatch(r"[a-z0-9-]{1,48}", where):
            raise BeeError("Bad server id")
        out["where"] = where
    if "os" in c:
        out["os"] = c["os"] or None
    return out


def describe_computer(c: dict, where_name: str | None = None) -> str:
    c = c or default_computer()
    if c["size"] == "max":
        parts = ["the whole machine"]
    else:
        parts = [f"{c['cpus']:g} CPU{'s' if c['cpus'] != 1 else ''}", f"{c['memory_gb']:g} GB"]
    if c.get("gpu"):
        parts.append("GPU")
    label = SIZES.get(c["size"], {}).get("label", "Custom")
    return f"{label} · {', '.join(parts)} · {where_name or ('this server' if c.get('where', 'here') == 'here' else c['where'])}"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40] or "bee"


class Registry:
    """Bees in memory, saved to data/hive.json; keys in data/bee_keys.json (owner-only file)."""

    def __init__(self, state_file: Path = STATE_FILE, keys_file: Path = KEYS_FILE):
        self.state_file, self.keys_file = state_file, keys_file
        try:
            saved = json.loads(state_file.read_text()).get("bees") or []
        except (FileNotFoundError, json.JSONDecodeError):
            saved = []
        have = {b["id"] for b in saved}
        self.bees: list[dict] = [dict(b) for b in DEFAULT_BEES if b["id"] not in have] + saved
        for b in self.bees:
            b.setdefault("created_by", None if b.get("builtin") else "you")
            b["computer"] = normalize_computer(b.get("computer"))
        try:
            self.keys: dict[str, str] = json.loads(keys_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self.keys = {}
        # set by the gateway: start / stop / move a Bee's Cell, and tell the web app
        self.on_created: Callable[[dict], Awaitable[None]] | None = None
        self.on_deleted: Callable[[dict], Awaitable[None]] | None = None
        self.on_computer: Callable[[dict, dict], Awaitable[None]] | None = None
        self.server_name: Callable[[str], str | None] = lambda where: None

    # ----- persistence -----
    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"bees": self.bees}, indent=2))
        tmp.replace(self.state_file)

    def _save_keys(self) -> None:
        self.keys_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.keys_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.keys))
        os.chmod(tmp, 0o600)
        tmp.replace(self.keys_file)

    # ----- lookups -----
    def get(self, bee_id: str) -> dict | None:
        return next((b for b in self.bees if b["id"] == bee_id), None)

    def find(self, words: str) -> dict | None:
        """A Bee by id or name, ignoring case and a trailing 'bee'."""
        q = re.sub(r"\bbees?\b", "", (words or "").lower()).strip(" .,'\"")
        if not q:
            return None
        for b in self.bees:
            if q == b["name"].lower() or b["id"] in (q, slug(q)):
                return b
        hits = [b for b in self.bees if b["name"].lower().startswith(q)]
        return hits[0] if len(hits) == 1 else None

    def children(self, bee_id: str) -> list[dict]:
        return [b for b in self.bees if b.get("created_by") == bee_id]

    def depth(self, bee_id: str | None) -> int:
        """Generations below you: a Bee you made is 1, its child 2 …; built-in Bees count as 1."""
        d, cur, seen = 0, bee_id, set()
        while cur and cur != "you" and cur not in seen:
            seen.add(cur)
            d += 1
            b = self.get(cur)
            cur = b.get("created_by") if b else None
        return d

    def key_for(self, bee_id: str) -> str:
        if bee_id not in self.keys:
            self.keys[bee_id] = secrets.token_urlsafe(24)
            self._save_keys()
        return self.keys[bee_id]

    def by_key(self, key: str) -> dict | None:
        if not key:
            return None
        for bee_id, k in self.keys.items():
            if secrets.compare_digest(k, key):
                return self.get(bee_id)
        return None

    def view(self, b: dict) -> dict:
        parent = b.get("created_by")
        parent_bee = self.get(parent) if parent and parent != "you" else None
        c = b["computer"]
        return {**{k: v for k, v in b.items() if k != "key"},
                "parent_name": "you" if parent == "you" else parent_bee["name"] if parent_bee else None,
                "children": [x["id"] for x in self.children(b["id"])],
                "computer_text": describe_computer(c, self.server_name(c.get("where", "here")))}

    # ----- changes -----
    async def create(self, name: str, skill: str, created_by: str = "you", tools: list[str] | None = None,
                     schedule: str | None = None, approval: bool = True, computer: dict | None = None) -> dict:
        name = " ".join((name or "").split())[:24]
        skill = " ".join((skill or "").split())[:90]
        if not name or not skill:
            raise BeeError("A Bee needs a name and a one-line skill")
        bee_id = slug(name)
        if not ID_RE.match(bee_id):
            raise BeeError("Use letters and numbers in the name")
        if self.get(bee_id):
            raise BeeError(f"A Bee called {self.get(bee_id)['name']} already exists")
        if len(self.bees) >= MAX_BEES:
            raise BeeError(f"The Hive already has {MAX_BEES} Bees, its limit; remove one first")
        if created_by != "you":
            parent = self.get(created_by)
            if not parent:
                raise BeeError("The parent Bee doesn't exist")
            if self.depth(created_by) + 1 > MAX_DEPTH:
                raise BeeError(f"Bees can only go {MAX_DEPTH} generations deep")
            if len(self.children(created_by)) >= MAX_CHILDREN:
                raise BeeError(f"{parent['name']} already has {MAX_CHILDREN} children, its limit")
            # a child starts small, on the same machine as its parent, unless told otherwise
            base = normalize_computer({"where": parent["computer"].get("where", "here")})
        else:
            base = default_computer()
        record = {"id": bee_id, "name": name, "skill": skill, "tools": list(tools or [])[:20], "schedule": schedule or None,
                  "approval": bool(approval), "builtin": False, "created_by": created_by,
                  "computer": normalize_computer(computer, base)}
        self.bees.append(record)
        self.save()
        self.key_for(bee_id)
        if self.on_created:
            await self.on_created(record)
        return record

    async def delete(self, bee_id: str) -> list[dict]:
        """Remove a Bee and its computer. Its children stay, adopted by its own parent. Returns them."""
        b = self.get(bee_id)
        if not b:
            raise BeeError("No such Bee")
        if b.get("builtin"):
            raise BeeError("Built-in Bees can't be removed")
        adopted = self.children(bee_id)
        for c in adopted:
            c["created_by"] = b.get("created_by") or "you"
        self.bees.remove(b)
        self.save()
        if self.keys.pop(bee_id, None):
            self._save_keys()
        if self.on_deleted:
            await self.on_deleted(b)
        return adopted

    async def set_computer(self, bee_id: str, change: dict) -> dict:
        b = self.get(bee_id)
        if not b:
            raise BeeError("No such Bee")
        old = dict(b["computer"])
        new = normalize_computer(change, old)
        if new == old:
            return b
        b["computer"] = new
        self.save()
        if self.on_computer:
            try:
                await self.on_computer(b, old)
            except Exception:
                b["computer"] = old  # the move failed: keep the Bee where it was
                self.save()
                raise
        return b


# ---------- the helper each Bee gets inside its own computer ----------
HELPER = r'''#!/usr/bin/env python3
"""mnx-bee: this Bee talks to its Hive. It can only create and list its own children.

  python3 .mnx/mnx-bee me
  python3 .mnx/mnx-bee children
  python3 .mnx/mnx-bee new "Name" "one-line skill" [small|medium|large|xl|max]
"""
import json, os, sys, urllib.request

hive, key = os.environ.get("MNX_HIVE"), os.environ.get("MNX_BEE_KEY")
if not hive or not key:
    sys.exit("This computer can't reach the Hive (no MNX_HIVE / MNX_BEE_KEY). On a remote server, set MNX_PUBLIC_URL on the Hive.")

def call(method, path, body=None):
    req = urllib.request.Request(hive.rstrip("/") + "/bee-api" + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"X-Bee-Key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        sys.exit(f"Hive said {e.code}: {e.read().decode()[:300]}")

a = sys.argv[1:]
if not a or a[0] in ("-h", "--help", "help"):
    print(__doc__)
elif a[0] == "me":
    print(json.dumps(call("GET", "/me"), indent=2))
elif a[0] == "children":
    for c in call("GET", "/children"):
        print(f"{c['name']:24} {c['computer_text']}  ·  {c['skill']}")
elif a[0] == "new" and len(a) >= 3:
    c = call("POST", "/children", {"name": a[1], "skill": a[2], "size": a[3] if len(a) > 3 else None})
    print(f"Created {c['name']} Bee ({c['computer_text']})")
else:
    sys.exit(__doc__)
'''


registry = Registry()  # the Hive's one registry, shared by the gateway and the chat
