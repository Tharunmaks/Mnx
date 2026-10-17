"""The Bee registry: the Bees each user makes, and the Bees that Bees make. There are no premade Bees: the Hive's
own work (chat, memory, the browser, the phone, training) is done by functions, not Bees.

Every Bee has
  - an owner (hive/users.py): only they see it, chat with it and use its computer
  - a card: name and its responsibility (one line), plus a profile: instructions, tone, greeting, starters,
    memory, voice (hive/bots.py)
  - its own chat (Peace mode), and a place on the board when all your Bees work together (Rush mode)
  - a family: who made it ("you" or another Bee); deleting a Bee hands its children to its own parent
  - its own computer (a Cell) with Vortex OS. Nobody picks its size: the owner's plan decides it (Free or Pro)
  - a key that lets the Bee, from inside its own computer, create and list its children and nothing else

Limits: the plan's number of Bees per user, MAX_DEPTH generations, MAX_CHILDREN per Bee, MAX_BEES on the Hive.
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

MAX_BEES = 400     # on the whole Hive; each user's plan has its own limit (hive/plan.py)
MAX_DEPTH = 4       # you → Bee → child → grandchild → great-grandchild
MAX_CHILDREN = 12

# Computers aren't chosen: every Bee gets its owner's plan's computer (hive/plan.py), with Vortex OS.
SIZES = {
    "free": {"cpus": 2, "memory_gb": 8, "disk_gb": 40, "label": "Free"},
    "pro": {"cpus": 4, "memory_gb": 16, "disk_gb": 100, "gpu_vram_gb": 16, "label": "Pro"},
}
ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")
# Ids the Hive's own functions use for their computers (the shared browser) or used before: never a user's Bee id.
RESERVED_IDS = {"browser", "phone", "cloud", "gpu", "queen", "chat", "memory", "reader", "architect", "data", "coding", "eval", "hive"}


class BeeError(ValueError):
    """A Bee can't be created or changed that way (a limit, a name clash)."""


OS_CHOICES = {"vortex": "Vortex OS"}
DEFAULT_OS = "vortex"


def computer_for(owner: str, base: dict | None = None) -> dict:
    """The computer the owner's plan gives: Free or Pro. A Pro computer goes to the Hive's own cloud when it has room."""
    from .plan import plan
    size = plan.of(owner) if plan.of(owner) in SIZES else "free"
    z = SIZES[size]
    base = dict(base or {})
    where = base.get("where", "here")
    if size == "pro" and where == "here" and not base:
        try:
            from .cloud import pool_place
            where = pool_place() or "here"
        except Exception:
            where = "here"
    out = {"size": size, "cpus": z["cpus"], "memory_gb": z["memory_gb"], "disk_gb": z["disk_gb"], "gpu": bool(z.get("gpu_vram_gb")),
           "where": where, "os": DEFAULT_OS}
    if base.get("cloud"):
        out["cloud"] = base["cloud"]
    return out


def describe_computer(c: dict, where_name: str | None = None) -> str:
    c = c or computer_for("owner")
    parts = [f"{c['cpus']:g} CPU{'s' if c['cpus'] != 1 else ''}", f"{c['memory_gb']:g} GB RAM"]
    if c.get("disk_gb"):
        parts.append(f"{c['disk_gb']:g} GB disk")
    if c.get("gpu"):
        vram = SIZES.get(c.get("size"), {}).get("gpu_vram_gb")
        parts.append(f"GPU {vram} GB VRAM" if vram else "GPU")
    label = SIZES.get(c["size"], {}).get("label", "Free")
    return f"{label} · {', '.join(parts)} · Vortex OS"


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
        from .bots import clean_profile
        # The Hive's own work (chat, memory, the browser, the phone, training …) is done by functions, not Bees:
        # Bees saved as built in are dropped. Every Bee is yours, with a responsibility, its own chat and a profile.
        self.bees: list[dict] = [dict(b) for b in saved if not b.get("builtin")]
        for b in self.bees:
            b.setdefault("owner", "owner")  # made before users had their own Bees: the owner's
            b.setdefault("created_by", "you")
            if not isinstance(b.get("bot"), dict):
                b["bot"] = clean_profile({})
            old = b.get("computer") or {}
            # always the owner's plan's computer now (sizes chosen before are gone); it stays on its server
            b["computer"] = computer_for(b["owner"], {"where": old.get("where", "here"), "cloud": old.get("cloud")})
            b.pop("computer_set", None)
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

    def of(self, owner: str) -> list[dict]:
        """One user's Bees."""
        return [b for b in self.bees if b.get("owner", "owner") == owner]

    def mine(self, bee_id: str, owner: str) -> dict | None:
        """The Bee, if it belongs to this user (the Hive's owner can reach every Bee)."""
        b = self.get(bee_id)
        return b if b and (owner == "owner" or b.get("owner", "owner") == owner) else None

    def find(self, words: str, owner: str | None = None) -> dict | None:
        """One of the user's Bees by id or name, ignoring case and a trailing 'bee'."""
        from .users import who
        pool = self.of(owner or who())
        q = re.sub(r"\bbees?\b", "", (words or "").lower()).strip(" .,'\"")
        if not q:
            return None
        for b in pool:
            if q == b["name"].lower() or b["id"] in (q, slug(q)):
                return b
        hits = [b for b in pool if b["name"].lower().startswith(q)]
        return hits[0] if len(hits) == 1 else None

    def children(self, bee_id: str) -> list[dict]:
        return [b for b in self.bees if b.get("created_by") == bee_id]

    def depth(self, bee_id: str | None) -> int:
        """Generations below you: a Bee you made is 1, its child 2 …"""
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
                     schedule: str | None = None, approval: bool = True, profile: dict | None = None,
                     owner: str | None = None) -> dict:
        """A new Bee: a name and its responsibility (one line). It gets its own chat and its plan's computer."""
        from .bots import clean_profile
        from .plan import plan
        from .users import who
        name = " ".join((name or "").split())[:24]
        skill = " ".join((skill or "").split())[:90]
        if not name or not skill:
            raise BeeError("A Bee needs a name and a responsibility")
        if created_by != "you":
            parent = self.get(created_by)
            if not parent:
                raise BeeError("The parent Bee doesn't exist")
            owner = parent.get("owner", "owner")
            if self.depth(created_by) + 1 > MAX_DEPTH:
                raise BeeError(f"Bees can only go {MAX_DEPTH} generations deep")
            if len(self.children(created_by)) >= MAX_CHILDREN:
                raise BeeError(f"{parent['name']} already has {MAX_CHILDREN} children, its limit")
        owner = owner or who()
        mine = self.of(owner)
        if any(b["name"].lower() == name.lower() for b in mine):
            raise BeeError(f"You already have a Bee called {name}")
        limit = plan.max_bees(owner)
        if len(mine) >= limit:
            raise BeeError(f"Your plan has room for {limit} Bees; remove one first"
                           + ("" if plan.of(owner) == "pro" else ", or get Pro for more"))
        if len(self.bees) >= MAX_BEES:
            raise BeeError("The Hive is full right now; try again later")
        base = slug(name)
        if not ID_RE.match(base):
            raise BeeError("Use letters and numbers in the name")
        bee_id, n = base, 2
        while self.get(bee_id) or bee_id in RESERVED_IDS:  # ids are unique across the Hive; names only within your own Bees
            bee_id = f"{base[:36]}_{n}"
            n += 1
        record = {"id": bee_id, "name": name, "skill": skill, "tools": list(tools or [])[:20], "schedule": schedule or None,
                  "approval": bool(approval), "builtin": False, "created_by": created_by, "owner": owner,
                  "computer": computer_for(owner), "bot": clean_profile(profile or {})}
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

    async def move(self, bee_id: str, where: str, cloud: str | None = None) -> dict:
        """The Hive moves a Bee's computer to another server (same size; files stay where they were)."""
        b = self.get(bee_id)
        if not b:
            raise BeeError("No such Bee")
        if where != "here" and not re.fullmatch(r"[a-z0-9-]{1,48}", where):
            raise BeeError("Bad server id")
        old = dict(b["computer"])
        new = {**old, "where": where}
        new.pop("cloud", None)
        if cloud:
            new["cloud"] = cloud
        b["computer"] = new
        self.save()
        if self.on_computer:
            try:
                await self.on_computer(b, old)
            except Exception:
                b["computer"] = old
                self.save()
                raise
        return b

    async def apply_plan(self, owner: str) -> list[dict]:
        """Give each of the owner's Bees the computer their plan has now (after Pro starts or ends)."""
        changed = []
        for b in self.of(owner):
            old = dict(b["computer"])
            new = computer_for(owner, {"where": old.get("where", "here"), "cloud": old.get("cloud")})
            if new.get("size") == "pro" and old.get("size") != "pro" and new.get("where") == "here":
                try:
                    from .cloud import pool_place
                    new["where"] = pool_place() or "here"
                except Exception:
                    pass
            if new == old:
                continue
            b["computer"] = new
            self.save()
            if self.on_computer:
                try:
                    await self.on_computer(b, old)
                except Exception:
                    b["computer"] = old  # the move failed: keep the Bee where it was
                    self.save()
                    continue
            changed.append(b)
        return changed


# ---------- the helper each Bee gets inside its own computer ----------
HELPER = r'''#!/usr/bin/env python3
"""mnx-bee: this Bee talks to its Hive. It can only create and list its own children.

  python3 .mnx/mnx-bee me
  python3 .mnx/mnx-bee children
  python3 .mnx/mnx-bee new "Name" "its responsibility"
  python3 .mnx/mnx-bee gpu train.py data.csv          run "python train.py" on the Hive's GPU (Pro, 16 GB VRAM)
  python3 .mnx/mnx-bee gpu --cmd "bash go.sh" go.sh   any command; the files listed go with it
"""
import json, os, sys, urllib.request

def from_file():
    """Desktop terminals (Vortex OS) start with a clean environment: read this Bee's .mnx/env instead."""
    import shlex
    out = {}
    try:
        for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env")):
            parts = shlex.split(line.strip())
            if len(parts) == 2 and parts[0] == "export" and "=" in parts[1]:
                k, v = parts[1].split("=", 1)
                out[k] = v
    except OSError:
        pass
    return out

saved = from_file()
hive = os.environ.get("MNX_HIVE") or saved.get("MNX_HIVE")
key = os.environ.get("MNX_BEE_KEY") or saved.get("MNX_BEE_KEY")
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
    c = call("POST", "/children", {"name": a[1], "skill": a[2]})
    print(f"Created {c['name']} Bee ({c['computer_text']})")
elif a[0] == "gpu" and len(a) >= 2:
    import base64, time
    rest = a[1:]
    cmd = None
    if rest[0] == "--cmd" and len(rest) >= 2:
        cmd, rest = rest[1], rest[2:]
    elif rest[0].endswith(".py"):
        cmd = "python " + " ".join(rest)
    else:
        cmd = " ".join(rest)
    files = {}
    for f in rest:
        if os.path.isfile(f):
            files[f] = base64.b64encode(open(f, "rb").read()).decode()
    job = call("POST", "/gpu", {"command": cmd, "files": files})
    print(f"GPU job {job['id']} on {job['gpu']}: {cmd}")
    while job["status"] in ("queued", "running"):
        time.sleep(20)
        job = call("GET", "/gpu/" + job["id"])
        print(f"  {job['status']}…", flush=True)
    print((job.get("log") or "")[-3000:])
    print(f"{job['status']}" + (f": {job['error']}" if job.get("error") else "") + f" · outputs in gpu-out/{job['id']}/")
else:
    sys.exit(__doc__)
'''


registry = Registry()  # the Hive's one registry, shared by the gateway and the chat
