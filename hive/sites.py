"""Bees build full-stack websites with Chroma (github.com/Tharunmaks/Chroma-node) and keep them in their Vortex OS computer.

  "build a website for my bakery where people can order cakes"
  "make a portfolio site for my photos"     "create a landing page for my app with a waitlist"
  "update my bakery site: add a contact page"   "list my sites"

How
  1. The Hive runs one Chroma platform (Node.js 22.5+) on this machine, started the first time it's needed
     (cloned from MNX_CHROMA_REPO into data/chroma). Each Hive user gets their own Chroma account; the Hive keeps
     its API token (data/chroma/accounts.json, owner-only).
  2. The model writes the site: HTML pages, CSS, a browser script that uses Chroma's realtime database
     (/assets/js/chroma-sdk.js), server functions in api/*.js, and the database collections with their rules.
  3. The files go into the Bee's own computer, ~/sites/<name>/, so you can open and edit them in its Vortex OS
     desktop (files, editor, terminal), and the Hive deploys them to Chroma.
  4. The site is live at <your Hive>/p/<name>/ (the Hive passes Chroma's site paths through), with its database,
     functions and analytics. Updating redeploys; Chroma keeps every deployment for rollback.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import secrets
import shutil
from pathlib import Path

import httpx

from . import brain
from .core import Task

DATA = Path(__file__).resolve().parent.parent / "data"
HOME = DATA / "chroma"
APP_DIR = HOME / "app"
ACCOUNTS = HOME / "accounts.json"
REPO = os.getenv("MNX_CHROMA_REPO", "https://github.com/Tharunmaks/Chroma-node")
PORT = int(os.getenv("MNX_CHROMA_PORT", "3100"))
# Chroma's own paths a deployed site uses; the gateway passes these through to Chroma (gateway/api.py).
PASS_THROUGH = ("/p/", "/d/", "/assets/", "/vendor/three/", "/vendor/pyre/", "/api/db/", "/api/fn/", "/api/a/")
MAX_FILES, MAX_BYTES = 40, 2_000_000

BUILD = re.compile(r"\b(?:build|make|create|design|generate|set\s+up|launch|code)\s+(?:me\s+|us\s+)?(?:a|an|my|our|the|one)?\s*"
                   r"(?:[\w'-]+\s+){0,4}?(?:full[\s-]*stack\s+)?(?:website|web\s*site|web\s*app|webapp|site|landing\s+page|"
                   r"online\s+store|web\s*store|e-?commerce\s+site|portfolio|blog|home\s*page|web\s+page|webpage)\b", re.I)
UPDATE = re.compile(r"\b(?:update|change|edit|improve|fix|redesign|add\s+\w+(?:\s+\w+){0,3}\s+to)\s+(?:my|the|our)\s+(?P<name>[\w -]{1,40}?)\s+"
                    r"(?:website|web\s*site|web\s*app|site|page|store|portfolio|blog)\b", re.I)
LIST = re.compile(r"\b(?:list|show|see)\s+(?:me\s+)?(?:all\s+)?my\s+(?:websites|sites|web\s*apps)\b|\bmy\s+(?:websites|sites)\b", re.I)
FILE_BLOCK = re.compile(r"^===\s*FILE:\s*(?P<path>[^\n=]+?)\s*===\s*\n(?P<body>.*?)\n===\s*END\s*===", re.S | re.M)
COLL_BLOCK = re.compile(r"^===\s*COLLECTIONS\s*===\s*\n(?P<body>.*?)\n===\s*END\s*===", re.S | re.M)
COLL_LINE = re.compile(r"^\s*-?\s*(?P<name>[a-z][a-z0-9_-]{0,39})\s*:\s*read\s*=\s*(?P<read>public|private)\s+write\s*=\s*(?P<write>public|create|private)",
                       re.I | re.M)
SAFE_PATH = re.compile(r"^(?!.*\.\.)[A-Za-z0-9_][\w./-]{0,120}$")

SPEC = """You build complete, working, good-looking full-stack websites that run on Chroma, a Node.js platform.
Write every file of the site in this exact format, nothing before or after:

=== FILE: index.html ===
(the file's full content)
=== END ===
=== FILE: style.css ===
...
=== END ===
=== COLLECTIONS ===
orders: read=private write=create
=== END ===

The site
- Plain HTML, CSS and JavaScript (ES modules). No npm packages, no build step, no frameworks. index.html is required;
  other pages are more .html files. Link files relatively (style.css, app.js, about.html). Mobile first, accessible,
  a clear design with real content written for this purpose (no lorem ipsum).
- Database: in a page's script, `import { chroma } from '/assets/js/chroma-sdk.js'; const app = chroma();`
  then `const orders = app.db('orders'); await orders.add({...}); const { docs } = await orders.list({ order: '-_created', limit: 50 });
  orders.subscribe((change) => { /* change.type: add|update|delete, change.doc */ });`. Every document gets _id and _created.
- Collections and who may use them go in COLLECTIONS, one per line: name: read=public|private write=public|create|private.
  read=public: anyone can list; private: only the site owner (in the dashboard). write=create: visitors can add but not
  change (forms, orders, sign-ups, guestbooks); public: anyone can change; private: only the owner. Keep private
  what visitors shouldn't see (orders, emails).
- Server functions: api/NAME.js with `export default async function (req, { db }) { const { docs } = await db.collection('menu').list({ limit: 50 });
  return { json: { ... } }; }`. The page calls `await app.fn('NAME', { method: 'POST', body: { ... } })`. Use them for
  totals, checks and anything that shouldn't run in the browser. req.body is the parsed JSON body.
- Analytics: `app.track('order.placed')` for the events that matter.
"""


class SiteError(RuntimeError):
    pass


# ---------------------------------------------------------------- what was asked
def parse(text: str) -> dict | None:
    t = " ".join((text or "").split())
    if not t:
        return None
    if LIST.search(t):
        return {"kind": "list"}
    m = UPDATE.search(t)
    if m:
        return {"kind": "update", "name": slug(m.group("name")), "goal": t}
    if not BUILD.search(t):
        return None
    from .cloud_chat import CODE_ASK
    if CODE_ASK.search(t):  # "build a website in react", "how do I make a site" are for the model to answer
        return None
    nm = re.search(r"\b(?:called|named)\s+[\"“']?(?P<n>[\w -]{1,40}?)[\"”']?(?=\s+(?:that|which|with|for|where)\b|[,.!?]|$)", t, re.I)
    return {"kind": "build", "name": slug(nm.group("n")) if nm else "", "goal": t}


def slug(text: str) -> str:
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", (text or "").lower())).strip("-")[:40]


def name_for(goal: str) -> str:
    """A project name from the request: "a website for my bakery where …" → "bakery"."""
    m = re.search(r"\bfor\s+(?:my|our|a|an|the)?\s*(?P<w>[\w'-]+(?:\s+[\w'-]+)?)", goal, re.I)
    stop = r"\b(?:website|site|web|app|page|that|which|where|with|people|can|my|our|the|a|an|for|me|us|build|make|create|design|" \
           r"generate|set|up|launch|code|full|stack|from|scratch|new|simple|nice|good)\b"
    for words in ((m.group("w") if m else ""), goal):  # what it's for, else the kind of site ("landing page" → landing)
        name = slug(" ".join(re.sub(stop, " ", words.lower()).split()[:2]))
        if name:
            return name
    return "my-site"


def parse_files(raw: str) -> tuple[dict[str, str], list[dict]]:
    files: dict[str, str] = {}
    for m in FILE_BLOCK.finditer(raw or ""):
        path = m.group("path").strip().removeprefix("./")
        if SAFE_PATH.match(path) and len(files) < MAX_FILES:  # no "..", no absolute paths: only files inside the site
            files[path] = m.group("body")
    if not files:  # a model that wrote one HTML page without the markers
        h = re.search(r"<!doctype html.*?</html>", raw or "", re.S | re.I)
        if h:
            files["index.html"] = h.group(0)
    cm = COLL_BLOCK.search(raw or "")
    cols = [{"name": c.group("name").lower(), "read": c.group("read").lower(), "write": c.group("write").lower()}
            for c in COLL_LINE.finditer(cm.group("body") if cm else "")]
    if sum(len(v.encode()) for v in files.values()) > MAX_BYTES:
        raise SiteError("The site came out larger than 2 MB")
    return files, cols


# ---------------------------------------------------------------- the Chroma platform on this machine
class Chroma:
    def __init__(self, home: Path = HOME, port: int = PORT):
        self.home, self.port = home, port
        self.app_dir = home / "app"
        self.accounts_file = home / "accounts.json"
        self.url = os.getenv("MNX_CHROMA_URL", f"http://127.0.0.1:{port}").rstrip("/")
        self.proc: asyncio.subprocess.Process | None = None
        self.transport: httpx.AsyncBaseTransport | None = None  # tests use a fake Chroma
        self._lock = asyncio.Lock()

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3, transport=self.transport) as c:
                return (await c.get(self.url + "/api/health")).status_code == 200
        except httpx.HTTPError:
            return False

    async def ensure(self) -> None:
        """Running, or start it (installing it the first time)."""
        async with self._lock:
            if await self.healthy():
                return
            node = shutil.which("node")
            if not node:
                raise SiteError("Chroma needs Node.js 22.5 or newer on the Hive (Termux: pkg install nodejs)")
            ver = (await _run([node, "--version"], 15))[1].strip().lstrip("v")
            if tuple(int(x) for x in re.findall(r"\d+", ver)[:2]) < (22, 5):
                raise SiteError(f"Chroma needs Node.js 22.5 or newer; this machine has {ver}")
            if not (self.app_dir / "server" / "index.js").exists():
                self.home.mkdir(parents=True, exist_ok=True)
                code, out = await _run(["git", "clone", "--depth", "1", REPO, str(self.app_dir)], 600)
                if code:
                    raise SiteError(f"Couldn't download Chroma: {out[-300:]}")
            if not (self.app_dir / "node_modules").exists():
                npm = shutil.which("npm") or "npm"
                code, out = await _run([npm, "install", "--omit=dev", "--no-audit", "--no-fund"], 900, cwd=self.app_dir)
                if code:
                    raise SiteError(f"Couldn't install Chroma: {out[-300:]}")
            env = {**os.environ, "PORT": str(self.port), "HOST": "127.0.0.1", "DATA_DIR": str(self.home / "data"), "TRUST_PROXY": "1"}
            log = open(self.home / "chroma.log", "ab")
            self.proc = await asyncio.create_subprocess_exec(node, "server/index.js", cwd=self.app_dir, env=env, stdout=log, stderr=log)
            for _ in range(60):
                await asyncio.sleep(0.5)
                if await self.healthy():
                    return
            raise SiteError("Chroma didn't start; see data/chroma/chroma.log")

    async def stop(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 10)
            except asyncio.TimeoutError:
                self.proc.kill()

    # ----- one Chroma account per Hive user -----
    def _accounts(self) -> dict:
        try:
            return json.loads(self.accounts_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_accounts(self, acc: dict) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        tmp = self.accounts_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(acc, indent=1))
        os.chmod(tmp, 0o600)
        tmp.replace(self.accounts_file)

    async def token(self, uid: str) -> str:
        acc = self._accounts()
        a = acc.get(uid)
        if a and a.get("token"):
            return a["token"]
        username = ("mnx-" + re.sub(r"[^a-z0-9]", "", uid.lower()))[:32]
        password = secrets.token_urlsafe(24)
        async with httpx.AsyncClient(timeout=30, transport=self.transport) as c:
            r = await c.post(self.url + "/api/auth/signup", json={"username": username, "password": password})
            if r.status_code == 409 or (r.status_code == 400 and "taken" in r.text):
                raise SiteError("A Chroma account with this user's name exists but the Hive doesn't have its password")
            if r.status_code >= 400:
                raise SiteError(f"Chroma refused the account: {_err(r)}")
            r = await c.post(self.url + "/api/auth/token", json={"username": username, "password": password, "name": "mnx-hive"})
            if r.status_code >= 400:
                raise SiteError(f"Chroma refused a token: {_err(r)}")
        acc[uid] = {"username": username, "password": password, "token": r.json()["token"]}
        self._save_accounts(acc)
        return acc[uid]["token"]

    async def _call(self, uid: str, method: str, path: str, body: dict | None = None) -> httpx.Response:
        tok = await self.token(uid)
        async with httpx.AsyncClient(timeout=120, transport=self.transport) as c:
            return await c.request(method, self.url + path, json=body, headers={"Authorization": f"Bearer {tok}"})

    async def projects(self, uid: str) -> list[dict]:
        r = await self._call(uid, "GET", "/api/projects")
        return r.json().get("projects", []) if r.status_code == 200 else []

    async def deploy(self, uid: str, name: str, files: dict[str, str | bytes], collections: list[dict], private: bool = False) -> dict:
        """Make the project if it's new (another name if that one is taken), upload, set up the collections. Private:
        a preview deployment only, at an unguessable /d/<id>/ link, never the public /p/<name>/."""
        mine = {p["name"] for p in await self.projects(uid)}
        name = base = slug(name) or "my-site"
        n = 2
        while name not in mine:
            r = await self._call(uid, "POST", "/api/projects", {"name": name})
            if r.status_code < 300:
                break
            if r.status_code == 409 or "taken" in r.text or "reserved" in r.text:
                name = f"{base[:36]}-{n}"
                n += 1
                if n > 30:
                    raise SiteError("Couldn't find a free name for the site")
                continue
            raise SiteError(f"Chroma: {_err(r)}")
        payload = {"target": "preview" if private else "production", "files": [{"path": p, "data": base64.b64encode(v.encode() if isinstance(v, str) else v).decode()}
                                                      for p, v in files.items()]}
        r = await self._call(uid, "POST", f"/api/projects/{name}/deployments", payload)
        if r.status_code >= 400:
            raise SiteError(f"Chroma: {_err(r)}")
        d = r.json()
        dep = d.get("deployment") or {}
        errors = [f"{x.get('file')}: {x.get('message')}" for x in dep.get("log") or [] if x.get("level") == "error"]
        if dep.get("status") != "READY":
            raise SiteError("The site didn't build: " + ("; ".join(errors)[:300] or dep.get("status", "unknown")))
        for col in collections:
            await self._call(uid, "POST", f"/api/projects/{name}/db", col)
        return {"name": name, "url": dep.get("url") if private else (d.get("production") or f"/p/{name}/"), "preview": dep.get("url"),
                "functions": dep.get("functions") or [], "files": len(files), "deployment": dep.get("id")}


def _err(r: httpx.Response) -> str:
    try:
        return str(r.json().get("error") or r.text)[:200]
    except ValueError:
        return r.text[:200]


async def _run(argv: list[str], timeout: float, cwd: Path | None = None) -> tuple[int, str]:
    try:
        p = await asyncio.create_subprocess_exec(*argv, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await asyncio.wait_for(p.communicate(), timeout)
        return p.returncode or 0, out.decode(errors="replace")
    except (OSError, asyncio.TimeoutError) as exc:
        return 1, str(exc)


chroma = Chroma()


# ---------------------------------------------------------------- the Bee's computer: ~/sites/<name>/
async def save_to_bee(b: dict, name: str, files: dict[str, str | bytes], project: str, folder: str = "sites") -> str:
    from .cells import cells
    cell = cells.get(b["id"], b["name"], b["computer"])
    meta = {"name": project, "server": "Mnx Hive (Chroma)"}
    for path, body in {**files, "chroma.json": json.dumps(meta, indent=2)}.items():
        rel = f"{folder}/{name}/{path}"
        data = body if isinstance(body, bytes) else body.encode()
        if cell.mode == "remote":
            await cell.rwrite(rel, data)
        else:
            p = cell.safe_path(rel)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
    return f"~/{folder}/{name}"


async def read_from_bee(b: dict, name: str) -> dict[str, str]:
    from .cells import cells
    cell = cells.get(b["id"], b["name"], b["computer"])
    if cell.mode == "remote":
        return {}
    root = cell.safe_path(f"sites/{name}")
    out = {}
    for p in sorted(root.rglob("*")) if root.is_dir() else []:
        if p.is_file() and p.name != "chroma.json" and p.stat().st_size < 200_000 and len(out) < MAX_FILES:
            out[str(p.relative_to(root))] = p.read_text(errors="replace")
    return out


# ---------------------------------------------------------------- in the chat
async def write_site(task: Task, goal: str, current: dict[str, str] | None = None) -> tuple[dict[str, str], list[dict]]:
    msgs = [{"role": "system", "content": SPEC}]
    if current:
        listing = "\n".join(f"=== FILE: {p} ===\n{v[:6000]}\n=== END ===" for p, v in current.items())
        msgs.append({"role": "user", "content": f"The site now:\n{listing}\n\nChange it: {goal}\nWrite every file of the new version."})
    else:
        msgs.append({"role": "user", "content": f"Build this site: {goal}"})
    raw = await brain.brain.chat(msgs, task.effort, max_tokens=8000, temperature=0.4, builder=True)
    files, cols = parse_files(raw)
    if "index.html" not in files:
        raise SiteError("The model didn't write an index.html")
    return files, cols


def _pick_bee(task: Task) -> dict | None:
    from .bees import registry
    if task.bee:
        return registry.mine(task.bee, task.user)
    mine = registry.of(task.user)
    words = re.compile(r"\b(?:web\w*|site\w*|code\w*|develop\w*|build\w*|design\w*|app\w*)\b", re.I)
    return next((b for b in mine if words.search(b["skill"]) or words.search(b["name"])), mine[0] if mine else None)


async def handle(task: Task, intent: dict) -> None:
    if intent["kind"] == "list":
        try:
            await chroma.ensure()
            ps = await chroma.projects(task.user)
        except SiteError as exc:
            await task.answer(str(exc))
            return
        if not ps:
            await task.answer("You don't have any sites yet. Say “build a website for … that …”.")
            return
        await task.emit("result", service="Sites", title=f"{len(ps)} site{'s' if len(ps) != 1 else ''}", badge="Chroma",
                        details=[[p["name"], f"{p['url']} · {p['deployments']} deployment{'s' if p['deployments'] != 1 else ''}"] for p in ps])
        await task.answer("Your sites: " + ", ".join(f"[{p['name']}]({p['url']})" for p in ps))
        return
    b = _pick_bee(task)
    if not b:
        await task.answer("A Bee builds the site in its own computer. Make one first (New Bee in Peace, e.g. “Webby: builds my websites”), then ask again.")
        return
    if not brain.brain.can_build():
        await task.answer("Building a site needs a model running on this Hive, and none is yet.")
        return
    bee = b["name"]
    name = intent.get("name") or name_for(intent["goal"])
    current = None
    if intent["kind"] == "update":
        current = await read_from_bee(b, name)
        if not current:
            await task.answer(f"{bee} doesn't have a site called {name} in ~/sites. Say “list my sites” to see them.")
            return
    s = await task.step("writing", f"{'Updating' if current else 'Writing'} the site: pages, styles, database and server functions", bee=bee)
    try:
        files, cols = await write_site(task, intent["goal"], current)
    except (SiteError, brain.BrainNotConnected) as exc:
        await task.finish_step(s, status="error", text=str(exc)[:200])
        await task.answer(f"I couldn't write the site: {exc}")
        return
    await task.finish_step(s, text=f"{len(files)} files" + (f", {len(cols)} collection{'s' if len(cols) != 1 else ''}" if cols else ""))
    s = await task.step("running", "Deploying with Chroma", bee=bee)
    try:
        await chroma.ensure()
        d = await chroma.deploy(task.user, name, files, cols)
        folder = await save_to_bee(b, d["name"], files, d["name"])
    except (SiteError, httpx.HTTPError, OSError, RuntimeError) as exc:
        await task.finish_step(s, status="error", text=str(exc)[:200])
        await task.answer(f"The site is written but didn't deploy: {exc}")
        return
    await task.finish_step(s, text=f"live at {d['url']}")
    from . import sandbox
    sandbox.record(task.user, "site", d["name"], bee, d["url"])
    details = [["Live", d["url"]], ["Files", ", ".join(list(files)[:8]) + (" …" if len(files) > 8 else "")],
               ["In Vortex OS", f"{bee}'s computer, {folder} (open its desktop to edit)"]]
    if cols:
        details.append(["Database", ", ".join(f"{c['name']} (read {c['read']}, write {c['write']})" for c in cols)])
    if d["functions"]:
        details.append(["Server functions", ", ".join(f"/api/fn/{d['name']}/{f}" for f in d["functions"])])
    await task.emit("result", bee=bee, service="Chroma", title=f"{d['name']} is live", badge="Site", details=details)
    await task.answer(f"[{d['name']}]({d['url']}) is live. Its files are in {bee}'s Vortex OS computer at {folder}; "
                      f"say “update my {d['name']} site: …” to change it.")
