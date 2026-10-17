"""Real cloud computers for Bees: one server per Bee, rented from a cloud provider.

Each Bee can have its own machine in the cloud: at least 8 GB memory and 40 GB disk (the Standard computer),
picked as the cheapest server type that fits, in the region you choose. The Hive starts it with its own SSH
key, Docker and Node.js (so Vortex OS runs there), adds it as a Lab server and moves the Bee onto it. Removing
the Bee, or moving it off, deletes the server so the billing stops.

Providers (API token saved in Lab → Secrets):
  Hetzner Cloud   HCLOUD_TOKEN         cheapest; Singapore is the nearest region to India
  DigitalOcean    DIGITALOCEAN_TOKEN   has a Bangalore region (blr1)
GPU machines (Paid plan) are rented on RunPod (RUNPOD_API_KEY) through the GPU Bee, as before.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

DATA = Path(__file__).resolve().parent.parent / "data"
MACHINES_FILE = DATA / "lab" / "machines.json"

MIN_MEMORY_GB = 8
MIN_DISK_GB = 40
MIN_CPUS = 2


class CloudError(RuntimeError):
    pass


def cloud_init(public_key: str) -> str:
    """First boot: only the Hive's key may log in (no passwords), then Docker and Node.js for the Bee's computer
    and Vortex OS. Runs in the background; SSH works while it installs."""
    return "\n".join([
        "#cloud-config",
        "ssh_pwauth: false",
        "disable_root: false",
        "users:",
        "  - name: root",
        "    ssh_authorized_keys:",
        f"      - {public_key.strip()}",
        "package_update: true",
        "packages: [docker.io, python3, git, curl, ca-certificates]",
        "runcmd:",
        "  - systemctl enable --now docker",
        "  - curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs make g++ || true",
        "  - mkdir -p /root/mnx && touch /root/mnx/.ready",
        "",
    ])


def server_name(bee_id: str) -> str:
    return "mnx-" + re.sub(r"[^a-z0-9-]", "-", bee_id.lower())[:50]


def pick(types: list[dict], memory_gb: float = MIN_MEMORY_GB, disk_gb: float = MIN_DISK_GB, cpus: int = MIN_CPUS,
         region: str | None = None) -> dict | None:
    """The cheapest type with at least this much memory, disk and CPUs (available in the region, if given)."""
    ok = [t for t in types if t["memory_gb"] >= memory_gb and t["disk_gb"] >= disk_gb and t["cpus"] >= cpus
          and (not region or region in t["regions"])]
    return min(ok, key=lambda t: (t["price_month"], t["memory_gb"])) if ok else None


# ---------------------------------------------------------------- Hetzner Cloud
class Hetzner:
    id, label, secret = "hetzner", "Hetzner Cloud", "HCLOUD_TOKEN"
    url = "https://api.hetzner.cloud/v1"
    currency = "EUR"
    preferred = ("sin", "fsn1", "nbg1", "hel1", "ash", "hil")  # Singapore first: the nearest to India

    def __init__(self, token: str, transport: httpx.AsyncBaseTransport | None = None):
        self.token, self.transport = token, transport

    async def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        async with httpx.AsyncClient(timeout=40, transport=self.transport) as c:
            r = await c.request(method, self.url + path, json=body, headers={"Authorization": f"Bearer {self.token}"})
        if r.status_code == 401:
            raise CloudError("Hetzner rejected the API token (HCLOUD_TOKEN)")
        if r.status_code == 204 or not r.content:
            return {}
        data = r.json()
        if r.status_code >= 400:
            err = data.get("error") or {}
            raise CloudError(f"Hetzner: {err.get('message') or r.text[:200]}")
        return data

    async def types(self) -> list[dict]:
        data = await self._call("GET", "/server_types?per_page=50")
        out = []
        for t in data.get("server_types") or []:
            if t.get("deprecated") or t.get("deprecation") or t.get("architecture", "x86") != "x86":
                continue
            prices = {p["location"]: p for p in t.get("prices") or []}
            if not prices:
                continue
            region = next((r for r in self.preferred if r in prices), next(iter(prices)))
            p = prices[region]
            out.append({"type": t["name"], "cpus": int(t.get("cores") or 0), "memory_gb": float(t.get("memory") or 0),
                        "disk_gb": float(t.get("disk") or 0), "regions": list(prices), "region": region,
                        "price_month": float(p["price_monthly"]["gross"]), "price_hour": float(p["price_hourly"]["gross"])})
        return out

    async def create(self, name: str, type_: str, region: str, user_data: str, labels: dict) -> dict:
        data = await self._call("POST", "/servers", {
            "name": name, "server_type": type_, "image": "ubuntu-24.04", "location": region, "user_data": user_data,
            "labels": labels, "start_after_create": True, "public_net": {"enable_ipv4": True, "enable_ipv6": True}})
        s = data.get("server") or {}
        if not s.get("id"):
            raise CloudError("Hetzner didn't create the server")
        return {"id": str(s["id"]), "status": s.get("status"), "ip": ((s.get("public_net") or {}).get("ipv4") or {}).get("ip")}

    async def get(self, server_id: str) -> dict:
        s = (await self._call("GET", f"/servers/{int(server_id)}")).get("server") or {}
        return {"id": str(s.get("id")), "status": "running" if s.get("status") == "running" else s.get("status"),
                "ip": ((s.get("public_net") or {}).get("ipv4") or {}).get("ip")}

    async def delete(self, server_id: str) -> None:
        try:
            await self._call("DELETE", f"/servers/{int(server_id)}")
        except CloudError as exc:
            if "not found" not in str(exc).lower():
                raise


# ---------------------------------------------------------------- DigitalOcean
class DigitalOcean:
    id, label, secret = "digitalocean", "DigitalOcean", "DIGITALOCEAN_TOKEN"
    url = "https://api.digitalocean.com/v2"
    currency = "USD"
    preferred = ("blr1", "sgp1", "fra1", "ams3", "lon1", "nyc3", "sfo3")  # Bangalore first

    def __init__(self, token: str, transport: httpx.AsyncBaseTransport | None = None):
        self.token, self.transport = token, transport

    async def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        async with httpx.AsyncClient(timeout=40, transport=self.transport) as c:
            r = await c.request(method, self.url + path, json=body, headers={"Authorization": f"Bearer {self.token}"})
        if r.status_code == 401:
            raise CloudError("DigitalOcean rejected the API token (DIGITALOCEAN_TOKEN)")
        if r.status_code == 204 or not r.content:
            return {}
        data = r.json()
        if r.status_code >= 400:
            raise CloudError(f"DigitalOcean: {data.get('message') or r.text[:200]}")
        return data

    async def types(self) -> list[dict]:
        data = await self._call("GET", "/sizes?per_page=200")
        out = []
        for s in data.get("sizes") or []:
            if not s.get("available") or not s.get("regions"):
                continue
            region = next((r for r in self.preferred if r in s["regions"]), s["regions"][0])
            out.append({"type": s["slug"], "cpus": int(s.get("vcpus") or 0), "memory_gb": float(s.get("memory") or 0) / 1024,
                        "disk_gb": float(s.get("disk") or 0), "regions": list(s["regions"]), "region": region,
                        "price_month": float(s.get("price_monthly") or 0), "price_hour": float(s.get("price_hourly") or 0)})
        return out

    async def create(self, name: str, type_: str, region: str, user_data: str, labels: dict) -> dict:
        data = await self._call("POST", "/droplets", {
            "name": name, "region": region, "size": type_, "image": "ubuntu-24-04-x64", "user_data": user_data,
            "tags": ["mnx"] + [f"mnx-{k}-{v}"[:255] for k, v in labels.items()]})
        d = data.get("droplet") or {}
        if not d.get("id"):
            raise CloudError("DigitalOcean didn't create the droplet")
        return {"id": str(d["id"]), "status": d.get("status"), "ip": None}

    async def get(self, server_id: str) -> dict:
        d = (await self._call("GET", f"/droplets/{int(server_id)}")).get("droplet") or {}
        ip = next((n["ip_address"] for n in (d.get("networks") or {}).get("v4") or [] if n.get("type") == "public"), None)
        return {"id": str(d.get("id")), "status": "running" if d.get("status") == "active" else d.get("status"), "ip": ip}

    async def delete(self, server_id: str) -> None:
        try:
            await self._call("DELETE", f"/droplets/{int(server_id)}")
        except CloudError as exc:
            if "not found" not in str(exc).lower() and "could not be found" not in str(exc).lower():
                raise


PROVIDERS = {"hetzner": Hetzner, "digitalocean": DigitalOcean}


# ---------------------------------------------------------------- machines the Hive rented for Bees
class Machines:
    """data/lab/machines.json: {machine id: {provider, server_id, bee, target, type, region, price…}}."""

    def __init__(self, path: Path = MACHINES_FILE):
        self.path = path
        try:
            self.items: dict[str, dict] = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self.items = {}
        self.transport: httpx.AsyncBaseTransport | None = None  # tests swap in a fake cloud
        self.poll = 5.0
        self.on_change = None

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, indent=2))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)
        if self.on_change:
            self.on_change()

    def _lab(self):
        from .lab import lab
        return lab

    def provider(self, pid: str):
        cls = PROVIDERS.get(pid)
        token = self._lab()._secrets.get(cls.secret) if cls else None
        if not cls or not token:
            raise CloudError(f"Save {cls.secret if cls else 'a provider token'} in Lab → Secrets first")
        return cls(token, self.transport)

    def configured(self) -> list[str]:
        secrets = self._lab()._secrets
        return [pid for pid, cls in PROVIDERS.items() if secrets.get(cls.secret)]

    async def quote(self, pid: str | None = None) -> dict:
        """The cheapest Standard (8 GB, 40 GB) machine from one provider, or the cheapest across all configured."""
        pids = [pid] if pid else self.configured()
        if not pids:
            raise CloudError("No cloud provider is connected. Save HCLOUD_TOKEN (Hetzner) or DIGITALOCEAN_TOKEN "
                             "(DigitalOcean) in Lab → Secrets")
        best, errors = None, []
        for p in pids:
            prov = self.provider(p)
            try:
                t = pick(await prov.types())
            except (CloudError, httpx.HTTPError) as exc:
                errors.append(str(exc))
                continue
            if t and (not best or t["price_month"] < best["price_month"]):
                best = {**t, "provider": p, "provider_label": prov.label, "currency": prov.currency}
        if not best:
            raise CloudError("; ".join(errors) or "No server type with 8 GB memory and 40 GB disk is available")
        return best

    def for_bee(self, bee_id: str) -> dict | None:
        return next((m for m in self.items.values() if m["bee"] == bee_id and m["status"] != "deleted"), None)

    async def provision(self, bee_id: str, quote: dict, timeout: float = 600) -> dict:
        """Start a server for the Bee, wait for SSH, add it as a Lab server. Returns the machine record."""
        if self.for_bee(bee_id):
            raise CloudError("This Bee already has a cloud computer")
        lab = self._lab()
        prov = self.provider(quote["provider"])
        key = await lab.hive_key()
        if not key:
            raise CloudError("The Hive has no SSH key (ssh-keygen is missing)")
        s = await prov.create(server_name(bee_id), quote["type"], quote["region"], cloud_init(key), {"mnx": "bee", "bee": bee_id})
        mid = f"{quote['provider']}-{s['id']}"
        rec = {"id": mid, "provider": quote["provider"], "server_id": s["id"], "bee": bee_id, "type": quote["type"],
               "region": quote["region"], "cpus": quote["cpus"], "memory_gb": quote["memory_gb"], "disk_gb": quote["disk_gb"],
               "price_month": quote["price_month"], "cost_per_hour": quote["price_hour"], "currency": quote["currency"],
               "started": int(time.time()), "status": "starting", "ip": s.get("ip"), "target": None}
        self.items[mid] = rec
        self._save()
        try:
            t0 = time.time()
            while not (s.get("status") == "running" and s.get("ip")):
                if time.time() - t0 > timeout:
                    raise CloudError("The server didn't start in 10 minutes")
                await asyncio.sleep(self.poll)
                s = await prov.get(rec["server_id"])
            rec["ip"] = s["ip"]
            t = lab.add_target(f"{PROVIDERS[rec['provider']].label} {rec['type']} · {bee_id}", s["ip"], 22, "root", "/root/mnx", None)
            rec["target"] = t["id"]
            for _ in range(int(timeout / max(self.poll, 0.01)) or 1):  # SSH comes up a little after "running"
                t = await lab.test_target(rec["target"])
                if t["status"] == "ok":
                    break
                await asyncio.sleep(self.poll)
            rec["status"] = "ready" if t["status"] == "ok" else "ssh-failed"
            if rec["status"] != "ready":
                raise CloudError(f"The server is up but the Hive can't log in over SSH yet: {t.get('error') or t['status']}")
        except BaseException:
            await self.release(mid)  # never leave a paid server running that nothing uses
            raise
        self._save()
        return rec

    async def release(self, mid: str) -> None:
        """Delete the server (billing stops) and its Lab entry; record what it cost."""
        rec = self.items.get(mid)
        if not rec or rec["status"] == "deleted":
            return
        try:
            await self.provider(rec["provider"]).delete(rec["server_id"])
        except CloudError as exc:
            rec["status"] = "delete-failed"
            rec["error"] = str(exc)[:300]
            self._save()
            raise
        lab = self._lab()
        if rec.get("target") in lab.targets:
            try:
                lab.remove_target(rec["target"])
            except ValueError:
                pass
        from . import usage
        usage.ledger.record_rental({**rec, "gpu_type": f"{rec['type']} (CPU)"})
        rec["status"] = "deleted"
        rec["ended"] = int(time.time())
        self.items.pop(mid, None)
        self._save()

    async def release_bee(self, bee_id: str, keep_target: str | None = None) -> None:
        m = self.for_bee(bee_id)
        if m and m.get("target") != keep_target:
            await self.release(m["id"])

    def views(self) -> list[dict]:
        now = time.time()
        return [{**m, "hours": round((now - m["started"]) / 3600, 2), "cost_so_far": round((now - m["started"]) / 3600 * m["cost_per_hour"], 4)}
                for m in self.items.values()]


machines = Machines()


def describe_quote(q: dict) -> str:
    sym = {"EUR": "€", "USD": "$"}.get(q.get("currency"), "")
    return (f"{q['provider_label']} {q['type']} · {q['cpus']} CPUs, {q['memory_gb']:g} GB RAM, {q['disk_gb']:g} GB disk · "
            f"{q['region']} · {sym}{q['price_month']:.2f}/month ({sym}{q['price_hour']:.4f}/hour)")


def to_rupees(amount: float, currency: str) -> float | None:
    rate: dict[str, Any] = {"USD": 84.0, "EUR": 91.0}  # approximate, for the card only
    return round(amount * rate[currency]) if currency in rate else None


async def give_bee(bee_id: str, quote: dict) -> dict:
    """Rent a cloud computer for the Bee and move it there. Its old place keeps its files."""
    from .bees import registry
    rec = await machines.provision(bee_id, quote)
    try:
        await registry.move(bee_id, rec["target"], rec["id"])
    except BaseException:
        await machines.release(rec["id"])
        raise
    return rec


async def take_back(bee_id: str) -> None:
    """Move the Bee home to this server and delete its cloud computer."""
    from .bees import registry
    b = registry.get(bee_id)
    if b and b["computer"].get("cloud"):
        await registry.move(bee_id, "here")
    await machines.release_bee(bee_id)


# ---------------------------------------------------------------- the Hive's own free cloud (for Plus computers)
# You add a free server once (Oracle Cloud's Always Free machine: 4 CPUs, 24 GB RAM, 200 GB disk); the Hive sets
# it up and places Plus Bees on it by itself. Users never bring a server or a key.
OVERCOMMIT = 2.0  # memory limits are ceilings, not reservations: a 24 GB machine holds three 16 GB Bees
SETUP = r"""set -e
if command -v apt-get >/dev/null; then PM="sudo apt-get install -y"; sudo apt-get update -y >/dev/null
elif command -v dnf >/dev/null; then PM="sudo dnf install -y"; else PM=""; fi
command -v docker >/dev/null || { [ -n "$PM" ] && ($PM docker.io || $PM docker || $PM podman-docker) >/dev/null; } || true
sudo systemctl enable --now docker >/dev/null 2>&1 || true
sudo usermod -aG docker "$USER" >/dev/null 2>&1 || true
command -v python3 >/dev/null || $PM python3 >/dev/null || true
command -v git >/dev/null || $PM git >/dev/null || true
command -v node >/dev/null || (curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - >/dev/null && $PM nodejs make g++ >/dev/null) || true
mkdir -p ~/mnx
echo "MNX_READY docker=$(command -v docker >/dev/null && echo yes || echo no) node=$(command -v node >/dev/null && echo yes || echo no)"
"""


def pool_servers() -> list[dict]:
    from .lab import lab
    return [t for t in lab.targets.values() if t.get("pool")]


def slots(t: dict) -> int:
    ram_gb = int((t.get("info") or {}).get("ram_mb") or 0) / 1024
    return max(1, int(ram_gb * OVERCOMMIT // 16)) if ram_gb else 1


def pool_place() -> str | None:
    """A Hive cloud server with room for one more Plus Bee, or None."""
    from .bees import registry
    for t in pool_servers():
        if t.get("status") != "ok":
            continue
        used = sum(1 for b in registry.bees if b["computer"].get("where") == t["id"])
        if used < slots(t):
            return t["id"]
    return None


async def add_pool_server(host: str, user: str = "ubuntu") -> dict:
    """Add a free server to the Hive's cloud: check SSH with the Hive's key, install Docker and Node, mark it."""
    from .lab import lab
    t = lab.add_target(f"Hive cloud · {host}", host, 22, user, "~/mnx", None)
    t = await lab.test_target(t["id"])
    if t["status"] != "ok":
        lab.remove_target(t["id"])
        raise CloudError("The Hive couldn't log in. Paste the Hive's SSH key into the server first (shown on this page), "
                         f"and check the user name ({user}): {(t.get('error') or '')[-200:]}")
    code, out = await lab.ssh(t, SETUP, 900)
    text = out.decode(errors="replace") if isinstance(out, bytes) else str(out)
    t["pool"] = True
    t["setup"] = (text.strip().splitlines() or [""])[-1][:200] if code == 0 else f"setup failed: {text.strip()[-200:]}"
    lab._save_targets()
    return t


def pool_view() -> list[dict]:
    from .bees import registry
    return [{"id": t["id"], "name": t["name"], "host": t.get("host"), "status": t.get("status"), "setup": t.get("setup"),
             "ram_gb": round(int((t.get("info") or {}).get("ram_mb") or 0) / 1024, 1), "cpus": (t.get("info") or {}).get("cpus"),
             "slots": slots(t), "bees": [b["name"] for b in registry.bees if b["computer"].get("where") == t["id"]]}
            for t in pool_servers()]
