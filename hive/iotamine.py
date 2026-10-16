"""Iotamine (iotamine.com): an Indian VPS host, as a home for the Hive's cloud.

Save your Iotamine API key on the Free cloud page (it's kept as the Hive-only secret IOTAMINE_API_KEY). Then:
  - your VPSes are listed; "Use for the Hive" adds one to the Hive's cloud (it must have the Hive's SSH key:
    either deployed by the Hive, or you add the key once by hand)
  - "Deploy" makes a new VPS with the Hive's SSH key already on it (Ubuntu, the size you pick), after showing the
    price per hour and per month from Iotamine's own rates. Deploying needs a key with "Read & write" scope.
When you upgrade a VPS on Iotamine (say to 10 vCPUs, 20 GB), the Hive sees the new size on its next check and
places more Bees on it.

API: https://iotamine.com/api/ with "Authorization: Api-Key <key>" (vps/, pop/, os/, sshkey/).
"""

from __future__ import annotations

import asyncio
import re
import secrets
import time

import httpx

BASE = "https://iotamine.com/api/"
IPV4 = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


class IotamineError(RuntimeError):
    pass


def _key() -> str:
    from .lab import lab
    k = lab._secrets.get("IOTAMINE_API_KEY", "")
    if not k:
        raise IotamineError("Save your Iotamine API key on the Free cloud page first")
    return k


class Iotamine:
    def __init__(self) -> None:
        self.transport: httpx.AsyncBaseTransport | None = None  # tests swap in a fake Iotamine
        self.poll = 10.0

    async def _call(self, method: str, path: str, params: dict | None = None, body: dict | None = None):
        async with httpx.AsyncClient(base_url=BASE, timeout=60, transport=self.transport,
                                     headers={"Authorization": f"Api-Key {_key()}", "Accept": "application/json"}) as c:
            try:
                r = await c.request(method, path, params=params, json=body)
            except httpx.RequestError as exc:
                raise IotamineError(f"Couldn't reach Iotamine: {exc}") from exc
        if r.status_code == 401:
            raise IotamineError("Iotamine says this API key is invalid, inactive or locked to another IP")
        if r.status_code == 403:
            raise IotamineError("This API key can't do that: on Iotamine → API Keys, set its scope to “Read & write”")
        if not r.is_success:
            try:
                d = r.json()
                msg = d.get("message") or d.get("error") or d.get("detail") or str(d) if isinstance(d, dict) else str(d)
            except ValueError:
                msg = r.text[:300]
            raise IotamineError(f"Iotamine: {msg}"[:400])
        return r.json() if r.content else None

    async def _list(self, path: str, params: dict | None = None) -> list:
        d = await self._call("GET", path, params={"page_size": 200, **(params or {})})
        if isinstance(d, dict) and "results" in d:
            return d["results"]
        return d if isinstance(d, list) else ([] if d is None else [d])

    # ----- what you have -----
    async def servers(self) -> list[dict]:
        return [view(v) for v in await self._list("vps/", {"all": "true"})]

    async def regions(self) -> list[dict]:
        return await self._list("pop/")

    async def images(self) -> list[dict]:
        return await self._list("os/")

    # ----- deploying -----
    async def ubuntu(self) -> dict:
        imgs = await self.images()
        def score(o):
            text = " ".join(str(o.get(k, "")) for k in ("name", "distro", "version", "title")).lower()
            if "ubuntu" not in text:
                return -1
            return 3 if "24.04" in text else 2 if "22.04" in text else 1
        best = max(imgs, key=score, default=None)
        if not best or score(best) < 0:
            raise IotamineError("Iotamine has no Ubuntu image to deploy")
        return best

    async def quote(self, cores: int, ram_gb: int, disk_gb: int, pop: int | None = None) -> dict:
        regions = await self.regions()
        if not regions:
            raise IotamineError("Iotamine listed no regions")
        r = next((x for x in regions if str(x.get("id")) == str(pop)), None) if pop else None
        r = r or regions[0]
        def price(k):
            try:
                return float(r.get(k) or 0)
            except (TypeError, ValueError):
                return 0.0
        hourly = cores * price("cpu_price") + ram_gb * price("ram_price") + disk_gb * price("disk_price") + price("ip_price") / 720
        return {"pop": r.get("id"), "region": r.get("name") or r.get("city") or r.get("location") or str(r.get("id")),
                "cores": cores, "ram_gb": ram_gb, "disk_gb": disk_gb, "hourly": round(hourly, 4), "monthly": round(hourly * 720, 2),
                "ip_monthly": price("ip_price")}

    async def hive_key_id(self, public_key: str):
        """The Hive's SSH key on the Iotamine account (added once)."""
        body = public_key.strip().split()
        for k in await self._list("sshkey/"):
            if (k.get("title") or "") == "mnx-hive" or (k.get("ssh_key") and k["ssh_key"].split()[:2] == body[:2]):
                return k.get("id")
        made = await self._call("POST", "sshkey/", body={"title": "mnx-hive", "ssh_key": public_key.strip()})
        return (made or {}).get("id")

    async def deploy(self, hostname: str, cores: int, ram_gb: int, disk_gb: int, pop: int | None = None,
                     timeout: float = 1200) -> dict:
        """Deploy a VPS with the Hive's SSH key on it and wait for its IP. Returns view() of it."""
        from .lab import lab
        public = await lab.hive_key()
        if not public:
            raise IotamineError("The Hive has no SSH key (ssh-keygen is missing)")
        q = await self.quote(cores, ram_gb, disk_gb, pop)
        os_img = await self.ubuntu()
        key_id = await self.hive_key_id(public)
        made = await self._call("POST", "vps/", body={
            "hostname": hostname, "pop": q["pop"], "cores": cores, "ram": ram_gb, "disk": disk_gb, "traffic": 5,
            "operating_system": os_img.get("id"), "ssh_key": key_id, "disable_pwd_auth": True,
            "password": secrets.token_urlsafe(18) + "A1!"})  # a strong root password nobody needs: login is by key
        vid = (made or {}).get("id")
        if not vid:
            raise IotamineError("Iotamine didn't return the new VPS")
        t0 = time.time()
        while True:
            v = view(await self._call("GET", f"vps/{vid}/"))
            if v["ip"] and not v["building"]:
                return v
            if time.time() - t0 > timeout:
                raise IotamineError(f"The VPS {vid} is still building after {int(timeout / 60)} minutes; check it on iotamine.com")
            await asyncio.sleep(self.poll)


def _find_ip(obj) -> str | None:
    """The VPS's public IPv4, wherever the API puts it."""
    if isinstance(obj, str):
        return obj if IPV4.match(obj) and not obj.startswith(("10.", "192.168.", "127.")) else None
    if isinstance(obj, dict):
        for k in ("ip", "ipv4", "ip_address", "main_ip", "primary_ip", "ips", "ip_addresses", "address"):
            if k in obj:
                hit = _find_ip(obj[k])
                if hit:
                    return hit
        for v in obj.values():
            hit = _find_ip(v)
            if hit:
                return hit
    if isinstance(obj, list):
        for v in obj:
            hit = _find_ip(v)
            if hit:
                return hit
    return None


def view(v: dict) -> dict:
    v = v or {}
    status = str(v.get("machine_status") or v.get("status") or "")
    return {"id": v.get("id"), "name": v.get("hostname") or v.get("name") or str(v.get("id")), "ip": _find_ip(v),
            "cores": v.get("cores"), "ram_gb": v.get("ram"), "disk_gb": v.get("disk"), "status": status,
            "building": bool(v.get("is_building")) or status.lower() == "building"}


iotamine = Iotamine()
