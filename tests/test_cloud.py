"""Real cloud computers for Bees (Hetzner / DigitalOcean, faked here), the Standard computer and the Free/Paid plan."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import bee_chat, bees, cloud, plan as plan_mod  # noqa: E402
from hive.bees import BeeError, Registry, normalize_computer  # noqa: E402

HCLOUD_TYPES = {"server_types": [
    {"name": "cx22", "cores": 2, "memory": 4.0, "disk": 40, "architecture": "x86", "deprecation": None,
     "prices": [{"location": "sin", "price_hourly": {"gross": "0.0080"}, "price_monthly": {"gross": "4.99"}}]},
    {"name": "cx32", "cores": 4, "memory": 8.0, "disk": 80, "architecture": "x86", "deprecation": None,
     "prices": [{"location": "fsn1", "price_hourly": {"gross": "0.0110"}, "price_monthly": {"gross": "6.80"}},
                {"location": "sin", "price_hourly": {"gross": "0.0140"}, "price_monthly": {"gross": "8.49"}}]},
    {"name": "cax21", "cores": 4, "memory": 8.0, "disk": 80, "architecture": "arm", "deprecation": None,
     "prices": [{"location": "fsn1", "price_hourly": {"gross": "0.0100"}, "price_monthly": {"gross": "5.99"}}]},
    {"name": "cpx31", "cores": 4, "memory": 8.0, "disk": 160, "architecture": "x86", "deprecation": None,
     "prices": [{"location": "sin", "price_hourly": {"gross": "0.0300"}, "price_monthly": {"gross": "18.00"}}]},
]}
DO_SIZES = {"sizes": [
    {"slug": "s-2vcpu-4gb", "vcpus": 2, "memory": 4096, "disk": 80, "price_monthly": 24, "price_hourly": 0.0357, "regions": ["blr1"], "available": True},
    {"slug": "s-4vcpu-8gb", "vcpus": 4, "memory": 8192, "disk": 160, "price_monthly": 48, "price_hourly": 0.0714, "regions": ["blr1", "sgp1"], "available": True},
]}


class FakeCloud:
    """Answers like Hetzner and DigitalOcean; records what was asked."""

    def __init__(self):
        self.calls, self.servers = [], {}

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.calls.append((req.method, req.url.host, req.url.path, json.loads(req.content) if req.content else None))
        assert req.headers["authorization"].startswith("Bearer ")
        path = req.url.path
        if req.url.host == "api.hetzner.cloud":
            if path == "/v1/server_types":
                return httpx.Response(200, json=HCLOUD_TYPES)
            if path == "/v1/servers" and req.method == "POST":
                sid = 1000 + len(self.servers)
                self.servers[str(sid)] = 0
                return httpx.Response(201, json={"server": {"id": sid, "status": "initializing", "public_net": {"ipv4": None}}})
            if path.startswith("/v1/servers/"):
                sid = path.rsplit("/", 1)[1]
                if req.method == "DELETE":
                    self.servers.pop(sid, None)
                    return httpx.Response(200, json={"action": {"id": 1}})
                self.servers[sid] += 1  # running from the second look on
                running = self.servers[sid] > 1
                return httpx.Response(200, json={"server": {"id": int(sid), "status": "running" if running else "starting",
                                                            "public_net": {"ipv4": {"ip": f"203.0.113.{sid[-1]}"}}}})
        if req.url.host == "api.digitalocean.com" and path == "/v2/sizes":
            return httpx.Response(200, json=DO_SIZES)
        return httpx.Response(404, json={"error": {"message": "not found"}})


class FakeLab:
    def __init__(self, secrets):
        self._secrets, self.targets, self.tested = secrets, {}, 0

    async def hive_key(self):
        return "ssh-ed25519 AAAAC3Nza mnx-hive"

    def add_target(self, name, host, port, user, workdir, key):
        t = {"id": f"t{len(self.targets)}", "name": name, "host": host, "status": "unchecked"}
        self.targets[t["id"]] = t
        return t

    async def test_target(self, tid):
        self.tested += 1
        self.targets[tid]["status"] = "ok" if self.tested > 1 else "failed"  # SSH answers on the second try
        return self.targets[tid]

    def remove_target(self, tid):
        self.targets.pop(tid, None)


@pytest.fixture
def world(tmp_path, monkeypatch):
    fake = FakeCloud()
    lab = FakeLab({"HCLOUD_TOKEN": "h-token", "DIGITALOCEAN_TOKEN": "do-token"})
    m = cloud.Machines(tmp_path / "machines.json")
    m.transport, m.poll = httpx.MockTransport(fake.handler), 0.0
    monkeypatch.setattr(m, "_lab", lambda: lab)
    monkeypatch.setattr(cloud, "machines", m)
    from hive import usage
    monkeypatch.setattr(usage.ledger, "save", lambda: None)
    return fake, lab, m


def run(coro):
    return asyncio.run(coro)


def test_standard_computer_is_8gb_ram_40gb_disk():
    c = normalize_computer(None)
    assert (c["size"], c["cpus"], c["memory_gb"], c["disk_gb"]) == ("standard", 2, 8, 40)
    assert "Standard · 2 CPUs, 8 GB RAM, 40 GB disk · Vortex OS" in bees.describe_computer(c)
    assert normalize_computer({"disk_gb": 100}, c)["disk_gb"] == 100


def test_bees_saved_on_the_old_small_default_get_standard(tmp_path):
    state = tmp_path / "hive.json"
    old_small = {"size": "small", "cpus": 1, "memory_gb": 1, "gpu": False, "where": "here", "os": "vortex"}
    state.write_text(json.dumps({"bees": [{"id": "scout", "name": "Scout", "skill": "x", "created_by": "you", "computer": old_small},
                                          {"id": "tiny", "name": "Tiny", "skill": "x", "created_by": "you", "computer_set": True,
                                           "computer": old_small}]}))
    r = Registry(state, tmp_path / "keys.json")
    assert r.get("scout")["computer"]["memory_gb"] == 8 and r.get("scout")["computer"]["disk_gb"] == 40
    assert r.get("tiny")["computer"]["size"] == "small"  # you chose Small on purpose: kept


def test_gpus_need_the_paid_plan(tmp_path, monkeypatch):
    monkeypatch.delenv("MNX_PLAN", raising=False)
    monkeypatch.setattr(plan_mod.plan, "path", tmp_path / "plan.json")
    r = Registry(tmp_path / "hive.json", tmp_path / "keys.json")
    assert plan_mod.plan.name == "free"
    with pytest.raises(BeeError, match="Paid plan"):
        run(r.create("Trainer", "trains", computer={"gpu": True}))
    run(r.create("Scout", "watches"))
    with pytest.raises(BeeError, match="Paid plan"):
        run(r.set_computer("scout", {"gpu": True}))
    plan_mod.plan.set("paid")
    assert run(r.set_computer("scout", {"gpu": True}))["computer"]["gpu"]
    monkeypatch.setenv("MNX_PLAN", "free")  # a hosted Hive locks the plan from its billing
    assert plan_mod.plan.name == "free" and plan_mod.plan.view()["locked"]
    with pytest.raises(ValueError, match="MNX_PLAN"):
        plan_mod.plan.set("paid")


def test_cheapest_type_with_8gb_and_40gb_wins(world):
    fake, lab, m = world
    q = run(m.quote("hetzner"))
    assert q["type"] == "cx32" and q["region"] == "sin" and q["price_month"] == 8.49  # x86 only (no arm), Singapore first
    assert run(m.quote())["provider"] == "hetzner"  # cheaper than DigitalOcean's $48
    d = run(m.quote("digitalocean"))
    assert d["type"] == "s-4vcpu-8gb" and d["region"] == "blr1" and d["memory_gb"] == 8


def test_provision_moves_nothing_until_ssh_works_and_release_stops_billing(world):
    fake, lab, m = world
    q = run(m.quote("hetzner"))
    rec = run(m.provision("scout", q))
    assert rec["status"] == "ready" and rec["ip"] == "203.0.113.0" and rec["target"] in lab.targets
    create = next(c for c in fake.calls if c[0] == "POST")
    assert create[3]["server_type"] == "cx32" and create[3]["location"] == "sin" and create[3]["name"] == "mnx-scout"
    ud = create[3]["user_data"]
    assert "ssh-ed25519 AAAAC3Nza mnx-hive" in ud and "ssh_pwauth: false" in ud and "docker.io" in ud
    assert m.for_bee("scout") and oct((m.path).stat().st_mode)[-3:] == "600"
    with pytest.raises(cloud.CloudError, match="already"):
        run(m.provision("scout", q))
    run(m.release(rec["id"]))
    assert not m.for_bee("scout") and not lab.targets and not fake.servers
    assert any(c[0] == "DELETE" for c in fake.calls)


def test_a_failed_start_deletes_the_server(world):
    fake, lab, m = world

    async def never(tid):
        lab.targets[tid]["status"] = "failed"
        return lab.targets[tid]
    lab.test_target = never
    q = run(m.quote("hetzner"))
    with pytest.raises(cloud.CloudError, match="SSH"):
        run(m.provision("scout", q, timeout=0.05))
    assert not fake.servers and not m.items and not lab.targets  # nothing left running, nothing billed


def test_no_provider_says_how_to_connect(tmp_path, monkeypatch):
    m = cloud.Machines(tmp_path / "machines.json")
    monkeypatch.setattr(m, "_lab", lambda: FakeLab({}))
    with pytest.raises(cloud.CloudError, match="HCLOUD_TOKEN"):
        run(m.quote())


@pytest.mark.parametrize("text,kind,bee,gpu", [
    ("give every bee a real cloud computer", "cloud", None, False),
    ("give each bee its own cloud pc", "cloud", None, False),
    ("give the coding bee a cloud computer with a gpu", "cloud", "coding", True),
    ("give the cloud bee its own cloud computer", "cloud", "cloud", False),
    ("move coding to the cloud", "cloud", "coding", False),
    ("bring the bees back home from the cloud", "cloud_back", None, None),
    ("delete the coding bee's cloud computer", "cloud_back", "coding", None),
])
def test_cloud_phrases(text, kind, bee, gpu):
    i = bee_chat.parse(text)
    assert i["kind"] == kind and i["bee"] == bee and (gpu is None or i["gpu"] == gpu)


def test_cloud_phrases_dont_steal_the_cloud_bee():
    assert bee_chat.parse("give the cloud bee a large computer")["kind"] == "computer"
    assert bee_chat.parse("give the scout bee a cloud computer") == {"kind": "cloud", "bee": None, "missing": "scout"}
    c = bee_chat.parse("create a bee called Scout that watches prices on its own cloud computer")
    assert c["kind"] == "create" and c["where"] == "cloud" and c["skill"] == "watches prices"
