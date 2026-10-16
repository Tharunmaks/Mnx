"""Iotamine VPSes as the Hive's cloud (a stand-in Iotamine API; no real key or money)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import iotamine as iota_mod  # noqa: E402


class FakeIotamine:
    def __init__(self):
        self.calls, self.keys, self.vps, self.looks = [], [], {}, 0

    def handler(self, req: httpx.Request) -> httpx.Response:
        assert req.headers["authorization"] == "Api-Key test-key-123"
        path, body = req.url.path, json.loads(req.content) if req.content else None
        self.calls.append((req.method, path, body))
        if path == "/api/pop/":
            return httpx.Response(200, json={"count": 1, "results": [{"id": 3, "name": "Mumbai", "cpu_price": "0.5", "ram_price": "0.25",
                                                                      "disk_price": "0.01", "ip_price": "144"}]})
        if path == "/api/os/":
            return httpx.Response(200, json=[{"id": 7, "name": "Debian 12"}, {"id": 9, "name": "Ubuntu 22.04"}, {"id": 11, "name": "Ubuntu 24.04 LTS"}])
        if path == "/api/sshkey/" and req.method == "GET":
            return httpx.Response(200, json={"results": self.keys})
        if path == "/api/sshkey/" and req.method == "POST":
            self.keys.append({"id": 42, **body})
            return httpx.Response(201, json={"id": 42, "title": body["title"]})
        if path == "/api/vps/" and req.method == "POST":
            self.vps["v1"] = {"id": "v1", "hostname": body["hostname"], "cores": body["cores"], "ram": body["ram"], "disk": body["disk"],
                              "is_building": True, "machine_status": "Building", "ips": []}
            return httpx.Response(201, json={"id": "v1", "is_building": True})
        if path == "/api/vps/v1/":
            self.looks += 1
            if self.looks > 1:
                self.vps["v1"].update(is_building=False, machine_status="Running", ips=[{"id": 5, "ip": "103.21.4.9"}])
            return httpx.Response(200, json=self.vps["v1"])
        if path == "/api/vps/":
            return httpx.Response(200, json={"results": [{"id": "u1", "hostname": "basic", "cores": 2, "ram": 8, "disk": 40,
                                                          "machine_status": "Running", "ip_addresses": [{"address": "10.0.0.4"}, {"address": "49.36.1.2"}]}]})
        return httpx.Response(404, json={"message": "not found"})


@pytest.fixture
def iota(monkeypatch):
    fake = FakeIotamine()
    i = iota_mod.Iotamine()
    i.transport, i.poll = httpx.MockTransport(fake.handler), 0.0

    class Lab:
        _secrets = {"IOTAMINE_API_KEY": "test-key-123"}

        @staticmethod
        async def hive_key():
            return "ssh-ed25519 AAAAC3Nza mnx-hive"
    import hive.lab as lab_mod
    monkeypatch.setattr(lab_mod, "lab", Lab)
    return i, fake


def run(coro):
    return asyncio.run(coro)


def test_lists_your_vps_with_its_public_ip(iota):
    i, _ = iota
    (v,) = run(i.servers())
    assert v == {"id": "u1", "name": "basic", "ip": "49.36.1.2", "cores": 2, "ram_gb": 8, "disk_gb": 40, "status": "Running", "building": False}


def test_price_uses_iotamines_own_rates_with_the_monthly_ip(iota):
    i, _ = iota
    q = run(i.quote(2, 8, 40))
    assert q["pop"] == 3 and q["region"] == "Mumbai"
    assert q["hourly"] == round(2 * 0.5 + 8 * 0.25 + 40 * 0.01 + 144 / 720, 4) and q["monthly"] == round(q["hourly"] * 720, 2)


def test_deploy_puts_the_hives_key_on_an_ubuntu_vps_and_waits_for_its_ip(iota):
    i, fake = iota
    v = run(i.deploy("mnx-hive-x", 2, 8, 40))
    assert v["ip"] == "103.21.4.9" and v["status"] == "Running"
    post = next(b for m, p, b in fake.calls if m == "POST" and p == "/api/vps/")
    assert post["operating_system"] == 11 and post["ssh_key"] == 42 and post["disable_pwd_auth"] and post["pop"] == 3
    assert (post["cores"], post["ram"], post["disk"]) == (2, 8, 40)
    run(i.deploy("mnx-hive-y", 2, 8, 40))  # the Hive's key is added to the account once
    assert sum(1 for m, p, _ in fake.calls if m == "POST" and p == "/api/sshkey/") == 1


def test_errors_say_what_to_fix(iota, monkeypatch):
    i, _ = iota
    i.transport = httpx.MockTransport(lambda r: httpx.Response(403, json={"detail": "read only"}))
    with pytest.raises(iota_mod.IotamineError, match="Read & write"):
        run(i.servers())
    import hive.lab as lab_mod
    monkeypatch.setattr(lab_mod.lab, "_secrets", {})
    with pytest.raises(iota_mod.IotamineError, match="Free cloud page"):
        run(i.servers())


def test_the_key_never_reaches_training_code():
    from hive.lab import HIVE_ONLY_SECRETS
    assert "IOTAMINE_API_KEY" in HIVE_ONLY_SECRETS
