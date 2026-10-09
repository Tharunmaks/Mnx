"""Usage: the ledger, the measurements and the /api/usage report."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import usage  # noqa: E402
from hive.usage import Ledger  # noqa: E402

DAY = 86400


def noon(days_ago: int = 0) -> float:
    t = time.localtime()
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 12, 0, 0, 0, 0, -1)) - days_ago * DAY


def test_ledger_adds_up_messages_per_day_and_effort(tmp_path):
    lg = Ledger(tmp_path / "usage.json")
    lg.record_task({"in": 10, "out": 90, "total": 100, "budget": 4000, "effort": "med"}, noon())
    lg.record_task({"in": 20, "out": 5000, "total": 5020, "budget": 4000, "effort": "med"}, noon())
    lg.record_task({"in": 5, "out": 5, "total": 10, "budget": 1000, "effort": "low"}, noon())
    lg.record_task({"in": 0, "out": 0, "total": 0, "budget": 4000, "effort": "med"}, noon())  # nothing was counted: not a message
    d = lg.days[usage.day_key(noon())]
    assert d["messages"] == 3 and d["in"] == 35 and d["out"] == 5095 and d["peak"] == 5020 and d["over_budget"] == 1
    assert d["efforts"]["med"] == {"messages": 2, "tokens": 5120, "peak": 5020, "budget": 4000}
    assert Ledger(tmp_path / "usage.json").days == lg.days  # saved


def test_summary_windows(tmp_path):
    lg = Ledger(tmp_path / "usage.json")
    for ago, total in [(0, 100), (1, 200), (6, 300), (7, 400), (20, 500)]:
        lg.record_task({"in": total, "out": 0, "total": total, "budget": 4000, "effort": "med"}, noon(ago))
    s = usage.token_summary(lg, usage.day_key(noon()))
    assert s["today"]["tokens"] == 100 and s["week"]["tokens"] == 600 and s["week"]["messages"] == 3
    assert s["all"]["tokens"] == 1500 and s["all"]["messages"] == 5 and s["all"]["since"] == usage.day_key(noon(20))
    assert len(s["series"]) == 14 and s["series"][-1]["tokens"] == 100 and s["series"][-2]["tokens"] == 200
    assert [d["tokens"] for d in s["series"]].count(0) == 14 - 4  # the day-20 message is outside the 14 days
    assert len({d["day"] for d in s["series"]}) == 14


def test_old_days_are_dropped(tmp_path):
    lg = Ledger(tmp_path / "usage.json")
    for ago in range(usage.KEEP_DAYS + 10):
        lg.days[usage.day_key(noon(ago))] = {"messages": 1, "in": 1, "out": 0, "peak": 1, "over_budget": 0, "efforts": {}}
    lg.record_task({"in": 1, "out": 1, "total": 2, "budget": 1000, "effort": "low"}, noon())
    assert len(lg.days) == usage.KEEP_DAYS


def test_rental_hours_and_cost():
    now = 1_000_000.0
    r = usage.rental_now({"started": now - 5400, "cost_per_hour": 0.4, "gpu_type": "RTX 4090", "count": 2}, now)
    assert r["hours"] == 1.5 and r["cost"] == 0.6 and r["count"] == 2 and r["gpu"] == "RTX 4090"
    assert usage.rental_now({"started": now - 3600}, now)["cost"] is None  # no price recorded: no invented cost


def test_stopping_a_rental_keeps_its_cost(tmp_path):
    lg = Ledger(tmp_path / "usage.json")
    out = lg.record_rental({"id": "pod1", "started": 1000, "cost_per_hour": 0.5, "gpu_type": "A40", "count": 1}, 1000 + 7200)
    assert out["hours"] == 2.0 and out["cost"] == 1.0 and Ledger(tmp_path / "usage.json").rentals[0]["id"] == "pod1"


@pytest.mark.parametrize("text,want", [("12.3MiB", 12897484), ("1GiB", 1073741824), ("512kB", 512000), ("0B", 0), ("2.5GB", 2500000000)])
def test_parse_size(text, want):
    assert usage.parse_size(text) == want


def test_parse_size_rejects_nonsense():
    assert usage.parse_size("") is None and usage.parse_size("lots") is None and usage.parse_size("12 parsecs") is None


def test_parse_docker_stats_keeps_only_cells():
    text = "mnx-cell-scout\t12.50%\t30.5MiB / 1GiB\nsomething-else\t99%\t1GiB / 2GiB\nmnx-cell-coding\t--\t4MiB / 16GiB\n"
    got = usage.parse_docker_stats(text)
    assert set(got) == {"mnx-cell-scout", "mnx-cell-coding"}
    assert got["mnx-cell-scout"] == {"cpu": 12.5, "mem": 31981568, "mem_limit": 1073741824}
    assert got["mnx-cell-coding"]["cpu"] is None and got["mnx-cell-coding"]["mem_limit"] == 17179869184


def test_run_seconds():
    assert usage.run_seconds({"started": 100, "ended": 160}, 999) == 60
    assert usage.run_seconds({"started": 100}, 190) == 90 and usage.run_seconds({}, 190) == 0


def test_dir_size(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.bin").write_bytes(b"x" * 1000)
    (tmp_path / "y.bin").write_bytes(b"y" * 24)
    assert usage.dir_size(tmp_path) == 1024


def test_usage_endpoint(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import gateway.api as gw
    from gateway import auth
    lg = Ledger(tmp_path / "usage.json")
    lg.record_task({"in": 40, "out": 60, "total": 100, "budget": 4000, "effort": "med"})
    monkeypatch.setattr(usage, "ledger", lg)

    async def no_docker():
        return {}

    monkeypatch.setattr(gw, "_docker_stats", no_docker)
    client = TestClient(gw.app)
    assert client.get("/api/usage").status_code == 401
    u = client.get("/api/usage", headers={"Authorization": f"Bearer {auth.TOKEN}"}).json()
    assert u["tokens"]["today"]["tokens"] == 100 and u["tokens"]["today"]["messages"] == 1
    assert u["bees"]["limit"] == 64 and u["bees"]["count"] == len(u["bees"]["rows"]) >= 11
    assert all(r["cpu_percent"] is None and r["measured"] is False for r in u["bees"]["rows"])  # not measured, not invented
    assert u["limits"]["efforts"]["med"]["tokens"] == 4000 and u["limits"]["efforts"]["maxxxx"]["tokens"] == 76800
    assert u["server"]["disk"]["total"] > 0 and u["rentals"]["active_cost"] >= 0 and "invoice" in u["rentals"]["note"]


@pytest.mark.parametrize("text,yes", [
    ("show my usage", True), ("how many tokens have I used", True), ("what are my limits", True), ("usage and limits", True),
    ("how much have I used today", True), ("check my limits", True),
    ("I have 1500 inr which gpu should I rent", False), ("create a bee called Scout that watches prices", False),
    ("run my stories model on the lpu", False), ("list my bees", False), ("how is training going", False)])
def test_usage_phrases(text, yes):
    from hive import cloud_chat
    i = cloud_chat.parse(text)
    assert (i is not None and i.kind == "usage") == yes, (text, i)


def test_usage_in_the_chat(monkeypatch, tmp_path):
    import asyncio
    from hive import cloud_chat
    from hive.core import Task
    lg = Ledger(tmp_path / "usage.json")
    lg.record_task({"in": 40, "out": 60, "total": 100, "budget": 4000, "effort": "med"})
    monkeypatch.setattr(usage, "ledger", lg)
    events = []

    async def emit(ev):
        events.append(ev)

    assert asyncio.run(cloud_chat.handle(Task("t", "c", "show my usage", emit, effort="high")))
    card = next(e for e in events if e["type"] == "result")
    rows = dict(card["details"])
    assert "100 in 1 message" in rows["Tokens today"] and "16,000 tokens at High" in rows["Limit per message"] and "of 64" in rows["Bees"]
    assert "nothing billing" in rows["Rented GPUs"]
