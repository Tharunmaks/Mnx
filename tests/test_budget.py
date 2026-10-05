"""The GPU Bee's budget planner: money parsing, what a budget buys, the pick, and live prices."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import budget, cloud_chat, providers  # noqa: E402
from hive.core import Task  # noqa: E402


@pytest.mark.parametrize("text,amount,code", [
    ("I have 1500 INR", 1500, "INR"), ("₹2k budget", 2000, "INR"), ("1.5 lakh rupees", 150000, "INR"), ("Rs. 750 only", 750, "INR"),
    ("$20, what can I train?", 20, "USD"), ("40 dollars", 40, "USD"), ("15 euros", 15, "EUR"), ("£10", 10, "GBP"), ("1,200 inr", 1200, "INR"),
])
def test_parse_money(text, amount, code):
    got = budget.parse_money(text)
    assert got and got[0] == amount and got[1] == code


@pytest.mark.parametrize("text", ["I have 1500 inroads in race engineer", "train a 10M model", "run my model on 8 lpu chips"])
def test_no_money_where_there_is_none(text):
    assert budget.parse_money(text) is None


def test_runpod_names_map_to_speeds():
    assert budget.speed_of("RTX 4090") == 60e12
    assert budget.speed_of("H100 PCIe") == 265e12 and budget.speed_of("H100 SXM") == 350e12
    assert budget.speed_of("A100 SXM") == 110e12 and budget.speed_of("L40S") == 120e12 and budget.speed_of("L40") == 63e12
    assert budget.speed_of("Mystery Accelerator") is None


def test_options_follow_the_arithmetic():
    gpu = {"name": "X 24GB", "memory_gb": 24, "speed": 60e12, "price": 0.5, "live": False}
    (o,) = budget.options(10.0, [gpu])
    assert o["hours"] == pytest.approx(20.0)
    assert o["work_hours"] == pytest.approx(20.0 - budget.SETUP_HOURS)
    assert o["flops"] == pytest.approx(60e12 * o["work_hours"] * 3600)
    assert o["pretrain_params"] == pytest.approx((o["flops"] / 120) ** 0.5)  # 6 N · 20 N = flops
    assert o["lora_max"] == pytest.approx((24 * 0.9 - 2) * 1e9 / 2.5)


def test_memory_caps_pretraining():
    gpu = {"name": "Small 16GB", "memory_gb": 16, "speed": 1e15, "price": 0.1, "live": False}
    (o,) = budget.options(1000.0, [gpu])
    assert o["pretrain_limit"] == "memory" and o["pretrain_params"] == pytest.approx(16e9 * 0.85 / 18)


def test_recommend_prefers_cheaper_when_compute_is_close_and_respects_memory():
    cheap = {"name": "Cheap 24GB", "memory_gb": 24, "speed": 60e12, "price": 0.45, "live": False}
    fast = {"name": "Fast 80GB", "memory_gb": 80, "speed": 350e12, "price": 2.5, "live": False}
    opts = budget.options(17.65, [cheap, fast])
    assert budget.recommend(opts)["name"] == "Cheap 24GB"
    assert budget.recommend(opts, need_gb=40)["name"] == "Fast 80GB"
    assert budget.recommend(budget.options(0.5, [fast])) is None  # less than an hour of real work


def test_pretrain_cost():
    gpu = {"name": "Fast 80GB", "memory_gb": 80, "speed": 350e12, "price": 2.5}
    c = budget.pretrain_cost(1e9, gpu)
    assert c["fits"] and c["hours"] == pytest.approx(6 * 1e9 * 20e9 / 350e12 / 3600 + budget.SETUP_HOURS)
    assert not budget.pretrain_cost(1e10, gpu)["fits"]


@pytest.mark.parametrize("text", ["I have 1500 inr which gpu should I rent", "₹2k budget to fine-tune a 7B model", "$20, what can I train?",
                                  "with 1.5 lakh rupees can I pretrain a 1B model", "I have 3k rupees", "which gpu should I rent"])
def test_budget_questions_reach_the_gpu_bee(text):
    assert cloud_chat.parse(text).kind == "gpu_budget"


def test_stop_renting_is_not_a_budget_question():
    assert cloud_chat.parse("stop renting the gpu").kind == "rent_stop"


def _run(text: str) -> list[dict]:
    events: list[dict] = []

    async def emit(ev):
        events.append(ev)

    class T(Task):
        async def ask(self, card_type, **card):
            await self.emit(card_type, **card)
            return {"action": card["actions"][0]["id"], "values": {}}

    asyncio.run(cloud_chat.handle(T("t", "c", text, emit)))
    return events


def test_chat_answer_with_the_built_in_table(monkeypatch):
    from hive.lab import lab
    monkeypatch.setattr(lab, "_secrets", {})
    events = _run("I have 1500 inr which gpu should I rent")
    card = next(e for e in events if e["type"] == "result")
    answer = next(e for e in events if e["type"] == "answer")["text"]
    assert "approximate" in card["title"] and len(card["details"]) >= 5
    assert answer.startswith("Rent the ") and "stop renting" in answer and "RUNPOD_API_KEY" in answer


def test_chat_uses_live_runpod_prices(monkeypatch):
    from hive.lab import lab
    monkeypatch.setattr(lab, "_secrets", {"RUNPOD_API_KEY": "test"})

    async def fake_types(key):
        return [{"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "memory_gb": 24, "price": 0.34, "stock": "High"},
                {"id": "NVIDIA H100 80GB HBM3", "name": "H100 SXM", "memory_gb": 80, "price": 2.99, "stock": "Low"},
                {"id": "AMD MI300X", "name": "MI300X", "memory_gb": 192, "price": 2.49, "stock": "Low"}]

    monkeypatch.setattr(providers, "runpod_gpu_types", fake_types)
    events = _run("$20 which gpu")
    card = next(e for e in events if e["type"] == "result")
    names = [row[0] for row in card["details"]]
    assert "approximate" not in card["title"] and names == ["RTX 4090 24GB", "H100 SXM 80GB"]  # unknown speeds are left out
    assert "$0.34/h" in card["details"][0][1]
    assert "RUNPOD_API_KEY" not in next(e for e in events if e["type"] == "answer")["text"]
