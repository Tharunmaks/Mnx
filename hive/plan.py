"""The Hive's plan: Free or Paid.

Every Bee gets a Standard computer (2 CPUs, 8 GB memory, 40 GB disk) on either plan. The Paid plan adds GPUs:
a GPU on a Bee's computer, here or on a rented cloud GPU machine.

Payments aren't wired in: the owner picks the plan on the Usage page, or locks it with MNX_PLAN=free|paid
(a hosted Hive sets that from its billing and the page can't change it).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PLAN_FILE = Path(__file__).resolve().parent.parent / "data" / "plan.json"

PLANS = {
    "free": {"label": "Free", "gpu": False, "computer": "Standard · 2 CPUs, 8 GB RAM, 40 GB disk"},
    "paid": {"label": "Paid", "gpu": True, "computer": "Standard · 2 CPUs, 8 GB RAM, 40 GB disk, plus GPUs"},
}


class Plan:
    def __init__(self, path: Path = PLAN_FILE):
        self.path = path

    @property
    def locked(self) -> bool:
        return os.getenv("MNX_PLAN", "").strip().lower() in PLANS

    @property
    def name(self) -> str:
        env = os.getenv("MNX_PLAN", "").strip().lower()
        if env in PLANS:
            return env
        try:
            saved = json.loads(self.path.read_text()).get("plan")
        except (FileNotFoundError, json.JSONDecodeError, AttributeError):
            saved = None
        return saved if saved in PLANS else "free"

    def set(self, name: str) -> None:
        name = (name or "").lower()
        if name not in PLANS:
            raise ValueError(f"Unknown plan {name!r}: use free or paid")
        if self.locked:
            raise ValueError("The plan is set by the server (MNX_PLAN) and can't be changed here")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"plan": name}))

    def allows_gpu(self) -> bool:
        return PLANS[self.name]["gpu"]

    def view(self) -> dict:
        return {"plan": self.name, **PLANS[self.name], "locked": self.locked,
                "plans": [{"id": k, **v} for k, v in PLANS.items()]}


plan = Plan()
