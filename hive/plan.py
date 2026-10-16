"""The Hive's plan: Free or Plus. Both cost nothing.

Free  every Bee gets a Standard computer: 2 CPUs, 8 GB RAM, 40 GB disk.
Plus  every Bee gets a Plus computer: 4 CPUs, 16 GB RAM, 100 GB disk and a GPU with 16 GB VRAM. You don't
      bring the GPU: the Hive gives it, from its own GPU pool (hive/gpu_pool.py: a free Kaggle GPU, or a GPU
      on the Hive's own machine).

Subscribing is one tap on the Usage page (no payment), or "subscribe me to plus" in the chat. A hosted Hive can
lock the plan with MNX_PLAN=free|plus.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PLAN_FILE = Path(__file__).resolve().parent.parent / "data" / "plan.json"

PLANS = {
    "free": {"label": "Free", "gpu": False, "size": "standard", "price": "₹0",
             "computer": "Standard · 2 CPUs, 8 GB RAM, 40 GB disk"},
    "plus": {"label": "Plus", "gpu": True, "size": "plus", "price": "₹0",
             "computer": "Plus · 4 CPUs, 16 GB RAM, 100 GB disk, GPU with 16 GB VRAM (the Hive gives the GPU)"},
}
ALIASES = {"paid": "plus", "pro": "plus", "premium": "plus"}  # "paid" was the old name for Plus


class Plan:
    def __init__(self, path: Path = PLAN_FILE):
        self.path = path

    @property
    def locked(self) -> bool:
        env = os.getenv("MNX_PLAN", "").strip().lower()
        return ALIASES.get(env, env) in PLANS

    @property
    def name(self) -> str:
        env = os.getenv("MNX_PLAN", "").strip().lower()
        env = ALIASES.get(env, env)
        if env in PLANS:
            return env
        try:
            saved = json.loads(self.path.read_text()).get("plan")
        except (FileNotFoundError, json.JSONDecodeError, AttributeError):
            saved = None
        saved = ALIASES.get(saved, saved)
        return saved if saved in PLANS else "free"

    def set(self, name: str) -> None:
        name = (name or "").lower()
        name = ALIASES.get(name, name)
        if name not in PLANS:
            raise ValueError(f"Unknown plan {name!r}: use free or plus")
        if self.locked:
            raise ValueError("The plan is set by the server (MNX_PLAN) and can't be changed here")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"plan": name}))

    def allows_gpu(self) -> bool:
        return PLANS[self.name]["gpu"]

    @property
    def size(self) -> str:
        """The computer every Bee gets on this plan, unless you chose another for it."""
        return PLANS[self.name]["size"]

    def view(self) -> dict:
        return {"plan": self.name, **PLANS[self.name], "locked": self.locked,
                "plans": [{"id": k, **v} for k, v in PLANS.items()]}


plan = Plan()
