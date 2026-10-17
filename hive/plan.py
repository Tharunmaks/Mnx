"""Plans: Free, or Pro bought with Razorpay (hive/billing.py). Each user has their own plan.

Free  every Bee gets a computer with 2 CPUs, 8 GB RAM and a 40 GB disk.
Pro   every Bee gets 4 CPUs, 16 GB RAM, a 100 GB disk and a GPU with 16 GB VRAM. The Hive gives the GPU (hive/gpu_pool.py).
      More Bees too. Pro lasts 30 days from each payment; paying again adds 30 more.

Computers aren't chosen: the plan decides them. The owner is always on Pro. A hosted Hive can put everyone on one
plan with MNX_PLAN=free|pro. The price is the owner's (python3 -m hive.admin price 499).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from . import users

PLAN_FILE = Path(__file__).resolve().parent.parent / "data" / "plans.json"
DAY = 86400
PRO_DAYS = 30
DEFAULT_PRICE_INR = 499

PLANS = {
    "free": {"label": "Free", "gpu": False, "size": "free", "max_bees": 5,
             "computer": "2 CPUs, 8 GB RAM, 40 GB disk"},
    "pro": {"label": "Pro", "gpu": True, "size": "pro", "max_bees": 20,
            "computer": "4 CPUs, 16 GB RAM, 100 GB disk, GPU with 16 GB VRAM"},
}
ALIASES = {"plus": "pro", "paid": "pro", "premium": "pro", "supergrok": "pro"}


def _norm(name: str | None) -> str:
    n = (name or "").strip().lower()
    return ALIASES.get(n, n)


class Plan:
    """Who is on Pro until when, the payments that paid for it, and the price. Saved to data/plans.json."""

    def __init__(self, path: Path = PLAN_FILE):
        self.path = path
        self._lock = threading.Lock()

    # ----- the file -----
    def _read(self) -> dict:
        try:
            d = json.loads(self.path.read_text())
            return d if isinstance(d, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, d: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    # ----- reading -----
    @property
    def locked(self) -> bool:
        return _norm(os.getenv("MNX_PLAN")) in PLANS

    def until(self, uid: str | None = None) -> int:
        uid = uid or users.who()
        return int(((self._read().get("users") or {}).get(uid) or {}).get("until") or 0)

    def of(self, uid: str | None = None) -> str:
        uid = uid or users.who()
        env = _norm(os.getenv("MNX_PLAN"))
        if env in PLANS:
            return env
        if users.is_owner(uid):
            return "pro"
        return "pro" if self.until(uid) > time.time() else "free"

    @property
    def name(self) -> str:
        """The plan of whoever is asking now."""
        return self.of()

    @property
    def size(self) -> str:
        return PLANS[self.name]["size"]

    def allows_gpu(self, uid: str | None = None) -> bool:
        return PLANS[self.of(uid)]["gpu"]

    def max_bees(self, uid: str | None = None) -> int:
        return PLANS[self.of(uid)]["max_bees"]

    @property
    def price_inr(self) -> int:
        try:
            return max(1, int(self._read().get("price_inr") or os.getenv("MNX_PRO_PRICE_INR") or DEFAULT_PRICE_INR))
        except ValueError:
            return DEFAULT_PRICE_INR

    def pro_users(self) -> list[str]:
        now = time.time()
        return [u for u, v in (self._read().get("users") or {}).items() if int(v.get("until") or 0) > now]

    def expired_users(self) -> list[str]:
        now = time.time()
        return [u for u, v in (self._read().get("users") or {}).items() if 0 < int(v.get("until") or 0) <= now]

    # ----- changing -----
    def set_price(self, inr: int) -> None:
        if not 1 <= int(inr) <= 500000:
            raise ValueError("The price must be between ₹1 and ₹5,00,000")
        with self._lock:
            d = self._read()
            d["price_inr"] = int(inr)
            self._write(d)

    def grant(self, uid: str, days: int = PRO_DAYS, payment: dict | None = None) -> bool:
        """Pro for `days` more (from now, or from the end of the Pro you have). Once per payment: returns False when
        that payment was already counted."""
        with self._lock:
            d = self._read()
            u = d.setdefault("users", {}).setdefault(uid, {})
            pays = u.setdefault("payments", [])
            pid = (payment or {}).get("id")
            if pid and any(p.get("id") == pid for p in pays):
                return False
            start = max(time.time(), float(u.get("until") or 0))
            u["until"] = int(start + days * DAY)
            if payment:
                pays.append({**payment, "at": int(time.time()), "days": days})
                del pays[:-50]
            self._write(d)
            return True

    def revoke(self, uid: str) -> None:
        with self._lock:
            d = self._read()
            u = (d.get("users") or {}).get(uid)
            if u and int(u.get("until") or 0) > 0:
                u["until"] = 1  # in the past: the plan loop gives their Bees Free computers again, then forgets it
                self._write(d)

    def forget_expired(self, uid: str) -> None:
        """Pro ran out and its Bees are back on Free computers: stop reporting it as expired."""
        with self._lock:
            d = self._read()
            u = (d.get("users") or {}).get(uid)
            if u and int(u.get("until") or 0) <= time.time():
                u["until"] = -1 if u.get("payments") else 0
                self._write(d)

    def view(self, uid: str | None = None) -> dict:
        uid = uid or users.who()
        name = self.of(uid)
        until = self.until(uid)
        return {"plan": name, **PLANS[name], "price_inr": self.price_inr, "locked": self.locked,
                "until": until if name == "pro" and until > 0 else None, "owner": users.is_owner(uid),
                "plans": [{"id": k, **v, "price_inr": 0 if k == "free" else self.price_inr} for k, v in PLANS.items()]}


plan = Plan()
