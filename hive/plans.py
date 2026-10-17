"""Plans in the chat: "get pro", "upgrade", "what plan am I on". Paying happens on the Pro page (Razorpay).

When Pro starts or ends, every Bee of that user gets the plan's computer (apply).
"""

from __future__ import annotations

import re
import time

from .bees import registry
from .core import Task
from .plan import PLANS, plan

SUBSCRIBE = re.compile(r"\b(?:subscribe|sign\s+me\s+up|upgrade|join|get|buy|switch|move|go|pay\s+for)\b.*\b(?:pro|plus|premium)\b"
                       r"|\b(?:pro|plus|premium)\s+(?:plan|subscription)\b|^\s*upgrade(?:\s+me)?\s*[.!]?\s*$", re.I)
DOWN = re.compile(r"\b(?:unsubscribe|cancel\s+(?:my\s+)?(?:pro|plus|subscription)|(?:switch|go|move)\s+(?:back\s+)?to\s+(?:the\s+)?free(?:\s+plan)?)\b", re.I)
WHICH = re.compile(r"\b(?:which|what)\s+plan\b|\bmy\s+(?:plan|subscription)\b", re.I)


def parse(text: str) -> dict | None:
    t = text or ""
    if re.search(r"\bbees?\b|\bbot\b", t, re.I):
        return None
    if DOWN.search(t):
        return {"kind": "down"}
    if SUBSCRIBE.search(t):
        return {"kind": "up"}
    if WHICH.search(t):
        return {"kind": "which"}
    return None


async def apply(uid: str) -> list[str]:
    """Give every one of the user's Bees the computer their plan has now."""
    return [b["name"] for b in await registry.apply_plan(uid)]


def _date(ts: int) -> str:
    return time.strftime("%d %b %Y", time.localtime(ts))


async def handle(task: Task, intent: dict) -> None:
    v = plan.view(task.user)
    if intent["kind"] == "which" or (intent["kind"] == "up" and v["plan"] == "pro"):
        until = f" until {_date(v['until'])}" if v.get("until") else ""
        await task.emit("result", service="Plan", title=f"You're on {v['label']}{until}", badge="Plan",
                        details=[[p["label"], ("₹0" if p["id"] == "free" else f"₹{p['price_inr']} / 30 days") + f" · {p['computer']} · up to {p['max_bees']} Bees"]
                                 for p in v["plans"]])
        await task.answer(f"You're on {v['label']}{until}. " + ("Every Bee has the Pro computer." if v["plan"] == "pro" else
                          f"Pro is ₹{v['price_inr']} for 30 days: open [Get Pro](pro.html)."))
        return
    if intent["kind"] == "down":
        await task.answer("Pro doesn't renew by itself, so there's nothing to cancel: it ends on its date and your Bees go back to "
                          "Free computers." if v["plan"] == "pro" else "You're on Free already.")
        return
    await task.emit("result", service="Plan", title=f"Pro · ₹{v['price_inr']} for 30 days", badge="Pro",
                    details=[["Every Bee gets", PLANS["pro"]["computer"]], ["Bees", f"up to {PLANS['pro']['max_bees']}"],
                             ["Pay with", "UPI, cards, netbanking or wallets (Razorpay)"]])
    await task.answer("Open [Get Pro](pro.html) to pay. Pro starts the moment the payment goes through.")
