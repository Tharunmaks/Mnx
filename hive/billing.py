"""Paying for Pro with Razorpay.

The owner sets the keys once, on the Hive's machine (never in the code, never by users):
  python3 -m hive.admin set RAZORPAY_KEY_ID           rzp_live_… (or rzp_test_… to try it without real money)
  python3 -m hive.admin set RAZORPAY_KEY_SECRET
  python3 -m hive.admin set RAZORPAY_WEBHOOK_SECRET   optional: Dashboard → Webhooks, URL <your Hive>/razorpay/webhook,
                                                      events payment.captured and order.paid
  python3 -m hive.admin price 499                     ₹ per 30 days of Pro

How a payment works
  1. the Pro page asks the Hive for an order: the Hive creates it on Razorpay for the current price, for this user
  2. Razorpay Checkout opens on the page; the user pays (UPI, card, netbanking, wallet)
  3. the page sends back order id, payment id and signature; the Hive checks the signature with the key secret
     (HMAC-SHA256 of "order_id|payment_id") and gives that user 30 days of Pro
  4. the webhook, when set, does the same from Razorpay's side, so Pro arrives even if the page closed mid-payment
Each payment counts once, whichever arrives first.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path

import httpx

from .plan import PRO_DAYS, plan

API = "https://api.razorpay.com/v1"
ORDERS_FILE = Path(__file__).resolve().parent.parent / "data" / "billing_orders.json"
SECRET_NAMES = ("RAZORPAY_KEY_ID", "RAZORPAY_KEY_SECRET", "RAZORPAY_WEBHOOK_SECRET")


class BillingError(RuntimeError):
    pass


def _secret(name: str) -> str:
    from .lab import lab
    return (lab._secrets.get(name) or os.getenv(name) or "").strip()


class Billing:
    def __init__(self, orders_file: Path = ORDERS_FILE):
        self.orders_file = orders_file
        self.transport: httpx.AsyncBaseTransport | None = None  # tests swap in a fake Razorpay
        self._lock = threading.Lock()

    # ----- setup -----
    @property
    def key_id(self) -> str:
        return _secret("RAZORPAY_KEY_ID")

    def ready(self) -> bool:
        return bool(self.key_id and _secret("RAZORPAY_KEY_SECRET"))

    def test_mode(self) -> bool:
        return self.key_id.startswith("rzp_test_")

    # ----- orders we created, and who they're for -----
    def _orders(self) -> dict:
        try:
            return json.loads(self.orders_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, orders: dict) -> None:
        self.orders_file.parent.mkdir(parents=True, exist_ok=True)
        if len(orders) > 5000:  # keep the newest
            orders = dict(sorted(orders.items(), key=lambda kv: kv[1].get("created", 0))[-5000:])
        tmp = self.orders_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(orders, indent=1))
        os.chmod(tmp, 0o600)
        tmp.replace(self.orders_file)

    # ----- 1. an order for this user -----
    async def create_order(self, uid: str) -> dict:
        if not self.ready():
            raise BillingError("Payments aren't set up on this Hive yet")
        amount = plan.price_inr * 100  # paise
        body = {"amount": amount, "currency": "INR", "receipt": f"pro_{uid}_{int(time.time())}"[:40],
                "notes": {"user": uid, "plan": "pro", "days": str(PRO_DAYS)}}
        try:
            async with httpx.AsyncClient(timeout=30, transport=self.transport) as c:
                r = await c.post(f"{API}/orders", json=body, auth=(self.key_id, _secret("RAZORPAY_KEY_SECRET")))
        except httpx.HTTPError as exc:
            raise BillingError(f"Couldn't reach Razorpay: {exc}") from exc
        if r.status_code == 401:
            raise BillingError("Razorpay rejected the Hive's keys")
        if r.status_code >= 400:
            try:
                msg = r.json()["error"]["description"]
            except (ValueError, KeyError, TypeError):
                msg = r.text[:200]
            raise BillingError(f"Razorpay said: {msg}")
        order = r.json()
        with self._lock:
            orders = self._orders()
            orders[order["id"]] = {"user": uid, "amount": amount, "status": "created", "created": int(time.time())}
            self._save(orders)
        return {"order_id": order["id"], "amount": amount, "currency": "INR", "key_id": self.key_id,
                "days": PRO_DAYS, "price_inr": plan.price_inr, "test": self.test_mode()}

    # ----- 3. the page says it's paid: check the signature -----
    def confirm(self, uid: str, order_id: str, payment_id: str, signature: str) -> dict:
        secret = _secret("RAZORPAY_KEY_SECRET")
        if not secret:
            raise BillingError("Payments aren't set up on this Hive yet")
        with self._lock:
            order = self._orders().get(order_id)
        if not order or order["user"] != uid:
            raise BillingError("That order isn't yours")
        want = hmac.new(secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(want, (signature or "").strip()):
            raise BillingError("The payment's signature didn't match, so Pro wasn't added")
        return self._paid(order_id, payment_id)

    # ----- 4. Razorpay says it's paid -----
    def webhook(self, body: bytes, signature: str) -> dict:
        secret = _secret("RAZORPAY_WEBHOOK_SECRET")
        if not secret:
            raise BillingError("No webhook secret is set on this Hive")
        want = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(want, (signature or "").strip()):
            raise BillingError("Bad webhook signature")
        try:
            event = json.loads(body)
        except ValueError as exc:
            raise BillingError("The webhook wasn't JSON") from exc
        if event.get("event") not in ("payment.captured", "order.paid"):
            return {"ok": True, "ignored": event.get("event")}
        pay = ((event.get("payload") or {}).get("payment") or {}).get("entity") or {}
        order_id, payment_id = pay.get("order_id"), pay.get("id")
        if not order_id or not payment_id:
            return {"ok": True, "ignored": "no order"}
        with self._lock:
            order = self._orders().get(order_id)
        if not order:
            return {"ok": True, "ignored": "not a Hive order"}
        if int(pay.get("amount") or 0) < order["amount"]:
            raise BillingError("Paid less than the order")
        return self._paid(order_id, payment_id)

    def _paid(self, order_id: str, payment_id: str) -> dict:
        with self._lock:
            orders = self._orders()
            order = orders[order_id]
            fresh = plan.grant(order["user"], PRO_DAYS, {"id": payment_id, "order": order_id, "amount": order["amount"]})
            order.update(status="paid", payment=payment_id, paid=order.get("paid") or int(time.time()))
            self._save(orders)
        return {"user": order["user"], "new": fresh, "until": plan.until(order["user"])}

    def view(self) -> dict:
        return {"ready": self.ready(), "key_id": self.key_id if self.ready() else "", "test": self.test_mode(),
                "price_inr": plan.price_inr, "days": PRO_DAYS}


billing = Billing()
