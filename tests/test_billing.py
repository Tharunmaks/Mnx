"""Paying for Pro with Razorpay (faked here): orders, the checkout signature, the webhook, and each payment once."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway import auth  # noqa: E402
from hive import billing as billing_mod, users  # noqa: E402
from hive.plan import plan  # noqa: E402

KEY_ID, SECRET, HOOK = "rzp_test_abc123", "rzp-secret-xyz", "hook-secret-789"
USER_TOKEN, OTHER_TOKEN = "user-token-pay", "user-token-other"


class FakeRazorpay:
    def __init__(self):
        self.orders = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/orders" and req.method == "POST"
        if req.headers.get("authorization") != httpx.BasicAuth(KEY_ID, SECRET)._auth_header:
            return httpx.Response(401, json={"error": {"description": "Authentication failed"}})
        body = json.loads(req.content)
        oid = f"order_{len(self.orders) + 1:04d}"
        self.orders.append(body)
        return httpx.Response(200, json={"id": oid, "amount": body["amount"], "currency": body["currency"], "status": "created"})


def sign(order_id: str, payment_id: str, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()


@pytest.fixture
def rzp(monkeypatch):
    import hive.lab as lab_mod
    fake = FakeRazorpay()
    monkeypatch.setattr(lab_mod.lab, "_secrets", {"RAZORPAY_KEY_ID": KEY_ID, "RAZORPAY_KEY_SECRET": SECRET, "RAZORPAY_WEBHOOK_SECRET": HOOK})
    monkeypatch.setattr(billing_mod.billing, "transport", httpx.MockTransport(fake.handler))
    return fake


def run(coro):
    return asyncio.run(coro)


def test_an_order_is_for_this_user_at_todays_price(rzp):
    u = users.id_for_token(USER_TOKEN)
    o = run(billing_mod.billing.create_order(u))
    assert o == {"order_id": "order_0001", "amount": 49900, "currency": "INR", "key_id": KEY_ID, "days": 30, "price_inr": 499, "test": True}
    assert rzp.orders[0]["notes"] == {"user": u, "plan": "pro", "days": "30"}
    plan.set_price(299)
    assert run(billing_mod.billing.create_order(u))["amount"] == 29900
    assert oct(billing_mod.billing.orders_file.stat().st_mode)[-3:] == "600"


def test_the_signature_is_the_proof(rzp):
    u, v = users.id_for_token(USER_TOKEN), users.id_for_token(OTHER_TOKEN)
    o = run(billing_mod.billing.create_order(u))["order_id"]
    with pytest.raises(billing_mod.BillingError, match="signature"):
        billing_mod.billing.confirm(u, o, "pay_1", sign(o, "pay_1", "wrong-secret"))
    with pytest.raises(billing_mod.BillingError, match="isn't yours"):
        billing_mod.billing.confirm(v, o, "pay_1", sign(o, "pay_1"))  # someone else's order can't make you Pro
    assert plan.of(u) == "free"
    r = billing_mod.billing.confirm(u, o, "pay_1", sign(o, "pay_1"))
    assert r["new"] and plan.of(u) == "pro" and plan.of(v) == "free"
    until = plan.until(u)
    assert not billing_mod.billing.confirm(u, o, "pay_1", sign(o, "pay_1"))["new"] and plan.until(u) == until  # once


def test_the_webhook_gives_pro_when_the_page_closed_and_counts_each_payment_once(rzp):
    u = users.id_for_token(USER_TOKEN)
    o = run(billing_mod.billing.create_order(u))["order_id"]
    event = json.dumps({"event": "payment.captured", "payload": {"payment": {"entity": {"id": "pay_9", "order_id": o, "amount": 49900}}}}).encode()
    with pytest.raises(billing_mod.BillingError, match="signature"):
        billing_mod.billing.webhook(event, "forged")
    good = hmac.new(HOOK.encode(), event, hashlib.sha256).hexdigest()
    assert billing_mod.billing.webhook(event, good)["new"] and plan.of(u) == "pro"
    until = plan.until(u)
    assert not billing_mod.billing.confirm(u, o, "pay_9", sign(o, "pay_9"))["new"] and plan.until(u) == until
    cheap = json.dumps({"event": "payment.captured", "payload": {"payment": {"entity": {"id": "pay_10", "order_id": o, "amount": 100}}}}).encode()
    with pytest.raises(billing_mod.BillingError, match="less"):
        billing_mod.billing.webhook(cheap, hmac.new(HOOK.encode(), cheap, hashlib.sha256).hexdigest())
    other = json.dumps({"event": "refund.created"}).encode()
    assert billing_mod.billing.webhook(other, hmac.new(HOOK.encode(), other, hashlib.sha256).hexdigest())["ignored"] == "refund.created"


def test_without_keys_nobody_can_pay(monkeypatch):
    import hive.lab as lab_mod
    monkeypatch.setattr(lab_mod.lab, "_secrets", {})
    for k in billing_mod.SECRET_NAMES:
        monkeypatch.delenv(k, raising=False)
    assert not billing_mod.billing.ready()
    with pytest.raises(billing_mod.BillingError, match="aren't set up"):
        run(billing_mod.billing.create_order("u_x"))


def test_keys_are_the_owners_and_never_reach_training_code():
    from hive.lab import HIVE_ONLY_SECRETS
    from hive.admin import KNOWN
    assert set(billing_mod.SECRET_NAMES) <= HIVE_ONLY_SECRETS and set(billing_mod.SECRET_NAMES) <= set(KNOWN)


def test_paying_through_the_gateway(rzp, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import gateway.api as gw
    monkeypatch.setattr(auth, "USER_TOKENS_FILE", tmp_path / "user_tokens.txt")
    (tmp_path / "user_tokens.txt").write_text(f"{USER_TOKEN}\n{OTHER_TOKEN}\n")
    client = TestClient(gw.app)
    user = {"Authorization": f"Bearer {USER_TOKEN}"}
    status = client.get("/api/billing", headers=user).json()
    assert status["ready"] and status["key_id"] == KEY_ID and SECRET not in json.dumps(status) and status["plan"]["plan"] == "free"
    assert client.post("/api/billing/order", headers={"Authorization": f"Bearer {auth.TOKEN}"}).status_code == 400  # the owner is Pro
    o = client.post("/api/billing/order", headers=user).json()
    assert o["order_id"] == "order_0001" and o["amount"] == 49900
    paid = {"razorpay_order_id": o["order_id"], "razorpay_payment_id": "pay_77", "razorpay_signature": sign(o["order_id"], "pay_77")}
    assert client.post("/api/billing/confirm", headers={"Authorization": f"Bearer {OTHER_TOKEN}"}, json=paid).status_code == 400
    assert client.post("/api/billing/confirm", headers=user, json={**paid, "razorpay_signature": "0" * 64}).status_code == 400
    r = client.post("/api/billing/confirm", headers=user, json=paid)
    assert r.status_code == 200 and r.json()["plan"] == "pro" and r.json()["until"]
    assert client.get("/api/whoami", headers=user).json()["plan"] == "pro"
    assert client.get("/api/whoami", headers={"Authorization": f"Bearer {OTHER_TOKEN}"}).json()["plan"] == "free"
    # the webhook needs Razorpay's signature, not a token
    body = json.dumps({"event": "order.paid", "payload": {"payment": {"entity": {"id": "pay_77", "order_id": o["order_id"], "amount": 49900}}}})
    assert client.post("/razorpay/webhook", content=body, headers={"X-Razorpay-Signature": "bad"}).status_code == 400
    good = hmac.new(HOOK.encode(), body.encode(), hashlib.sha256).hexdigest()
    assert client.post("/razorpay/webhook", content=body, headers={"X-Razorpay-Signature": good}).json() == {"ok": True}
