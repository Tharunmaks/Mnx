"""The chat works without WebSocket support: the server says so on /api/ping and serves the same conversation as a stream."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def gw():
    from fastapi.testclient import TestClient
    import gateway.api as g
    from gateway import auth
    return g, TestClient(g.app), {"Authorization": f"Bearer {auth.TOKEN}"}


def test_ping_says_whether_websockets_work(gw, monkeypatch):
    g, client, _ = gw
    assert client.get("/api/ping").json()["ok"] is True
    monkeypatch.setattr(g, "_ws_supported", lambda: False)
    assert client.get("/api/ping").json() == {"ok": True, "ws": False}  # the page then goes straight to the stream


def test_sending_needs_an_open_stream_and_a_token(gw):
    g, client, auth = gw
    assert client.post("/api/live/send?sid=abcdef123", json={"type": "user_message", "text": "hi"}).status_code == 401
    assert client.post("/api/live/send?sid=abcdef123", headers=auth, json={"type": "user_message", "text": "hi"}).status_code == 409
    assert client.post("/api/live/send?sid=x", headers=auth, json={}).status_code == 422  # a short id is refused
    assert client.get("/api/live?sid=abcdef123").status_code == 401  # the stream needs the token too


def test_a_message_over_the_stream_is_answered_in_that_streams_queue(gw):
    g, client, auth = gw
    sse = g.SseClient()
    sse.user = "owner"  # /api/live notes whose stream it is; only they can send into it
    g.LIVE["sess_test1"] = sse
    g.CLIENTS.add(sse)  # as /api/live does: an open page, which the task's events reach
    other = g.SseClient()
    other.user = "u_someoneelse"
    g.LIVE["sess_other"] = other
    assert client.post("/api/live/send?sid=sess_other", headers=auth, json={"type": "user_message", "text": "hi"}).status_code == 409
    g.LIVE.pop("sess_other", None)
    r = client.post("/api/live/send?sid=sess_test1", headers=auth,
                    json={"type": "user_message", "task": "t_sse1", "text": "roll call", "mode": "rush"})
    assert r.status_code == 200

    async def drain():
        got = []
        for _ in range(400):
            try:
                ev = await asyncio.wait_for(sse.queue.get(), 5)
            except asyncio.TimeoutError:
                break
            got.append(ev)
            if ev.get("type") in ("task_done", "task_error"):
                break
        return got
    # the task runs on the app's own loop (the TestClient portal); read what it queued
    deadline = __import__("time").time() + 20
    events = []
    while __import__("time").time() < deadline:
        while not sse.queue.empty():
            events.append(sse.queue.get_nowait())
        if any(e.get("type") in ("task_done", "task_error") for e in events):
            break
        __import__("time").sleep(0.1)
    g.LIVE.pop("sess_test1", None)
    g.CLIENTS.discard(sse)
    assert {e["task"] for e in events if "task" in e} == {"t_sse1"}
    assert any(e["type"] == "answer" for e in events) and events[-1]["type"] == "task_done"


def test_a_reader_that_stops_reading_never_blocks_the_hive():
    import gateway.api as g
    sse = g.SseClient()

    async def flood():
        for i in range(2500):
            await sse.send_json({"i": i})
    asyncio.run(flood())
    assert sse.queue.qsize() == 2000 and sse.queue.get_nowait()["i"] == 500  # the oldest were dropped, the Hive moved on


def test_the_websocket_path_still_works(gw):
    g, client, auth = gw
    from gateway import auth as a
    with client.websocket_connect(f"/ws?token={a.TOKEN}") as ws:
        ws.send_json({"type": "user_message", "task": "t_ws1", "text": "roll call", "mode": "rush"})
        seen = []
        while True:
            ev = ws.receive_json()
            seen.append(ev["type"])
            if ev["type"] in ("task_done", "task_error"):
                break
    assert "answer" in seen and seen[-1] == "task_done"
