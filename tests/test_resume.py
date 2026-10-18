"""A page that reloads picks its running tasks back up: the task keeps going on the Hive, its events reach every page its owner
has open, and a card still waiting for an answer can be answered from the new page."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway import auth  # noqa: E402


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import gateway.api as gw
    monkeypatch.setattr(auth, "USER_TOKENS_FILE", tmp_path / "t.txt")
    (tmp_path / "t.txt").write_text("user-token-a\nuser-token-b\n")

    async def waits_for_you(task):   # a task that stops at a card, like "Which drinks?"
        reply = await task.ask("clarify", title="Which drinks?", ask_anyway=True,
                               fields=[{"name": "what", "label": "What", "required": True, "input": "text"}],
                               actions=[{"id": "go", "label": "Add"}])
        await task.answer(f"Added {reply['values'].get('what')} to the cart.")
    monkeypatch.setattr(gw.queen, "handle", waits_for_you)
    gw.TASK_LOG.clear()
    with TestClient(gw.app) as client:   # one event loop for every connection, as on the real server
        yield client


def _until(ws, kind):
    while True:
        ev = ws.receive_json()
        if ev.get("type") == kind:
            return ev


def test_a_reloaded_page_gets_the_task_back_and_can_answer_its_card(gateway):
    with gateway.websocket_connect("/ws?token=user-token-a") as first:
        first.send_json({"type": "user_message", "task": "t_drinks", "chat": "c1", "text": "add drinks to my blinkit cart"})
        card = _until(first, "clarify")
    # the page reloaded: that connection is gone, the task is still waiting for the card on the Hive
    with gateway.websocket_connect("/ws?token=user-token-a") as again:
        again.send_json({"type": "resume", "tasks": ["t_drinks", "t_gone"]})
        state = _until(again, "task_state")
        assert state["task"] == "t_drinks" and state["running"] is True
        assert [e["type"] for e in state["events"]] == ["clarify"] and state["events"][0]["id"] == card["id"]
        gone = _until(again, "task_state")
        assert gone == {"type": "task_state", "task": "t_gone", "unknown": True}   # e.g. the Hive restarted
        again.send_json({"type": "action", "task": "t_drinks", "ref": card["id"], "action": "go", "values": {"what": "Coca-Cola 2 L"}})
        answer = _until(again, "answer")
        assert answer["text"] == "Added Coca-Cola 2 L to the cart." and _until(again, "task_done")["task"] == "t_drinks"
        again.send_json({"type": "resume", "tasks": ["t_drinks"]})
        done = _until(again, "task_state")
        assert done["running"] is False and [e["type"] for e in done["events"]][-1] == "task_done"


def test_someone_else_cant_see_or_answer_your_task(gateway):
    with gateway.websocket_connect("/ws?token=user-token-a") as mine:
        mine.send_json({"type": "user_message", "task": "t_mine", "chat": "c1", "text": "add drinks to my blinkit cart"})
        card = _until(mine, "clarify")
        with gateway.websocket_connect("/ws?token=user-token-b") as theirs:
            theirs.send_json({"type": "resume", "tasks": ["t_mine"]})
            assert _until(theirs, "task_state") == {"type": "task_state", "task": "t_mine", "unknown": True}
            theirs.send_json({"type": "action", "task": "t_mine", "ref": card["id"], "action": "go", "values": {"what": "theirs"}})
        mine.send_json({"type": "action", "task": "t_mine", "ref": card["id"], "action": "go", "values": {"what": "Sprite"}})
        assert _until(mine, "answer")["text"] == "Added Sprite to the cart."
