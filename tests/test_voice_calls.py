"""Fish Audio voice, calls (in the app and through Twilio), and the Rush / Peace modes."""

from __future__ import annotations

import asyncio
import json
import struct
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import calls as calls_mod, modes, voice as voice_mod  # noqa: E402
from hive.calls import Calls, card_answer, card_prompt, parse_call, twilio_signature_ok  # noqa: E402
from hive.voice import Voice, VoiceError, packb, speakable  # noqa: E402


def unpackb(b: bytes):
    """Just enough MessagePack to read back what packb wrote."""
    def one(i):
        t = b[i]
        if t <= 0x7F:
            return t, i + 1
        if 0x80 <= t <= 0x8F:
            out, i = {}, i + 1
            for _ in range(t & 0x0F):
                k, i = one(i)
                out[k], i = one(i)
            return out, i
        if 0xA0 <= t <= 0xBF:
            n = t & 0x1F
            return b[i + 1:i + 1 + n].decode(), i + 1 + n
        if t == 0xC0:
            return None, i + 1
        if t in (0xC2, 0xC3):
            return t == 0xC3, i + 1
        if t == 0xC4:
            n = b[i + 1]
            return b[i + 2:i + 2 + n], i + 2 + n
        if t == 0xC5:
            n = struct.unpack(">H", b[i + 1:i + 3])[0]
            return b[i + 3:i + 3 + n], i + 3 + n
        if t == 0xC6:
            n = struct.unpack(">I", b[i + 1:i + 5])[0]
            return b[i + 5:i + 5 + n], i + 5 + n
        if t == 0xD9:
            n = b[i + 1]
            return b[i + 2:i + 2 + n].decode(), i + 2 + n
        raise ValueError(hex(t))
    return one(0)[0]


class FakeFish:
    def __init__(self, status=200):
        self.requests, self.status = [], status

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        if self.status != 200:
            return httpx.Response(self.status, text="nope")
        if req.url.path == "/v1/tts":
            body = json.loads(req.content)
            return httpx.Response(200, content=b"ID3" + body["text"].encode(), headers={"content-type": "audio/mpeg"})
        if req.url.path == "/v1/asr":
            data = unpackb(req.content)
            return httpx.Response(200, json={"text": data["audio"].decode(), "duration": 1.2, "segments": []})
        if req.url.path == "/model":
            return httpx.Response(200, json={"total": 1, "items": [{"_id": "abc123voice", "title": "Calm Indian English", "languages": ["en"]}]})
        return httpx.Response(404)


@pytest.fixture
def fish(tmp_path, monkeypatch):
    v = Voice(tmp_path / "voice.json")
    fake = FakeFish()
    v.transport = httpx.MockTransport(fake.handler)
    v.update({"fish_key": "fk-secret", "voice_id": "abc123voice", "model": "s1"})
    monkeypatch.setattr(voice_mod, "voice", v)
    monkeypatch.setattr(calls_mod, "voice", v)
    return v, fake


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- Fish Audio
def test_messagepack_roundtrip():
    data = {"audio": b"\x00\x01" * 300, "language": "en", "ignore_timestamps": True, "x": None}
    assert unpackb(packb(data)) == data


def test_fish_tts_sends_the_voice_and_model_and_returns_mp3(fish):
    v, fake = fish
    audio = run(v.tts("Hello **there**, see https://x.io and `ls`"))
    req = fake.requests[-1]
    body = json.loads(req.content)
    assert req.url.host == "api.fish.audio" and req.headers["authorization"] == "Bearer fk-secret" and req.headers["model"] == "s1"
    assert body["reference_id"] == "abc123voice" and body["format"] == "mp3" and body["text"] == "Hello there , see the link on screen and ls"
    assert audio.startswith(b"ID3")


def test_fish_asr_uses_messagepack(fish):
    v, fake = fish
    assert run(v.asr(b"take a screenshot")) == "take a screenshot"
    req = fake.requests[-1]
    assert req.headers["content-type"] == "application/msgpack" and unpackb(req.content)["language"] == "en"


def test_fish_errors_and_settings_privacy(fish, tmp_path):
    v, fake = fish
    fake.status = 402
    with pytest.raises(VoiceError, match="out of credit"):
        run(v.tts("hi"))
    view = v.view()
    assert "fish_key" not in view and view["fish_key_set"] and "fk-secret" not in json.dumps(view)
    assert oct((tmp_path / "voice.json").stat().st_mode)[-3:] == "600"
    v.update({"fish_key": ""})  # an empty secret field keeps the key
    assert v.fish_ready
    with pytest.raises(VoiceError, match="country code"):
        v.update({"phone": "98765"})
    assert v.update({"phone": "+91 98765 43210"})["phone"] == "+919876543210"
    assert not v.view()["twilio_ready"] and "Twilio" in v.view()["phone_calls"]


def test_speakable_trims_long_text():
    long = "Sentence one is here. " * 200
    out = speakable(long)
    assert len(out) <= voice_mod.MAX_TTS_CHARS and out.endswith(".")
    assert speakable("```py\nprint(1)\n```done") == "(code on screen) done"


# ---------------------------------------------------------------- "call me …"
NOW = datetime(2026, 10, 16, 8, 0, tzinfo=ZoneInfo("Asia/Kolkata"))


@pytest.mark.parametrize("text,minutes,reason,request_,channel", [
    ("call me", None, "", "", "auto"),
    ("call me in 10 minutes", 10, "", "", "auto"),
    ("give me a call in half an hour", 30, "", "", "auto"),
    ("call me at 9am and remind me to check the training", 60, "This is your reminder: check the training", "", "auto"),
    ("call me at 7am", 23 * 60, "", "", "auto"),
    ("call me on my phone at 7:30 pm tomorrow", 35 * 60 + 30, "", "", "phone"),
    ("call me and tell me how my training is going", None, "how my training is going", "how my training is going", "auto"),
])
def test_call_phrases(text, minutes, reason, request_, channel):
    p = parse_call(text, now_dt=NOW)
    got = round((p["at"] - NOW).total_seconds() / 60) if p["at"] else None
    assert (got, p["reason"], p["request"], p["channel"]) == (minutes, reason, request_, channel)


def test_not_a_call():
    assert parse_call("talk to my model", now_dt=NOW) is None and parse_call("call the api", now_dt=NOW) is None
    assert parse_call("talk to me", now_dt=NOW)["talk"]


CARD = {"id": "e9", "type": "confirm", "title": "Rent 2 cloud computers?", "text": "This spends real money. More here.",
        "details": [["Each", "Hetzner cx32"], ["Total", "€16.98 per month"]],
        "actions": [{"id": "go", "label": "Rent them"}, {"id": "cancel", "label": "Cancel"}]}


def test_cards_are_read_out_and_answered_by_voice():
    p = card_prompt(CARD)
    assert p.startswith("Rent 2 cloud computers?") and "This spends real money." in p and "More here" not in p
    assert "Total: €16.98 per month." in p and "Say yes to rent them, or no." in p
    assert card_answer(CARD, "yes please") == ("go", {})
    assert card_answer(CARD, "no") == ("cancel", {})
    assert card_answer(CARD, "rent them") == ("go", {})
    assert card_answer(CARD, "what's the weather") is None
    choose = {"id": "e3", "type": "clarify", "title": "New Bee", "fields": [
        {"name": "size", "label": "Its computer", "input": "select", "value": "standard",
         "options": [{"value": "standard", "label": "Standard · 2 CPUs"}, {"value": "large", "label": "Large · 4 CPUs"}]},
        {"name": "name", "label": "Name", "value": "Scout"}],
        "actions": [{"id": "create", "label": "Create Bee"}, {"id": "cancel", "label": "Cancel"}]}
    assert card_answer(choose, "make it large") == ("create", {"size": "large", "name": "Scout"})
    typing = {**choose, "fields": [{"name": "host", "label": "Server address", "required": True}]}
    assert "needs typing" in card_prompt(typing) and card_answer(typing, "yes") is None


# ---------------------------------------------------------------- a call in the app
@pytest.fixture
def phone_calls(tmp_path, fish):
    c = Calls(tmp_path / "calls.json")
    events = []

    async def broadcast(ev):
        events.append(ev)
    c.broadcast = broadcast
    c.ring_seconds = 0.2
    return c, events


async def _ask_then_answer(task):
    """A request that needs a yes: like renting computers."""
    reply = await task.ask("confirm", **{k: v for k, v in CARD.items() if k not in ("id", "type")})
    await task.answer("Rented." if reply["action"] == "go" else "Okay, nothing rented.")


def test_app_call_rings_answers_speaks_with_fish_and_does_what_you_say(phone_calls):
    c, events = phone_calls

    async def scenario():
        c.run = _ask_then_answer
        call = await c.start(None, "This is your reminder: stretch")
        assert call.state == "ringing" and call.via == "app"
        await c.answer(call.id)
        assert call.state == "live" and "stretch" in call.transcript[0]["text"] and call.transcript[0]["audio"]
        assert c.audio_for(call.id, 1).startswith(b"ID3")
        await c.heard(call.id, "rent two computers")
        for _ in range(50):
            if call.card:
                break
            await asyncio.sleep(0.01)
        assert call.card and "Say yes" in call.transcript[-1]["text"]
        await c.heard(call.id, "yes")
        await asyncio.wait_for(call.runner, 2)
        assert call.transcript[-1]["text"] == "Rented." and not call.working
        await c.heard(call.id, "okay bye")
        assert call.state == "ended" and call.transcript[-1]["text"] == "Okay, bye!"
    run(scenario())
    kinds = [e["type"] for e in events]
    assert "call_said" in kinds and "call_heard" in kinds and "call_event" in kinds


def test_unanswered_calls_are_missed_and_scheduled_calls_survive_a_restart(phone_calls, tmp_path):
    c, events = phone_calls

    async def scenario():
        call = await c.start(None, "hi")
        await asyncio.sleep(0.35)
        assert call.state == "missed"
        later = await c.start("chat", "", at=datetime.now().timestamp() + 3600)
        assert later.state == "scheduled"
        again = Calls(tmp_path / "calls.json")
        assert later.id in again.calls and again.calls[later.id].state == "scheduled"
        again.broadcast = c.broadcast
        again.ring_seconds = 5
        rung = await again.due(datetime.now().timestamp() + 7200)
        assert [x.id for x in rung] == [later.id] and again.calls[later.id].state == "ringing"
    run(scenario())


def test_talk_mode_starts_picked_up(phone_calls):
    c, _ = phone_calls

    async def scenario():
        call = await c.start(None, answer=True, talk=True)
        assert call.state == "live" and "listening" in call.transcript[0]["text"]
    run(scenario())


# ---------------------------------------------------------------- a real phone call (Twilio)
def test_twilio_signature_matches_twilios_own_example():
    url = "https://mycompany.com/myapp.php?foo=1&bar=2"
    params = {"CallSid": "CA1234567890ABCDE", "Caller": "+12349013030", "Digits": "1234", "From": "+12349013030", "To": "+18005551212"}
    assert twilio_signature_ok(url, params, "0/KCTR6DLpKmkAf8muzZqo1nDgQ=", "12345")
    assert not twilio_signature_ok(url, {**params, "Digits": "9"}, "0/KCTR6DLpKmkAf8muzZqo1nDgQ=", "12345")


def test_phone_call_through_twilio_plays_fish_audio_and_listens(phone_calls, fish, monkeypatch):
    c, _ = phone_calls
    v, _ = fish
    monkeypatch.setenv("MNX_PUBLIC_URL", "https://hive.example.com")
    v.update({"phone": "+919876543210", "twilio_sid": "AC123", "twilio_token": "tw-secret", "twilio_from": "+15005550006"})
    placed = []

    def twilio(req: httpx.Request) -> httpx.Response:
        placed.append(req)
        return httpx.Response(201, json={"sid": "CA999", "status": "queued"})
    c.transport = httpx.MockTransport(twilio)

    async def scenario():
        async def answer_it(task):
            await task.answer("Your training is at step 400.")
        c.run = answer_it
        call = await c.start(None, request="how is my training going")
        assert call.state == "ringing" and call.via == "phone" and call.sid == "CA999"
        form = dict(x.split("=", 1) for x in placed[0].content.decode().split("&"))
        assert form["To"] == "%2B919876543210" and "hive.example.com%2Fvoice%2Ftwilio%2F" in form["Url"]
        xml = await c.twilio(call, "start", {})
        assert call.state == "live" and f"/voice/twilio/{call.id}/{call.secret}/audio/1.mp3" in xml
        assert "<Redirect" in xml or "<Gather" in xml
        await asyncio.wait_for(call.runner, 2)
        xml = await c.twilio(call, "wait", {})
        assert "/audio/2.mp3" in xml and "<Gather" in xml and 'language="en-IN"' in xml
        xml = await c.twilio(call, "heard", {"SpeechResult": "bye"})
        assert "<Hangup/>" in xml and call.state == "ended"
    run(scenario())


def test_phone_only_call_without_twilio_fails_clearly(phone_calls):
    c, _ = phone_calls

    async def scenario():
        call = await c.start(None, channel="phone")
        assert call.state == "failed" and "Twilio" in call.error
    run(scenario())


# ---------------------------------------------------------------- Rush and Peace
@pytest.mark.parametrize("text,cmd", [("$ ls -la", "ls -la"), ("run: df -h", "df -h"), ("run `uname -a`", "uname -a"), ("run my model", None)])
def test_peace_shell_commands(text, cmd):
    assert modes.shell_command(text) == cmd


@pytest.mark.parametrize("text,kind", [
    ("everyone run pip install numpy", "run"), ("all bees run: df -h", "run"), ("every bee $ uname -a", "run"),
    ("roll call", "roll"), ("everyone report", "roll"), ("what is everyone doing", "roll"),
    ("run my model", None), ("give every bee a real cloud computer", None),
])
def test_rush_commands(text, kind):
    assert modes.rush_kind(text) == kind


# ---------------------------------------------------------------- through the gateway
@pytest.fixture
def gateway(monkeypatch, tmp_path, fish):
    from fastapi.testclient import TestClient
    import gateway.api as gw
    from gateway import auth
    from hive import bee_chat, bees, cells as cells_mod, queen
    r = bees.Registry(tmp_path / "hive.json", tmp_path / "bee_keys.json")

    async def nothing(*a):
        return None
    r.on_created = r.on_deleted = r.on_computer = nothing
    for mod in (gw, modes, queen, bee_chat, calls_mod):
        monkeypatch.setattr(mod, "registry", r)
    monkeypatch.setattr(cells_mod, "CELLS", tmp_path / "cells")
    monkeypatch.setattr(cells_mod.cells, "cells", {})
    monkeypatch.setattr(cells_mod.cells, "mode", "local")
    c = Calls(tmp_path / "calls.json")
    monkeypatch.setattr(gw, "calls", c)
    monkeypatch.setattr(calls_mod, "calls", c)
    monkeypatch.setattr(gw, "voice", fish[0])
    run(r.create("Scout", "watches prices"))
    return TestClient(gw.app), auth.TOKEN, r, c


def _until_done(ws, answer=None):
    events = []
    while True:
        ev = ws.receive_json()
        events.append(ev)
        if answer and ev.get("type") == "confirm":
            ws.send_json({"type": "action", "task": ev["task"], "ref": ev["id"], "action": answer, "values": {}})
        if ev.get("type") in ("task_done", "task_error"):
            return events


def test_peace_chat_runs_on_the_bees_own_computer_and_answers_as_the_bee(gateway):
    client, token, r, _ = gateway
    with client.websocket_connect(f"/ws?token={token}") as ws:
        ws.send_json({"type": "user_message", "task": "t_p1", "chat": "peace_scout", "text": "$ echo hello-from-$MNX_BEE",
                      "bee": "scout", "mode": "peace"})
        events = _until_done(ws)
        out = next(e for e in events if e["type"] == "output")
        assert out["text"].strip().splitlines()[-1] == "hello-from-scout" and out["bee"] == "Scout" and out["code"] == 0
        assert {e.get("bee") for e in events if e.get("bee")} == {"Scout"}  # the Bee answers, not the Queen
        ws.send_json({"type": "user_message", "task": "t_p2", "text": "who are you", "bee": "scout", "mode": "peace"})
        card = next(e for e in _until_done(ws) if e["type"] == "result")
        assert card["title"] == "Scout" and ["Skill", "watches prices"] in card["details"]
        ws.send_json({"type": "user_message", "task": "t_p3", "text": "what's the capital of France", "bee": "scout", "mode": "peace"})
        answer = next(e for e in _until_done(ws) if e["type"] == "answer")
        assert answer["text"].startswith("I'm the Scout Bee") and "$ command" in answer["text"]


def test_a_bee_makes_its_own_children_in_its_peace_chat(gateway):
    client, token, r, _ = gateway
    with client.websocket_connect(f"/ws?token={token}") as ws:
        ws.send_json({"type": "user_message", "task": "t_c1", "text": "create a bee called Pinger that pings sites", "bee": "scout", "mode": "peace"})
        while True:
            ev = ws.receive_json()
            if ev["type"] == "clarify":
                ws.send_json({"type": "action", "task": "t_c1", "ref": ev["id"], "action": "create", "values": {}})
            if ev["type"] in ("task_done", "task_error"):
                break
    assert r.get("pinger")["created_by"] == "scout"


def test_rush_roll_call_and_everyone_run(gateway):
    client, token, r, _ = gateway
    with client.websocket_connect(f"/ws?token={token}") as ws:
        ws.send_json({"type": "user_message", "task": "t_r1", "text": "roll call", "mode": "rush"})
        events = _until_done(ws)
        rush = next(e for e in events if e["type"] == "rush")
        assert len(rush["bees"]) == len(r.bees)
        reporters = {e["bee"] for e in events if e["type"] == "result"}
        assert reporters == {b["name"] for b in r.bees}
        ws.send_json({"type": "user_message", "task": "t_r2", "text": "everyone run echo ready-$MNX_BEE", "mode": "rush"})
        events = _until_done(ws, answer="go")
        outs = {e["bee"]: e["text"].strip().splitlines()[-1] for e in events if e["type"] == "output"}
        assert outs["Scout"] == "ready-scout" and outs["Cloud"] == "ready-cloud" and len(outs) == len(r.bees)
        summary = [e for e in events if e["type"] == "result"][-1]
        assert summary["title"] == f"{len(r.bees)} of {len(r.bees)} Bees ran it"


def test_rush_is_off_in_a_normal_chat(gateway):
    client, token, r, _ = gateway
    with client.websocket_connect(f"/ws?token={token}") as ws:
        ws.send_json({"type": "user_message", "task": "t_n1", "text": "roll call"})
        assert not any(e["type"] == "rush" for e in _until_done(ws))


def test_calls_api_and_twilio_webhooks_are_guarded(gateway, monkeypatch):
    client, token, r, c = gateway
    auth = {"Authorization": f"Bearer {token}"}
    assert client.post("/api/calls", json={}).status_code == 401
    call = client.post("/api/calls", headers=auth, json={"bee": "scout", "answer": True}).json()
    assert call["state"] == "live" and call["bee"] == "Scout" and call["voice"] == "fish"
    assert call["transcript"][0]["text"].startswith("Hi, it's the Scout Bee") and call["transcript"][0]["audio"]
    audio = client.get(f"/api/calls/{call['id']}/audio/1", headers=auth)
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/mpeg"
    bye = client.post(f"/api/calls/{call['id']}/say", headers=auth, json={"text": "bye"}).json()
    assert bye["state"] == "ended"
    assert client.post(f"/api/calls/{call['id']}/say", headers=auth, json={"text": "hello"}).status_code == 409
    assert client.post("/api/calls", headers=auth, json={"bee": "nobody"}).status_code == 404
    # Twilio's webhooks: the call's secret and Twilio's signature, or nothing
    secret = c.get(call["id"]).secret
    assert client.post(f"/voice/twilio/{call['id']}/wrong/start").status_code == 404
    assert client.post(f"/voice/twilio/{call['id']}/{secret}/start", data={"CallSid": "CA1"}).status_code == 403
    assert client.get(f"/voice/twilio/{call['id']}/wrong/audio/1.mp3").status_code == 404
    assert client.get(f"/voice/twilio/{call['id']}/{secret}/audio/1.mp3").status_code == 200


def test_voice_settings_api_never_shows_the_key(gateway):
    client, token, *_ = gateway
    auth = {"Authorization": f"Bearer {token}"}
    v = client.get("/api/voice", headers=auth).json()
    assert v["fish_key_set"] and "fk-secret" not in json.dumps(v)
    assert client.put("/api/voice", headers=auth, json={"model": "gpt"}).status_code == 400
    assert client.put("/api/voice", headers=auth, json={"voice_name": "Calm"}).json()["voice_name"] == "Calm"
    assert client.get("/api/voice/voices", headers=auth).json()[0]["id"] == "abc123voice"
    tts = client.post("/api/voice/tts", headers=auth, json={"text": "Hello from the Hive"})
    assert tts.status_code == 200 and tts.content.startswith(b"ID3")
    assert client.post("/api/voice/asr", headers=auth, content=b"open my desktop").json() == {"text": "open my desktop"}


def test_a_hive_call_is_rush_and_a_bee_call_is_that_bees_own_chat(phone_calls):
    c, _ = phone_calls
    seen = []

    async def scenario():
        async def note(task):
            seen.append((task.bee, task.mode))
            await task.answer("ok")
        c.run = note
        hive = await c.start(None, answer=True, talk=True)
        await c.heard(hive.id, "roll call")
        await asyncio.wait_for(hive.runner, 2)
        bee = await c.start("chat", answer=True, talk=True)
        await c.heard(bee.id, "status")
        await asyncio.wait_for(bee.runner, 2)
    run(scenario())
    assert seen == [(None, "rush"), ("chat", "peace")]


def test_provider_keys_never_reach_training_code():
    from hive.lab import HIVE_ONLY_SECRETS
    assert {"RUNPOD_API_KEY", "HCLOUD_TOKEN", "DIGITALOCEAN_TOKEN"} <= HIVE_ONLY_SECRETS
