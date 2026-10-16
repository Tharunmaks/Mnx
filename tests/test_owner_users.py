"""Only the owner enters or sees credentials; users get the product but never the keys."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway import auth  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import gateway.api as gw
    from hive import voice as voice_mod, brain as brain_mod
    monkeypatch.setattr(auth, "USER_TOKENS_FILE", tmp_path / "user_tokens.txt")
    (tmp_path / "user_tokens.txt").write_text("# users\nuser-token-abc\n")
    monkeypatch.setattr(gw.lab, "_secrets", {"IOTAMINE_API_KEY": "secret-iota-key", "KAGGLE_USERNAME": "tharun"})
    monkeypatch.setattr(gw.lab, "set_secret", lambda n, v: gw.lab._secrets.__setitem__(n, v))
    v = voice_mod.Voice(tmp_path / "voice.json")
    v.update({"fish_key": "fish-secret", "phone": "+919876543210"})
    monkeypatch.setattr(gw, "voice", v)
    b = brain_mod.Brain(tmp_path / "brain.json")
    b.update({"url": "http://127.0.0.1:11434", "key": "brain-secret", "model": "llama3.2"})
    monkeypatch.setattr(brain_mod, "brain", b)
    return TestClient(gw.app)


OWNER = lambda: {"Authorization": f"Bearer {auth.TOKEN}"}  # noqa: E731
USER = {"Authorization": "Bearer user-token-abc"}


def test_roles():
    assert auth.role(auth.TOKEN) == "owner" and auth.role("nope") is None and auth.role("") is None


def test_whoami(client):
    assert client.get("/api/whoami", headers=OWNER()).json() == {"role": "owner"}
    assert client.get("/api/whoami", headers=USER).json() == {"role": "user"}
    assert client.get("/api/whoami", headers={"Authorization": "Bearer wrong"}).status_code == 401


@pytest.mark.parametrize("method,path,body", [
    ("post", "/api/cloud/iotamine/key", {"key": "stolen-attempt"}),
    ("post", "/api/cloud/kaggle", {"username": "x", "token": "y"}),
    ("post", "/api/cloud/pool", {"host": "1.2.3.4"}),
    ("post", "/api/cloud/iotamine/deploy", {"confirm": True}),
    ("get", "/api/cloud/iotamine", None),
    ("get", "/api/cloud/free", None),
    ("get", "/api/cloud/quote", None),
    ("post", "/api/lab/secrets", {"name": "X_KEY", "value": "v"}),
    ("post", "/api/lab/targets", {"name": "a", "host": "1.2.3.4", "user": "root"}),
    ("post", "/api/lab/providers/runpod/rent", {}),
    ("put", "/api/brain", {"url": "http://evil.example"}),
    ("put", "/api/voice", {"fish_key": "evil"}),
    ("post", "/api/bees/scout/cloud", None),
])
def test_users_cannot_touch_credentials_or_setup(client, method, path, body):
    r = getattr(client, method)(path, headers=USER, **({"json": body} if body is not None else {}))
    assert r.status_code == 403, (path, r.status_code)
    assert getattr(client, method)(path, **({"json": body} if body is not None else {})).status_code == 401


def test_users_get_the_product_but_never_a_key(client):
    for path in ("/api/voice", "/api/brain", "/api/cloud/status", "/api/computers", "/api/usage", "/api/lab", "/api/plan", "/api/bots"):
        r = client.get(path, headers=USER)
        assert r.status_code == 200, path
        text = json.dumps(r.json())
        for secret in ("secret-iota-key", "fish-secret", "brain-secret", "+919876543210", "127.0.0.1:11434", "ssh-ed25519"):
            assert secret not in text, (path, secret)
    st = client.get("/api/cloud/status", headers=USER).json()
    assert set(st) == {"plan", "gpu_ready", "gpu_source", "hours_week", "weekly_limit", "servers"}
    assert client.get("/api/lab", headers=USER).json()["secrets"] == []
    assert client.get("/api/computers", headers=USER).json()["cloud"] == []
    assert client.get("/api/usage", headers=USER).json()["cloud"]["providers"] == []
    # users can still subscribe to Plus and use the GPU endpoints
    assert client.put("/api/plan", headers=USER, json={"plan": "plus"}).status_code == 200
    assert client.get("/api/gpu", headers=USER).status_code == 200


def test_the_owner_sees_and_sets_credentials(client):
    assert client.get("/api/voice", headers=OWNER()).json()["fish_key_set"]
    assert client.get("/api/brain", headers=OWNER()).json()["url"] == "http://127.0.0.1:11434"
    assert "secret-iota-key" not in json.dumps(client.get("/api/lab", headers=OWNER()).json())  # names only, never values
    assert "IOTAMINE_API_KEY" in client.get("/api/lab", headers=OWNER()).json()["secrets"]
    assert client.get("/api/cloud/free", headers=OWNER()).status_code == 200


def test_admin_command_stores_credentials_privately(tmp_path, monkeypatch, capsys):
    from hive import admin
    import hive.lab as lab_mod
    store = {}
    monkeypatch.setattr(lab_mod.lab, "_secrets", store)
    monkeypatch.setattr(lab_mod.lab, "set_secret", lambda n, v: store.__setitem__(n, v) if v else store.pop(n, None))
    assert admin.main(["set", "iotamine_api_key", "abc-123"]) == 0 and store == {"IOTAMINE_API_KEY": "abc-123"}
    assert admin.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "✓ IOTAMINE_API_KEY" in out and "abc-123" not in out
    assert admin.main(["unset", "IOTAMINE_API_KEY"]) == 0 and store == {}
    monkeypatch.setattr(auth, "USER_TOKENS_FILE", tmp_path / "u.txt")
    assert admin.main(["user-token"]) == 0
    tok = capsys.readouterr().out.split(": ")[1].strip()
    monkeypatch.setattr(auth, "USER_TOKENS_FILE", tmp_path / "u.txt")
    assert auth.role(tok) == "user" and oct((tmp_path / "u.txt").stat().st_mode)[-3:] == "600"
