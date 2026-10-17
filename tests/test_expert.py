"""Expert Stream as the models' engine, and Expert Craft training on a Bee's computer, from the chat."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive import expert, models as models_mod  # noqa: E402


@pytest.mark.parametrize("text,want", [
    ("expert craft plan 70B for 2 hours", {"kind": "plan", "params": 70e9, "minutes": 120}),
    ("train a 70B model with expert craft called stories for 3 hours", {"kind": "train", "name": "stories", "minutes": 180}),
    ("train a 7b model with expert craft on my notes.txt", {"kind": "train", "params": 7e9, "data": "notes.txt"}),
    ("train a 500B model with expert craft", {"kind": "train", "params": 70e9}),  # 70B is the most it does
    ("expert craft status stories", {"kind": "status", "name": "stories"}),
    ("ask my expert craft model stories: Once upon a time", {"kind": "sample", "name": "stories", "prompt": "Once upon a time"}),
    ("set up expert stream", {"kind": "stream"}),
])
def test_phrases(text, want):
    got = expert.parse(text)
    for k, v in want.items():
        assert got[k] == v, (k, got)


@pytest.mark.parametrize("text", ["train qwen 2.5 coder", "what is a stream", "craft a poem for me", "teach me python"])
def test_not_expert_phrases(text):
    assert expert.parse(text) is None


def test_models_run_on_expert_stream_when_it_is_built(tmp_path, monkeypatch):
    m = models_mod.Models(tmp_path / "models")
    monkeypatch.setattr(expert, "es_serve_path", lambda: "/opt/es/es_serve")
    monkeypatch.delenv("MNX_ENGINE", raising=False)
    assert m._server_argv("/m/mnx.gguf", 8081) == ["/opt/es/es_serve", "-g", "/m/mnx.gguf", "-p", "8081", "-c", "4096"]
    assert m.engine() == "Expert Stream"
    monkeypatch.setenv("MNX_ENGINE", "llama")  # the owner can keep llama.cpp
    try:
        argv = m._server_argv("/m/mnx.gguf", 8081)
    except models_mod.ModelError:  # no llama.cpp on this machine
        argv = []
    assert "/opt/es/es_serve" not in argv and m.engine() == "llama.cpp"
    monkeypatch.setattr(expert, "es_serve_path", lambda: None)
    monkeypatch.setenv("MNX_ENGINE", "expertstream")
    with pytest.raises(models_mod.ModelError, match="install-expertstream"):
        m._server_argv("/m/mnx.gguf", 8081)


def test_only_the_owner_sets_up_the_engine():
    import asyncio
    from hive.core import Task
    events = []

    async def emit(ev):
        events.append(ev)
    t = Task("t1", "c1", "set up expert stream", emit, user="u_someone00001")
    asyncio.run(expert.handle(t, expert.parse(t.text)))
    assert "owner" in events[-1]["text"]
