"""The opt-in local-model bridge, tested against a stub server.

No model is downloaded here. A tiny HTTP server stands in for Ollama or
llama-server and returns canned replies, which is enough to check the part
that matters: MNX1 runs everything the model says and only ships code that
survives.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from mnx.agent.skills.code import CodeSkill, consensus, pass_vector
from mnx.agent.skills.local_model import LocalModelError, check, local_proposer
from mnx.agent.skills.local_model import test_proposer as make_tests


class Stub:
    """A fake OpenAI-compatible server. ``code`` and ``tests`` are reply queues."""

    def __init__(self, code, tests=()):
        self.code = list(code)
        self.tests = list(tests)
        self.seen = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep the test output clean
                pass

            def do_GET(self):
                self._send({"data": [{"id": "stub-coder"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.seen.append(body)
                wants_tests = "test cases" in body["messages"][0]["content"]
                queue = outer.tests if wants_tests else outer.code
                reply = queue.pop(0) if queue else ""
                self._send({"choices": [{"message": {"content": reply}}]})

            def _send(self, payload):
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self):
        self.httpd.shutdown()


def fence(code):
    return f"Here you go:\n```python\n{code}```\n"


PROMPT = 'def shout(s):\n    """Return s upper-cased with an exclamation mark."""\n'


def test_a_missing_server_says_what_to_do():
    with pytest.raises(LocalModelError) as exc:
        check("http://127.0.0.1:1/v1")
    assert "ollama" in str(exc.value).lower()


def test_the_bridge_reports_the_models_the_server_has():
    stub = Stub([])
    try:
        assert check(stub.url) == ["stub-coder"]
    finally:
        stub.close()


def test_a_wrong_proposal_is_dropped_and_a_right_one_is_shipped():
    stub = Stub([fence("def shout(s):\n    return s.lower()\n")] * 2
                + [fence("def shout(s):\n    return s.upper() + '!'\n")] * 2)
    try:
        skill = CodeSkill(proposers=[local_proposer("stub-coder", stub.url, samples=4)])
        ans = skill.answer(PROMPT + "assert shout('hi') == 'HI!'\nassert shout('a') == 'A!'\nassert shout('') == '!'\n")
    finally:
        stub.close()
    assert ans is not None and not ans.failed
    assert "upper()" in ans.text and "lower()" not in ans.text


def test_nothing_is_shipped_when_every_proposal_fails():
    stub = Stub([fence("def shout(s):\n    return s.lower()\n")] * 4)
    try:
        skill = CodeSkill(proposers=[local_proposer("stub-coder", stub.url, samples=4)])
        ans = skill.answer(PROMPT + "assert shout('hi') == 'HI!'\nassert shout('a') == 'A!'\nassert shout('') == '!'\n")
    finally:
        stub.close()
    assert ans.failed and "```" not in ans.text


def test_with_no_asserts_the_model_writes_tests_and_the_candidates_vote():
    stub = Stub(code=[fence("def shout(s):\n    return s.upper() + '!'\n"),
                      fence("def shout(s):\n    return '{}!'.format(s.upper())\n"),
                      fence("def shout(s):\n    return s + '!'\n"),
                      fence("def shout(s):\n    return s.upper()\n")],
                tests=[fence("assert shout('hi') == 'HI!'\nassert shout('a') == 'A!'\nassert shout('xy') == 'XY!'\n")])
    try:
        skill = CodeSkill(proposers=[local_proposer("stub-coder", stub.url, samples=4)],
                          test_makers=[make_tests("stub-coder", stub.url, rounds=1)])
        ans = skill.answer(PROMPT)
    finally:
        stub.close()
    assert ans is not None and not ans.failed
    assert "upper()" in ans.text and "'!'" in ans.text or "!}" in ans.text or "}!" in ans.text
    assert "tests are mine, not yours" in ans.text


def test_generated_tests_nobody_passes_mean_no_answer():
    stub = Stub(code=[fence("def shout(s):\n    return s.upper() + '!'\n")] * 4,
                tests=[fence("assert shout('hi') == 'nope'\n")])
    try:
        skill = CodeSkill(proposers=[local_proposer("stub-coder", stub.url, samples=4)],
                          test_makers=[make_tests("stub-coder", stub.url, rounds=1)])
        ans = skill.answer(PROMPT)
    finally:
        stub.close()
    assert ans is None or ans.failed


def test_consensus_needs_a_clear_winner():
    a = "def f(x):\n    return x + 1\n"
    b = "def f(x):\n    return x + 2\n"
    tests = ["assert f(1) == 2", "assert f(2) == 4"]
    assert consensus([a, b], tests) is None  # one test each, tied
    assert pass_vector("def f(x):\n    raise ValueError\n", tests) == "00"
