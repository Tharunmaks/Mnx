"""Optional proposer: a code model running on your own machine, verified by MNX1.

MNX1's own answers need no model and no network. This module is the opt-in
bridge for people who want higher coverage than hand-written recipes and
program search can reach: it asks a model served on ``localhost`` for
candidate functions and hands them back as *proposals*. Nothing it returns is
trusted. The coding agent runs every proposal against the tests in a
``python -I`` sandbox, clusters the survivors by how they behave on fresh
probe inputs, and only ships code that passes. A proposal that fails is
dropped silently, so the "always right or silent" rule still holds.

It speaks the OpenAI-compatible ``/v1/chat/completions`` endpoint, which both
`Ollama <https://ollama.com>`_ and ``llama-server`` from llama.cpp expose, so
no extra Python package is needed -- only the standard library::

    ollama serve &
    ollama pull qwen2.5-coder:7b
    python -m mnx.agent --model qwen2.5-coder:7b

Still free and still offline: the model file lives on your disk and nothing
leaves the machine. There is no API key anywhere in this file.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Callable, List, Optional

DEFAULT_URL = "http://localhost:11434/v1"

Proposer = Callable[[str, List[str]], List[str]]

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)

SYSTEM = (
    "You are a Python coding assistant. Reply with one complete Python function in a single "
    "```python code block and nothing else: no explanation, no tests, no example calls. "
    "Keep the signature exactly as given and make the body satisfy the docstring and the asserts."
)


class LocalModelError(RuntimeError):
    """The local server could not be reached or refused the request."""


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # localhost only by default
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:  # pragma: no cover - needs a server
        raise LocalModelError(f"{url} answered HTTP {exc.code}: {exc.read().decode()[:200]}") from exc
    except OSError as exc:
        raise LocalModelError(
            f"cannot reach a local model at {url} ({exc}). Start one with "
            "`ollama serve` (then `ollama pull qwen2.5-coder:7b`), or point --model-url at your own server."
        ) from exc


def check(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> List[str]:
    """Names of the models the local server has, or raise ``LocalModelError``."""
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:  # pragma: no cover - needs a server
        raise LocalModelError(f"{url} answered HTTP {exc.code}") from exc
    except OSError as exc:
        raise LocalModelError(
            f"cannot reach a local model server at {base_url} ({exc}). "
            "Install Ollama from https://ollama.com, run `ollama serve`, then `ollama pull qwen2.5-coder:7b`."
        ) from exc
    return [m.get("id", "") for m in data.get("data", [])]


def extract(text: str, name: Optional[str]) -> Optional[str]:
    """The function from a model reply: the fenced block, or the def onwards."""
    text = _THINK.sub("", text or "")
    blocks = [m.group(1) for m in _FENCE.finditer(text)]
    if not blocks:
        start = text.find("def ")
        if start < 0:
            return None
        blocks = [text[start:]]
    for block in blocks:
        if name and f"def {name}" not in block:
            continue
        lines = block.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        if lines and not any(l.startswith(("def ", "class ", "import ", "from ", "@")) for l in lines):
            continue
        return "\n".join(lines).rstrip() + "\n"
    return None


def ask_once(model: str, prompt: str, tests: List[str], base_url: str, temperature: float,
             timeout: float, max_tokens: int) -> Optional[str]:
    user = prompt.rstrip() + "\n\nIt must satisfy:\n" + "\n".join(tests)
    data = _post(base_url.rstrip("/") + "/chat/completions", {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }, timeout)
    choices = data.get("choices") or []
    if not choices:
        return None
    text = (choices[0].get("message") or {}).get("content", "")
    m = re.search(r"def\s+(\w+)\s*\(", prompt)
    return extract(text, m.group(1) if m else None)


def local_proposer(model: str, base_url: str = DEFAULT_URL, samples: int = 4,
                   temperature: float = 0.6, timeout: float = 180.0,
                   max_tokens: int = 768) -> Proposer:
    """A ``(prompt, tests) -> [code]`` proposer backed by a model on localhost.

    The first sample is greedy; the rest are sampled, so one model can still
    fill the behaviour groups the coding agent votes over. Server errors
    after the first call are swallowed: a missing proposal only means MNX1
    falls back to its own recipes and program search.
    """

    def propose(prompt: str, tests: List[str]) -> List[str]:
        out: List[str] = []
        for i in range(max(1, samples)):
            try:
                code = ask_once(model, prompt, tests, base_url, 0.0 if i == 0 else temperature,
                                timeout, max_tokens)
            except LocalModelError:
                if i == 0:
                    raise
                break
            if code and code not in out:
                out.append(code)
        return out

    return propose


TEST_SYSTEM = (
    "You write Python test cases. Given a function signature and its docstring, reply with 5 to 10 "
    "`assert` lines that any correct implementation must pass, one per line, inside a single "
    "```python code block. Call only the given function. No comments, no implementation, no prints. "
    "Work out each expected value yourself and only write asserts you are sure about."
)


def _assert_lines(text: str, name: str) -> List[str]:
    text = _THINK.sub("", text or "")
    blocks = [m.group(1) for m in _FENCE.finditer(text)] or [text]
    out: List[str] = []
    for block in blocks:
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("assert ") or f"{name}(" not in line:
                continue
            try:
                compile(line, "<model>", "exec")
            except SyntaxError:
                continue
            if line not in out:
                out.append(line)
    return out


def test_proposer(model: str, base_url: str = DEFAULT_URL, rounds: int = 2, temperature: float = 0.6,
                  timeout: float = 180.0, max_tokens: int = 512) -> Callable[[str], List[str]]:
    """A ``prompt -> [assert line]`` maker: tests for a signature that came with none.

    Model-written tests are not ground truth, so the coding agent never treats
    a pass as proof: it uses them only to pick between its own candidate
    programs, and says in the answer that the tests were generated.
    """

    def make(prompt: str) -> List[str]:
        m = re.search(r"def\s+(\w+)\s*\(", prompt)
        if not m:
            return []
        out: List[str] = []
        for i in range(max(1, rounds)):
            try:
                data = _post(base_url.rstrip("/") + "/chat/completions", {
                    "model": model,
                    "messages": [{"role": "system", "content": TEST_SYSTEM},
                                 {"role": "user", "content": prompt.rstrip()}],
                    "temperature": 0.0 if i == 0 else temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }, timeout)
            except LocalModelError:
                if i == 0:
                    raise
                break
            choices = data.get("choices") or []
            if not choices:
                continue
            for line in _assert_lines((choices[0].get("message") or {}).get("content", ""), m.group(1)):
                if line not in out:
                    out.append(line)
        return out

    return make
