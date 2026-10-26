"""Optional neural proposer: sample code from MNXCoder, keep what passes tests.

This is the AlphaCode / "generate-then-verify" recipe at toy scale. A small
model that is right 1 time in 20 becomes useful when a verifier can pick out
that one sample. The skill is only enabled when PyTorch is installed and a
checkpoint trained with ``python -m mnx.train`` is passed (``--ckpt``).

Question format::

    def square(x):
        \"\"\"Return x squared.\"\"\"
    assert square(3) == 9
    assert square(-2) == 4
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Callable, List, Optional, Tuple

from mnx.agent.base import Answer, Skill

Sampler = Callable[[str, int], List[str]]


def split_prompt_and_tests(question: str) -> Optional[Tuple[str, List[str]]]:
    text = question.strip().strip("`")
    text = re.sub(r"^python\n", "", text)
    lines = text.splitlines()
    tests = [l.strip() for l in lines if l.strip().startswith("assert ")]
    body = [l for l in lines if not l.strip().startswith("assert ")]
    if not tests or not any(l.startswith("def ") for l in body):
        return None
    return "\n".join(body).rstrip() + "\n", tests


def assemble(prompt: str, completion: str) -> str:
    """Prompt + completion, cut where the model leaves the function body."""
    out = []
    for i, line in enumerate(completion.splitlines()):
        if i > 0 and line and not line[0].isspace():
            break
        out.append(line)
    return prompt + "\n".join(out) + "\n"


def run_tests(code: str, tests: List[str], timeout: float = 3.0) -> Tuple[bool, str]:
    """Run code + asserts in a separate, isolated Python process with limits."""
    script = code + "\n" + "\n".join(tests) + "\nprint('MNX_OK')\n"

    def limits():  # pragma: no cover - runs in the child
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20, 512 * 2**20))
            resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 1))
        except Exception:
            pass

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=timeout,
            preexec_fn=limits if sys.platform != "win32" else None,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    ok = proc.returncode == 0 and "MNX_OK" in proc.stdout
    return ok, (proc.stderr.strip().splitlines() or [""])[-1]


def torch_sampler(ckpt: str, temperature: float = 0.8, max_new_tokens: int = 96) -> Sampler:
    from mnx.generate import complete
    from mnx.train import load_checkpoint

    model, tok, _ = load_checkpoint(ckpt)

    def sample(prompt: str, k: int) -> List[str]:
        # End the prompt before its last newline: in training "\n" + indent is usually one token, so a prompt
        # that already ends in "\n" forces an unusual split and the model starts a new function instead.
        head = prompt[:-1] if prompt.endswith("\n") else prompt
        outs = [complete(model, tok, head, max_new_tokens=max_new_tokens, temperature=temperature, top_k=40, seed=i)
                for i in range(k)]
        return [o[1:] if head is not prompt and o.startswith("\n") else o for o in outs]

    return sample


class NeuralCodeSkill(Skill):
    name = "neural-code"
    description = "sample function bodies from the MNXCoder transformer and keep one that passes the asserts"

    def __init__(self, sampler: Sampler, k: int = 16):
        self.sampler = sampler
        self.k = k

    def answer(self, question: str) -> Optional[Answer]:
        parsed = split_prompt_and_tests(question)
        if parsed is None:
            return None
        prompt, tests = parsed
        trace = [f"sampling {self.k} completions, verifying each against {len(tests)} assert(s) in a sandboxed process"]
        for i, completion in enumerate(self.sampler(prompt, self.k)):
            code = assemble(prompt, completion)
            ok, err = run_tests(code, tests)
            if ok:
                trace.append(f"sample {i + 1} passed all tests")
                return Answer(f"```python\n{code}```", self.name, 0.9, True, trace)
            trace.append(f"sample {i + 1} failed: {err[:80]}")
        return Answer(f"None of {self.k} samples passed the tests.", self.name, 0.3, False, trace, failed=True)

