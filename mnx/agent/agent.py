"""The MNX agent: route → try skills → verify → answer or abstain.

::

    question ─► router (naive Bayes prior over skills)
             ─► skills tried in prior order, each may return a candidate
                (math / units / dates / logic / sequence / synth / memory /
                 knowledge / chat / optional neural-code)
             ─► rank candidates by confidence, verification and prior
             ─► best answer, or "I don't know" below the threshold
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from mnx.agent.base import Answer, Skill
from mnx.agent.retrieval import Memory, build_index, default_knowledge_dir
from mnx.agent.router import NaiveBayesRouter, train_default_router
from mnx.agent.skills.chat import ChatSkill
from mnx.agent.skills.dates import DateSkill
from mnx.agent.skills.knowledge import KnowledgeSkill, MemorySkill
from mnx.agent.skills.logic import LogicSkill
from mnx.agent.skills.math import MathSkill
from mnx.agent.skills.sequences import SequenceSkill
from mnx.agent.skills.synth import SynthSkill
from mnx.agent.skills.units import UnitSkill

ABSTAIN = (
    "I don't know. None of my skills could produce an answer I trust, and I didn't find it in my knowledge base. "
    "You can teach me with \"remember that ...\" or point me at your own notes with --kb."
)

#: skills that are slow; only tried when the router gives them some weight
_EXPENSIVE = {"synth", "neural-code"}


class MNXAgent:
    """Compose skills into one assistant.

    Parameters
    ----------
    knowledge_dirs:
        Folders of ``.md`` / ``.txt`` files to index. Defaults to the bundled
        ``data/knowledge``.
    memory_path:
        Optional JSONL file where "remember that ..." facts are persisted.
    ckpt:
        Optional MNXCoder checkpoint directory; enables the neural code skill
        (requires PyTorch).
    threshold:
        Minimum ranking score for an answer; below it the agent abstains.
    """

    def __init__(
        self,
        knowledge_dirs: Optional[Iterable[Path]] = None,
        memory_path: Optional[Path] = None,
        ckpt: Optional[str] = None,
        threshold: float = 0.35,
        today: Optional[dt.date] = None,
        router: Optional[NaiveBayesRouter] = None,
        synth_time_limit: float = 4.0,
    ):
        dirs = list(knowledge_dirs) if knowledge_dirs is not None else [default_knowledge_dir()]
        self.memory = Memory(memory_path)
        self.index = build_index(dirs, self.memory.facts)
        self.router = router or train_default_router()
        self.threshold = threshold
        skills: List[Skill] = [
            ChatSkill(),
            MemorySkill(self.index, self.memory),
            MathSkill(),
            UnitSkill(),
            DateSkill(today),
            LogicSkill(),
            SequenceSkill(),
            SynthSkill(synth_time_limit),
            KnowledgeSkill(self.index),
        ]
        if ckpt:
            from mnx.agent.skills.neural import NeuralCodeSkill, torch_sampler

            skills.append(NeuralCodeSkill(torch_sampler(ckpt)))
        self.skills: Dict[str, Skill] = {s.name: s for s in skills}

    # ------------------------------------------------------------------ API
    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        q = question.strip()
        if not q:
            return Answer("Ask me something, or type 'help'.", "chat", 1.0)
        posterior = dict(self.router.predict(q))
        # router labels map to skills; memory questions are answered by retrieval
        prior = {name: posterior.get(name, 0.0) for name in self.skills}
        prior["knowledge"] = max(prior.get("knowledge", 0.0), posterior.get("memory", 0.0) * 0.5)
        prior["neural-code"] = posterior.get("synth", 0.0)
        order = sorted(self.skills, key=lambda n: prior.get(n, 0.0), reverse=True)

        candidates: List[Tuple[float, Answer]] = []
        tried: List[str] = []
        for name in order:
            p = prior.get(name, 0.0)
            if name in _EXPENSIVE and p < 0.05 and not _looks_like_examples(q):
                continue
            tried.append(name)
            ans = self.skills[name].answer(q)
            if ans is None:
                continue
            candidates.append((ans.score(p), ans))
            if name == "memory" or (ans.verified and ans.confidence >= 0.9 and p >= 0.5):
                break  # confident, verified and expected: no need to try the rest

        top = ", ".join(f"{k} {v:.2f}" for k, v in sorted(posterior.items(), key=lambda kv: -kv[1])[:3])
        header = [f"router: {top}", f"tried: {', '.join(tried)}"]
        if not candidates:
            return Answer(ABSTAIN, "abstain", 0.0, False, header + ["no skill produced a candidate"])
        candidates.sort(key=lambda sa: sa[0], reverse=True)
        score, best = candidates[0]
        others = [f"{a.skill} ({s:.2f})" for s, a in candidates[1:4]]
        header.append(f"chose {best.skill} (score {score:.2f})" + (f" over {', '.join(others)}" if others else ""))
        header.append(f"{(time.perf_counter() - start) * 1000:.0f} ms")
        if score < self.threshold:
            return Answer(ABSTAIN, "abstain", score, False, header + best.trace, best.sources)
        best.trace = header + best.trace
        return best


def _looks_like_examples(q: str) -> bool:
    import re

    return bool(re.search(r"->|→|=>|[a-z]\s*\([^)]*\)\s*=.*,", q))


def format_answer(ans: Answer, show_trace: bool = False) -> str:
    lines = [ans.text]
    if show_trace:
        tag = "verified" if ans.verified else "unverified"
        lines.append(f"\n[{ans.skill} · {tag} · confidence {ans.confidence:.2f}]")
        lines.extend(f"  {t}" for t in ans.trace)
    return "\n".join(lines)


def ask(question: str, **kwargs) -> Answer:
    """One-shot helper: ``mnx.agent.ask("solve x^2 = 9").text``."""
    return MNXAgent(**kwargs).ask(question)


__all__: Sequence[str] = ["MNXAgent", "ask", "format_answer", "ABSTAIN"]
