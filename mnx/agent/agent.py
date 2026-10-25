"""MNX1: an orchestrator that delegates to specialist agents and learns from use.

::

    question
       ├─ feedback? ("correct" / "wrong, the answer is X") → learn, reply
       ├─ corrected before? → your answer, credited to you
       ├─ solved before by a slow agent? → replay the verified answer
       └─ router (naive Bayes prior over skills, updated online)
            → specialist agents, each a group of skills:
                coder        code (recipes, tests, run/check), synth, neural proposals
                mathematician math, units, dates, sequence
                reasoner     logic
                researcher   knowledge, memory
                conversation chat
            → rank candidates by confidence, verification and prior
            → best answer, or "I don't know" below the threshold
       → log to the transcript, self-train on verified answers
"""

from __future__ import annotations

import datetime as dt
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from mnx.agent.base import Answer, Skill
from mnx.agent.learning import Learner
from mnx.agent.retrieval import Memory, build_index, default_knowledge_dir
from mnx.agent.router import NaiveBayesRouter, train_default_router
from mnx.agent.skills.chat import ChatSkill
from mnx.agent.skills.code import CodeSkill
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

#: specialist agents and the skills each one owns
AGENTS: Dict[str, Tuple[str, ...]] = {
    "coder": ("code", "synth", "neural-code"),
    "mathematician": ("math", "units", "dates", "sequence"),
    "reasoner": ("logic",),
    "researcher": ("knowledge", "memory", "learned"),
    "conversation": ("chat", "feedback"),
}
SKILL_TO_AGENT = {s: a for a, skills in AGENTS.items() for s in skills}

#: skills that are slow; only tried when the router gives them some weight
_EXPENSIVE = {"synth"}


class MNXAgent:
    """Compose specialist agents into one assistant.

    Parameters
    ----------
    knowledge_dirs:
        Folders of ``.md`` / ``.txt`` files to index. Defaults to the bundled
        ``data/knowledge``.
    memory_path:
        Optional JSONL file where "remember that ..." facts are persisted.
    state_dir:
        Folder where MNX1 keeps what it learns (transcript, corrections,
        router examples, experience, learned code recipes). ``None`` keeps
        learning in memory for this session only.
    ckpt:
        Optional MNXCoder checkpoint directory; its samples become extra
        proposals for the coding agent (requires PyTorch).
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
        state_dir: Optional[Path] = None,
        learn: bool = True,
    ):
        dirs = list(knowledge_dirs) if knowledge_dirs is not None else [default_knowledge_dir()]
        self.learner = Learner(state_dir) if learn else None
        if memory_path is None and self.learner and self.learner.dir:
            memory_path = self.learner.dir / "memory.jsonl"
        self.memory = Memory(memory_path)
        self.index = build_index(dirs, self.memory.facts)
        self.router = router or train_default_router()
        if self.learner:
            for label, text in self.learner.intents:
                self.router.partial_fit(label, text)
        self.threshold = threshold
        proposers = []
        if ckpt:
            from mnx.agent.skills.neural import assemble, torch_sampler

            sampler = torch_sampler(ckpt)
            proposers.append(lambda prompt, tests: [assemble(prompt, c) for c in sampler(prompt, 16)])
        skills: List[Skill] = [
            ChatSkill(),
            MemorySkill(self.index, self.memory),
            CodeSkill(learned_dir=self.learner.recipe_dir if self.learner else None, proposers=proposers),
            MathSkill(),
            UnitSkill(),
            DateSkill(today),
            LogicSkill(),
            SequenceSkill(),
            SynthSkill(synth_time_limit),
            KnowledgeSkill(self.index),
        ]
        self.skills: Dict[str, Skill] = {s.name: s for s in skills}

    # ------------------------------------------------------------------ API
    def ask(self, question: str) -> Answer:
        q = question.strip()
        if not q:
            return Answer("Ask me something, or type 'help'.", "chat", 1.0)
        ans = self._answer(q)
        if self.learner:
            self.learner.observe(q, ans, self.router, SKILL_TO_AGENT.get(ans.skill, "none"))
        return ans

    def _answer(self, q: str) -> Answer:
        start = time.perf_counter()
        if self.learner:
            for shortcut in (self.learner.handle_feedback(q, self.router), self.learner.lookup_correction(q), self.learner.replay(q)):
                if shortcut is not None:
                    shortcut.trace = [f"agent: {SKILL_TO_AGENT.get(shortcut.skill, 'learner')}"] + shortcut.trace
                    return shortcut
        posterior = dict(self.router.predict(q))
        # router labels map to skills; memory questions are answered by retrieval
        prior = {name: posterior.get(name, 0.0) for name in self.skills}
        prior["knowledge"] = max(prior.get("knowledge", 0.0), posterior.get("memory", 0.0) * 0.5)
        prior["code"] = max(prior.get("code", 0.0), posterior.get("neural-code", 0.0))
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
        header.insert(0, f"agent: {SKILL_TO_AGENT.get(best.skill, '?')} → skill: {best.skill}")
        header.append(f"chose {best.skill} (score {score:.2f})" + (f" over {', '.join(others)}" if others else ""))
        header.append(f"{(time.perf_counter() - start) * 1000:.0f} ms")
        if score < self.threshold and best.failed and best.skill != "knowledge":
            best.trace = header + best.trace  # the skill's own "can't do this, here's why" beats a generic refusal
            return best
        if score < self.threshold:
            return Answer(ABSTAIN, "abstain", score, False, header[1:] + best.trace, best.sources)
        best.trace = header + best.trace
        return best


class MNX1(MNXAgent):
    """MNXAgent that learns persistently into ``~/.mnx`` by default."""

    def __init__(self, state_dir: Optional[Path] = Path("~/.mnx"), **kwargs):
        super().__init__(state_dir=state_dir, **kwargs)


def _looks_like_examples(q: str) -> bool:
    return bool(re.search(r"->|→|=>|[a-z]\s*\([^)]*\)\s*=.*,", q))


def format_answer(ans: Answer, show_trace: bool = False) -> str:
    lines = [ans.text]
    if show_trace:
        tag = "verified" if ans.verified else "unverified"
        lines.append(f"\n[{SKILL_TO_AGENT.get(ans.skill, ans.skill)}/{ans.skill} · {tag} · confidence {ans.confidence:.2f}]")
        lines.extend(f"  {t}" for t in ans.trace)
    return "\n".join(lines)


def ask(question: str, **kwargs) -> Answer:
    """One-shot helper: ``mnx.agent.ask("solve x^2 = 9").text``."""
    return MNXAgent(**kwargs).ask(question)


__all__: Sequence[str] = ["MNXAgent", "MNX1", "AGENTS", "ask", "format_answer", "ABSTAIN"]
