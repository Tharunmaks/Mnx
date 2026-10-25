"""MNX agent: a small, from-scratch reasoning assistant (no LLM, no API key).

It combines a learned router with verified symbolic skills, retrieval over
local files, and test-time program search. See ``README.md`` for the design
and its limits. This subpackage needs only the Python standard library.
"""

from mnx.agent.agent import ABSTAIN, AGENTS, MNX1, MNXAgent, ask, format_answer
from mnx.agent.base import Answer, Skill

__all__ = ["MNX1", "MNXAgent", "AGENTS", "Answer", "Skill", "ask", "format_answer", "ABSTAIN"]
