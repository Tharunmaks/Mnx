"""Optional tool agents that reach outside the machine: the web, Chrome, Gmail and Drive.

MNX1 is offline by default and none of this is loaded unless you ask for it::

    python -m mnx.agent --tools web                  # read pages and search the web (standard library)
    python -m mnx.agent --tools web --chrome         # ... through your own Chrome, via Playwright
    python -m mnx.agent --tools gmail,drive          # after: python -m mnx.agent.tools.google login

Tool agents answer explicit requests ("open ...", "check my email",
"search my drive for ...") before the router runs, because only they can
answer those. What they return is live data, not a verified answer, so it is
never replayed from experience or used to self-train the router. Text from
pages, emails and files is shown to you and never treated as instructions.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from mnx.agent.base import Skill

#: tool agents and the skills each one owns
TOOL_AGENTS: Dict[str, Tuple[str, ...]] = {"browser": ("web",), "mail": ("gmail",), "files": ("drive",)}
TOOL_NAMES = ("web", "gmail", "drive")


def build_tools(names: Sequence[str], chrome: bool = False, cdp: str = "http://localhost:9222", google_dir=None) -> List[Skill]:
    """Instantiate the requested tool skills ("web", "gmail", "drive"; "google" means both of the last two)."""
    wanted = []
    for n in names:
        n = n.strip().lower()
        if not n:
            continue
        expanded = ("gmail", "drive") if n == "google" else ("web", "gmail", "drive") if n == "all" else (n,)
        for e in expanded:
            if e not in TOOL_NAMES:
                raise ValueError(f"unknown tool {n!r}; choose from web, gmail, drive, google, all")
            if e not in wanted:
                wanted.append(e)
    skills: List[Skill] = []
    if "web" in wanted:
        from mnx.agent.tools.web import ChromeFetcher, WebSkill

        skills.append(WebSkill(ChromeFetcher(cdp) if chrome else None))
    if "gmail" in wanted or "drive" in wanted:
        from mnx.agent.tools.google import GoogleAuth, GoogleClient

        client = GoogleClient(GoogleAuth(google_dir))
        if "gmail" in wanted:
            from mnx.agent.tools.gmail import GmailSkill

            skills.append(GmailSkill(client))
        if "drive" in wanted:
            from mnx.agent.tools.drive import DriveSkill

            skills.append(DriveSkill(client))
    return skills


__all__ = ["TOOL_AGENTS", "TOOL_NAMES", "build_tools"]
