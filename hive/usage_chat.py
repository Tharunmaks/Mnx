"""Usage in the chat: "show my usage", "how many tokens have I used", "what are my limits"."""

from __future__ import annotations

import re
import time

from . import architect, effort, usage
from .bees import MAX_BEES, MAX_CHILDREN, MAX_DEPTH, registry
from .core import Task

ASK = re.compile(r"\b(?:my|the)\s+(?:usage|limits?|quota)\b|\bhow (?:many|much)\b.*\b(?:tokens?|used|usage|spent)\b|"
                 r"\b(?:show|see|check)\b.*\b(?:usage|limits)\b|\bwhat are (?:my|the) limits\b|\busage (?:and|&) limits\b", re.I)


def matches(text: str) -> bool:
    return bool(ASK.search(text))


async def handle(task: Task) -> None:
    from .lab import lab
    t = usage.token_summary(usage.ledger)
    now = time.time()
    pods = [usage.rental_now(p, now) for p in lab.pods.values()]
    today, week, total = t["today"], t["week"], t["all"]
    peak = today["peak"]
    level = effort.level(task.effort)
    rows = [["Tokens today", f"{today['tokens']:,} in {today['messages']} message{'s' if today['messages'] != 1 else ''} ({today['in']:,} in, {today['out']:,} out)"],
            ["Last 7 days", f"{week['tokens']:,} tokens, {week['messages']:,} messages"],
            ["All time", f"{total['tokens']:,} tokens, {total['messages']:,} messages" + (f" since {total['since']}" if total["since"] else "")],
            ["Limit per message", f"{level['tokens']:,} tokens at {level['label']} effort; your biggest today was {peak:,} ({100 * peak // level['tokens']}%)"],
            ["Bees", f"{len(registry.bees)} of {MAX_BEES} · {MAX_DEPTH} generations · {MAX_CHILDREN} children each"],
            ["Training", f"{sum(1 for r in lab.runs.values() if r['status'] in ('preparing', 'running'))} running · {len(lab.runs)} runs · {len(lab.models)} models"],
            ["Rented GPUs", ", ".join(f"{p['count']}× {p['gpu']} for {architect.fmt_time(p['hours'] * 3600)}"
                                      + (f" (${p['cost']:.2f})" if p["cost"] is not None else "") for p in pods) or "none rented, nothing billing"]]
    await task.emit("result", service="Hive", title="Your usage and limits", badge="Usage", details=rows,
                    text=t["estimate"] + " The Usage page shows the daily chart, each Bee's CPU, memory and disk, and GPU rental costs.")
    await task.answer(f"Today {today['tokens']:,} tokens in {today['messages']} messages; {week['tokens']:,} this week. "
                      f"Open Usage in the sidebar for the full picture.")
