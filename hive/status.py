"""Which Bees are busy right now (shown on the Hive page and the chat's Bees panel)."""

from __future__ import annotations

busy: dict[str, str] = {}  # bee id → "busy" | "failed"
on_change = None  # set by the gateway: tells open pages to refresh their Bees panel


def _changed() -> None:
    if on_change:
        try:
            on_change()
        except Exception:
            pass


def set_busy(*bee_ids: str) -> None:
    for b in bee_ids:
        busy[b.lower()] = "busy"
    _changed()


def clear(*bee_ids: str) -> None:
    if any(b.lower() in busy for b in bee_ids):
        for b in bee_ids:
            busy.pop(b.lower(), None)
        _changed()
