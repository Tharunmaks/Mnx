"""Token reading: estimate how many tokens a piece of text costs, without a tokenizer.

Close enough to a BPE tokenizer (within ~10–15% on English): short words are one token, long
words split every ~5 letters, numbers every 3 digits, punctuation is one token each, and
every character of a non-Latin script (Tamil, Hindi, Chinese…) counts as one. The web app's
shared.js has the same rule for the live counter under the message box.
"""

from __future__ import annotations

import math
import re

_PIECE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|\s+|.", re.S)
TEXT_KEYS = ("text", "title", "detail", "label", "note")


def count(text: str | None) -> int:
    if not text:
        return 0
    n = 0
    for m in _PIECE.finditer(str(text)):
        p = m.group()
        if p.isspace():
            n += p.count("\n") and 1  # a newline run is a token; spaces ride along with the next word
        elif p[0].isascii() and p[0].isalpha():
            n += 1 if len(p) <= 7 else math.ceil(len(p) / 5)
        elif p.isdigit():
            n += math.ceil(len(p) / 3)
        else:
            n += 1  # punctuation, symbols, or one character of another script
    return n


def count_event(ev: dict) -> int:
    """Tokens the Hive 'wrote' in one event: its text, card titles, rows, fields and buttons."""
    n = 0
    for k in TEXT_KEYS:
        v = ev.get(k)
        if isinstance(v, str):
            n += count(v)
    for row in ev.get("details") or []:
        if isinstance(row, (list, tuple)):
            n += sum(count(str(x)) for x in row)
    for f in ev.get("fields") or []:
        if isinstance(f, dict):
            n += count(str(f.get("label", ""))) + count(str(f.get("value", "")))
    for a in ev.get("actions") or []:
        if isinstance(a, dict):
            n += count(str(a.get("label", "")))
    return n
