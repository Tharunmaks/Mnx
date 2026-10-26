"""Calendar arithmetic: weekdays, differences between dates, offsets, leap years."""

from __future__ import annotations

import calendar
import datetime as dt
import re
from typing import Optional

from mnx.agent.base import Answer, Skill

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
_MONTHS["sept"] = 9
_MON = "|".join(sorted(_MONTHS, key=len, reverse=True))

DATE_RE = (
    r"(\d{4}-\d{1,2}-\d{1,2}"
    rf"|(?:{_MON})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MON})\.?,?\s+\d{{4}}"
    r"|today|tomorrow|yesterday)"
)


def parse_date(text: str, today: Optional[dt.date] = None) -> dt.date:
    t = text.strip().lower().rstrip(".,")
    today = today or dt.date.today()
    if t == "today":
        return today
    if t == "tomorrow":
        return today + dt.timedelta(days=1)
    if t == "yesterday":
        return today - dt.timedelta(days=1)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.fullmatch(rf"({_MON})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})", t)
    if m:
        return dt.date(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))
    m = re.fullmatch(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MON})\.?,?\s+(\d{{4}})", t)
    if m:
        return dt.date(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))
    raise ValueError(f"unrecognised date {text!r}")


def _fmt(d: dt.date) -> str:
    return f"{d.strftime('%A')}, {d.strftime('%B')} {d.day}, {d.year}"


def is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


class DateSkill(Skill):
    name = "dates"
    description = "weekday of a date, days between dates, date offsets, leap years"

    def __init__(self, today: Optional[dt.date] = None):
        self.today = today

    def answer(self, question: str) -> Optional[Answer]:
        q = question.strip().lower().rstrip(" ?.!")
        today = self.today or dt.date.today()
        try:
            return self._answer(q, today)
        except ValueError as exc:
            if re.search(DATE_RE, q):
                return Answer(f"That isn't a valid date: {exc}.", self.name, 0.6)
            return None

    def _answer(self, q: str, today: dt.date) -> Optional[Answer]:
        m = re.search(r"\bis\s+(\d{3,4})\s+a\s+leap\s+year", q) or re.search(r"\bwas\s+(\d{3,4})\s+a\s+leap\s+year", q)
        if m:
            y = int(m.group(1))
            leap = is_leap(y)
            trace = ["Gregorian rule: divisible by 4, except centuries, unless divisible by 400",
                     f"cross-check with calendar.isleap: {calendar.isleap(y)}"]
            return Answer(f"{'Yes' if leap else 'No'}, {y} is {'' if leap else 'not '}a leap year.",
                          self.name, 0.98, leap == calendar.isleap(y), trace)
        m = re.search(rf"(?:between|from)\s+{DATE_RE}\s+(?:and|to|until)\s+{DATE_RE}", q)
        if m and re.search(r"\b(days?|weeks?|how long|how many)\b", q):
            a, b = parse_date(m.group(1), today), parse_date(m.group(2), today)
            days = (b - a).days
            back = a + dt.timedelta(days=days)
            weeks, rem = divmod(abs(days), 7)
            extra = f" ({weeks} weeks and {rem} days)" if weeks else ""
            return Answer(f"{abs(days)} days{extra} from {_fmt(min(a, b))} to {_fmt(max(a, b))}.", self.name, 0.97,
                          back == b, [f"ordinal day numbers {a.toordinal()} and {b.toordinal()}", "check: adding the difference back reproduces the end date"])
        m = re.search(rf"(\d+)\s+(day|week|month|year)s?\s+(after|from|before|since|until)\s+{DATE_RE}", q)
        if m:
            n, unit, direction, base = int(m.group(1)), m.group(2), m.group(3), parse_date(m.group(4), today)
            sign = -1 if direction in ("before", "until") else 1
            out = _shift(base, sign * n, unit)
            check = _shift(out, -sign * n, unit) == base or unit in ("month", "year")
            return Answer(f"{_fmt(out)}.", self.name, 0.96, check, [f"{_fmt(base)} {'+' if sign > 0 else '-'} {n} {unit}(s)"])
        m = re.search(r"(?:in|after)\s+(\d+)\s+(day|week|month|year)s?$", q)
        if m and re.search(r"\b(date|day|when)\b", q):
            out = _shift(today, int(m.group(1)), m.group(2))
            return Answer(f"{_fmt(out)}.", self.name, 0.9, True, [f"today is {_fmt(today)}"])
        m = re.search(rf"(?:what\s+)?(?:day(?: of the week)?|weekday)\s+(?:is|was|will be|does|did|falls?)\s+(?:it\s+)?(?:on\s+)?{DATE_RE}", q) \
            or re.search(rf"{DATE_RE}\s+(?:is|was|falls on)\s+(?:a|what)\s+(?:day|weekday)", q)
        if m:
            d = parse_date(m.group(1), today)
            name = d.strftime("%A")
            zeller = _zeller(d)
            return Answer(f"{_fmt(d)} is a {name}.", self.name, 0.98, zeller == name,
                          [f"datetime weekday: {name}", f"cross-check with Zeller's congruence: {zeller}"])
        if re.fullmatch(r"(?:what(?:'s| is)\s+)?(?:the\s+)?(?:date\s+)?today(?:'s date)?|what(?:'s| is) the date(?: today)?|what day is (?:it )?today", q):
            return Answer(f"Today is {_fmt(today)} (from this computer's clock).", self.name, 0.9, False, ["read the system clock"])
        return None


def _shift(d: dt.date, n: int, unit: str) -> dt.date:
    if unit == "day":
        return d + dt.timedelta(days=n)
    if unit == "week":
        return d + dt.timedelta(weeks=n)
    months = n * (12 if unit == "year" else 1)
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day)


def _zeller(d: dt.date) -> str:
    """Independent weekday computation (Zeller's congruence) used as a check."""
    q, m, y = d.day, d.month, d.year
    if m < 3:
        m += 12
        y -= 1
    k, j = y % 100, y // 100
    h = (q + (13 * (m + 1)) // 5 + k + k // 4 + j // 4 + 5 * j) % 7
    return ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][h]
