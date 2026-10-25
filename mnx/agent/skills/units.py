"""Unit conversion with exact conversion factors.

Each unit maps to ``(dimension, factor)`` where ``factor`` converts the unit
into the SI base unit of its dimension, as an exact Fraction wherever the
definition is exact (1 inch is defined as exactly 0.0254 m, and so on).
Temperatures are affine and handled separately.
"""

from __future__ import annotations

import re
from fractions import Fraction as F
from typing import Dict, Optional, Tuple

from mnx.agent.base import Answer, Skill, fmt_number

_UNITS: Dict[str, Tuple[str, F]] = {}


def _add(dim: str, factor, *names: str) -> None:
    for n in names:
        _UNITS[n] = (dim, F(factor))


# length (metre)
_add("length", 1, "m", "meter", "meters", "metre", "metres")
_add("length", 1000, "km", "kilometer", "kilometers", "kilometre", "kilometres")
_add("length", F(1, 100), "cm", "centimeter", "centimeters", "centimetre", "centimetres")
_add("length", F(1, 1000), "mm", "millimeter", "millimeters", "millimetre", "millimetres")
_add("length", F(1, 10**6), "um", "µm", "micrometer", "micrometers", "micron", "microns")
_add("length", F(1, 10**9), "nm", "nanometer", "nanometers", "nanometre", "nanometres")
_add("length", F("0.0254"), "in", "inch", "inches")
_add("length", F("0.3048"), "ft", "foot", "feet")
_add("length", F("0.9144"), "yd", "yard", "yards")
_add("length", F("1609.344"), "mi", "mile", "miles")
_add("length", 1852, "nmi", "nautical mile", "nautical miles")
_add("length", F(149597870700), "au", "astronomical unit", "astronomical units")
_add("length", F(9460730472580800), "ly", "light year", "light years", "light-year", "light-years")
# mass (kilogram)
_add("mass", 1, "kg", "kilogram", "kilograms", "kilo", "kilos")
_add("mass", F(1, 1000), "g", "gram", "grams")
_add("mass", F(1, 10**6), "mg", "milligram", "milligrams")
_add("mass", 1000, "t", "tonne", "tonnes", "metric ton", "metric tons")
_add("mass", F("0.45359237"), "lb", "lbs", "pound", "pounds")
_add("mass", F("0.45359237") / 16, "oz", "ounce", "ounces")
_add("mass", F("0.45359237") * 14, "st", "stone", "stones")
# time (second)
_add("time", 1, "s", "sec", "secs", "second", "seconds")
_add("time", F(1, 1000), "ms", "millisecond", "milliseconds")
_add("time", 60, "min", "mins", "minute", "minutes")
_add("time", 3600, "h", "hr", "hrs", "hour", "hours")
_add("time", 86400, "day", "days")
_add("time", 604800, "week", "weeks")
_add("time", F(31556952), "year", "years", "yr", "yrs")  # Gregorian mean year
# area (square metre)
_add("area", 1, "m2", "m^2", "sq m", "square meter", "square meters", "square metre", "square metres")
_add("area", 10**6, "km2", "km^2", "sq km", "square kilometer", "square kilometers", "square kilometre", "square kilometres")
_add("area", F("0.09290304"), "ft2", "ft^2", "sq ft", "square foot", "square feet")
_add("area", F("4046.8564224"), "acre", "acres")
_add("area", 10000, "ha", "hectare", "hectares")
# volume (cubic metre)
_add("volume", F(1, 1000), "l", "liter", "liters", "litre", "litres")
_add("volume", F(1, 10**6), "ml", "milliliter", "milliliters", "millilitre", "millilitres")
_add("volume", 1, "m3", "m^3", "cubic meter", "cubic meters", "cubic metre", "cubic metres")
_add("volume", F("0.003785411784"), "gal", "gallon", "gallons", "us gallon", "us gallons")
_add("volume", F("0.003785411784") / 16, "cup", "cups")
_add("volume", F("0.003785411784") / 128, "fl oz", "fluid ounce", "fluid ounces")
_add("volume", F("0.003785411784") / 8, "pint", "pints")
# speed (metre / second)
_add("speed", 1, "m/s", "meters per second", "metres per second")
_add("speed", F(1000, 3600), "km/h", "kph", "kmh", "kilometers per hour", "kilometres per hour")
_add("speed", F("1609.344") / 3600, "mph", "miles per hour")
_add("speed", F(1852, 3600), "knot", "knots", "kn")
# data (byte)
_add("data", F(1, 8), "bit", "bits")
_add("data", 1, "b", "byte", "bytes")
for _i, (_p, _name) in enumerate([("k", "kilo"), ("m", "mega"), ("g", "giga"), ("t", "tera"), ("p", "peta")], 1):
    _add("data", 1000**_i, f"{_p}b", f"{_name}byte", f"{_name}bytes")
    _add("data", 1024**_i, f"{_p}ib", f"{_name[:2]}bibyte", f"{_name[:2]}bibytes")
# energy (joule)
_add("energy", 1, "j", "joule", "joules")
_add("energy", 1000, "kj", "kilojoule", "kilojoules")
_add("energy", F("4.184"), "cal", "calorie", "calories")
_add("energy", F("4184"), "kcal", "kilocalorie", "kilocalories")
_add("energy", 3600000, "kwh", "kilowatt hour", "kilowatt hours", "kilowatt-hour", "kilowatt-hours")

_TEMPS = {
    "c": "C", "°c": "C", "celsius": "C", "degrees celsius": "C", "degree celsius": "C", "centigrade": "C",
    "f": "F", "°f": "F", "fahrenheit": "F", "degrees fahrenheit": "F", "degree fahrenheit": "F",
    "k": "K", "kelvin": "K", "kelvins": "K",
}


def _to_kelvin(v: F, u: str) -> F:
    return v + F("273.15") if u == "C" else (v - 32) * F(5, 9) + F("273.15") if u == "F" else v


def _from_kelvin(v: F, u: str) -> F:
    return v - F("273.15") if u == "C" else (v - F("273.15")) * F(9, 5) + 32 if u == "F" else v


def _lookup(name: str) -> Optional[Tuple[str, object]]:
    n = name.strip().lower().rstrip(".?!")
    n = re.sub(r"^(?:a|an|one|the)\s+", "", n)
    if n in _TEMPS:
        return "temperature", _TEMPS[n]
    if n in _UNITS:
        return _UNITS[n]
    return None


_NUM = r"(-?\d+(?:,\d{3})*(?:\.\d+)?(?:e-?\d+)?|-?\.\d+)"
_U = r"([a-z°µ/^0-9\- ]+?)"
_PATTERNS = [
    re.compile(rf"^(?:convert\s+)?{_NUM}\s*{_U}\s+(?:to|in|into|as|in terms of)\s+{_U}$"),
    re.compile(rf"^how many\s+{_U}\s+(?:are\s+)?(?:there\s+)?(?:in|is|are|make up)\s+(?:{_NUM}\s*)?{_U}$"),
]


_DISPLAY = {"c": "°C", "f": "°F", "k": "K", "gb": "GB", "mb": "MB", "kb": "KB", "tb": "TB", "pb": "PB",
            "gib": "GiB", "mib": "MiB", "kib": "KiB", "tib": "TiB", "kwh": "kWh", "j": "J", "kj": "kJ", "l": "L"}


def _display(unit: str) -> str:
    return _DISPLAY.get(unit.strip().lower(), unit.strip())


class UnitSkill(Skill):
    name = "units"
    description = "unit conversion with exact factors (length, mass, time, volume, speed, data, energy, temperature)"

    def answer(self, question: str) -> Optional[Answer]:
        q = question.strip().lower()
        q = re.sub(r"^(?:please\s+)?(?:can you\s+)?(?:what is|what's|whats)\s+", "", q).rstrip(" ?.!")
        m = _PATTERNS[0].match(q)
        if m:
            value, src, dst = m.group(1), m.group(2), m.group(3)
        else:
            m = _PATTERNS[1].match(q)
            if not m:
                return None
            dst, value, src = m.group(1), m.group(2) or "1", m.group(3)
        a, b = _lookup(src), _lookup(dst)
        if a is None or b is None:
            return None
        v = F(value.replace(",", ""))
        src = re.sub(r"^(?:a|an|one|the)\s+", "", src.strip())
        dst = re.sub(r"^(?:a|an|one|the)\s+", "", dst.strip())
        if a[0] != b[0]:
            return Answer(f"Can't convert {src.strip()} ({a[0]}) to {dst.strip()} ({b[0]}): different dimensions.",
                          self.name, 0.9, True, [f"{src.strip()} is {a[0]}, {dst.strip()} is {b[0]}"])
        if a[0] == "temperature":
            out = _from_kelvin(_to_kelvin(v, a[1]), b[1])
            trace = [f"{fmt_number(v)} °{a[1]} → {fmt_number(_to_kelvin(v, a[1]))} K → °{b[1]}"]
            back = _from_kelvin(_to_kelvin(out, b[1]), a[1])
        else:
            out = v * a[1] / b[1]
            trace = [f"1 {src.strip()} = {fmt_number(a[1])} SI units of {a[0]}; 1 {dst.strip()} = {fmt_number(b[1])}"]
            back = out * b[1] / a[1]
        trace.append(f"check: converting back gives {fmt_number(back)} {'✓' if back == v else '✗'}")
        shown = fmt_number(out) if out.denominator == 1 else f"{float(out):.6g}"
        return Answer(f"{value} {_display(src)} = {shown} {_display(dst)}", self.name, 0.97, back == v, trace)
