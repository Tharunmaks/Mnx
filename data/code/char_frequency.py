"""Count how often each character appears in a string.

tags: character frequency count letters histogram
"""

def char_frequency(s: str) -> dict:
    counts = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    return counts


# tests
assert char_frequency("aab") == {"a": 2, "b": 1}
