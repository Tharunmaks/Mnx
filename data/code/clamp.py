"""Limit a number to the range lo..hi.

tags: math, clamp, range, limit
"""

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# tests
assert clamp(5, 0, 10) == 5
assert clamp(-3, 0, 10) == 0
assert clamp(42, 0, 10) == 10
