"""Least common multiple of two integers.

tags: lcm least common multiple
"""

def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    x, y = abs(a), abs(b)
    while y:
        x, y = y, x % y
    return abs(a * b) // x


# tests
assert lcm(4, 6) == 12
assert lcm(0, 3) == 0
