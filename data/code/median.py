"""Median of a list of numbers.

tags: median middle value statistics
"""

def median(nums: list) -> float:
    s = sorted(nums)
    n = len(s)
    if n == 0:
        raise ValueError("empty list")
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


# tests
assert median([3, 1, 2]) == 2
assert median([4, 1, 3, 2]) == 2.5
