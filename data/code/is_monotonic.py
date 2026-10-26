"""Return True if a list is sorted either increasing or decreasing.

tags: list, monotonic, increasing, decreasing
"""

def is_monotonic(xs):
    up = all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))
    down = all(xs[i] >= xs[i + 1] for i in range(len(xs) - 1))
    return up or down

# tests
assert is_monotonic([1, 2, 3])
assert is_monotonic([3, 2, 1])
assert not is_monotonic([1, 3, 2])
