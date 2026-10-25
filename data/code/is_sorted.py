"""Return True if a list is in non-decreasing order.

tags: list, sorted, order, monotonic, check
"""

def is_sorted(xs):
    return all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))

# tests
assert is_sorted([1, 2, 3])
assert not is_sorted([3, 1])
assert is_sorted([])
