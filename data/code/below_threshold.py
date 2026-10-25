"""Return True if every number in the list is below a threshold.

tags: list, threshold, below, all, less
"""

def below_threshold(xs, t):
    return all(x < t for x in xs)

# tests
assert below_threshold([1, 2], 100)
assert not below_threshold([1, 20], 5)
