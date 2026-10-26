"""Return the running maximum seen so far at each position.

tags: list, rolling, running, maximum, so far
"""

def rolling_max(numbers):
    out = []
    best = None
    for n in numbers:
        best = n if best is None else max(best, n)
        out.append(best)
    return out

# tests
assert rolling_max([1, 3, 2]) == [1, 3, 3]
assert rolling_max([]) == []
