"""Return the distinct items of a list, keeping their first order.

tags: list, unique, distinct, order, duplicates, first
"""

def unique_everseen(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# tests
assert unique_everseen([3, 1, 3, 2]) == [3, 1, 2]
assert unique_everseen([]) == []
