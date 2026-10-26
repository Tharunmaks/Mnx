"""Return the sorted distinct items of a list.

tags: list, unique, distinct, sorted, duplicates
"""

def unique_sorted(xs):
    return sorted(set(xs))

# tests
assert unique_sorted([5, 3, 5, 2]) == [2, 3, 5]
assert unique_sorted([]) == []
