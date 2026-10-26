"""Rotate a list to the right by k positions.

tags: rotate list shift right k
"""

def rotate_list(items: list, k: int) -> list:
    if not items:
        return []
    k %= len(items)
    return items[-k:] + items[:-k] if k else list(items)


# tests
assert rotate_list([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]
assert rotate_list([], 3) == []
