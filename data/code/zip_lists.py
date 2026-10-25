"""Pair up the items of two lists.

tags: list, zip, pair, combine, together
"""

def zip_lists(a, b):
    return list(zip(a, b))

# tests
assert zip_lists([1, 2], ['a', 'b']) == [(1, 'a'), (2, 'b')]
assert zip_lists([], []) == []
