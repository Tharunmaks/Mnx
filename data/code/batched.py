"""Split an iterable into tuples of at most n items.

tags: list, batch, chunk, split, groups, size
"""

def batched(items, n):
    items = list(items)
    return [tuple(items[i:i + n]) for i in range(0, len(items), n)]

# tests
assert batched([1, 2, 3], 2) == [(1, 2), (3,)]
assert batched([], 2) == []
