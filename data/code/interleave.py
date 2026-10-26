"""Alternate the items of two lists.

tags: list, interleave, alternate, merge, zip
"""

def interleave(a, b):
    out = []
    for x, y in zip(a, b):
        out.append(x)
        out.append(y)
    return out + list(a[len(b):]) + list(b[len(a):])

# tests
assert interleave([1, 3], [2, 4]) == [1, 2, 3, 4]
assert interleave([1], []) == [1]
