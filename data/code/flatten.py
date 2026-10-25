"""Flatten an arbitrarily nested list.

tags: flatten nested list recursive
"""

def flatten(items: list) -> list:
    out = []
    for x in items:
        if isinstance(x, list):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out


# tests
assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]
