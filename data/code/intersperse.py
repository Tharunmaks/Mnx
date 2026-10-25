"""Insert a delimiter between every two consecutive items of a list.

tags: list, intersperse, delimiter, insert, between
"""

def intersperse(xs, delimiter):
    if not xs:
        return []
    out = [xs[0]]
    for x in xs[1:]:
        out.append(delimiter)
        out.append(x)
    return out

# tests
assert intersperse([1, 2, 3], 0) == [1, 0, 2, 0, 3]
assert intersperse([], 4) == []
