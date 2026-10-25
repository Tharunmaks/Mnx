"""All permutations of a list.

tags: permutations all orderings list recursion
"""

def permutations(items: list) -> list:
    if len(items) <= 1:
        return [list(items)]
    out = []
    for i, x in enumerate(items):
        for rest in permutations(items[:i] + items[i + 1:]):
            out.append([x] + rest)
    return out


# tests
assert sorted(permutations([1, 2, 3])) == [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
