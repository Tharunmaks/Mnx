"""All subsets of a list (the power set).

tags: power set subsets combinations
"""

def power_set(items: list) -> list:
    out = [[]]
    for x in items:
        out += [s + [x] for s in out]
    return out


# tests
assert sorted(power_set([1, 2])) == [[], [1], [1, 2], [2]]
