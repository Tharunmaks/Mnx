"""Remove duplicates from a list while keeping the original order.

tags: remove duplicates unique list order dedupe
"""

def remove_duplicates(items: list) -> list:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# tests
assert remove_duplicates([3, 1, 3, 2, 1]) == [3, 1, 2]
