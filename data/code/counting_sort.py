"""Sort a list of small non-negative integers by counting them.

tags: sort, counting, integers, algorithm, buckets
"""

def counting_sort(xs):
    if not xs:
        return []
    counts = [0] * (max(xs) + 1)
    for x in xs:
        counts[x] += 1
    out = []
    for value, c in enumerate(counts):
        out.extend([value] * c)
    return out

# tests
assert counting_sort([3, 1, 2, 1]) == [1, 1, 2, 3]
assert counting_sort([]) == []
