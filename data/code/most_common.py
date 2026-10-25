"""Return the item that appears most often in a list.

tags: list, most, common, frequent, mode
"""

def most_common(items):
    counts = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    return max(counts, key=counts.get)

# tests
assert most_common([1, 2, 2, 3]) == 2
assert most_common(['a']) == 'a'
