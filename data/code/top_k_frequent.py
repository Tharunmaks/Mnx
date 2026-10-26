"""The k most frequent elements of a list.

tags: top k most frequent elements common
"""

from collections import Counter


def top_k_frequent(items: list, k: int) -> list:
    return [x for x, _ in Counter(items).most_common(k)]


# tests
assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]
