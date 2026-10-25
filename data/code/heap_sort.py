"""Sort a list using a binary heap.

tags: sort, heap, algorithm, list, heapq
"""

import heapq


def heap_sort(xs):
    h = list(xs)
    heapq.heapify(h)
    return [heapq.heappop(h) for _ in range(len(h))]

# tests
assert heap_sort([3, 1, 2]) == [1, 2, 3]
assert heap_sort([]) == []
