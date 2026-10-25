"""Return the index where a value should be inserted to keep a list sorted.

tags: list, binary, search, insert, position, sorted, bisect
"""

def binary_search_insert(xs, value):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo

# tests
assert binary_search_insert([1, 3, 5], 4) == 2
assert binary_search_insert([], 1) == 0
