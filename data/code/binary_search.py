"""Return the index of target in a sorted list using binary search, or -1 if absent.

tags: binary search sorted list find index
"""

def binary_search(items: list, target) -> int:
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


# tests
assert binary_search([1, 3, 5, 7, 9], 7) == 3
assert binary_search([1, 3, 5], 4) == -1
assert binary_search([], 1) == -1
