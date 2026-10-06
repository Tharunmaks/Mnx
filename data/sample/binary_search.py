def binary_search(items, target):
    """Return the index of target in sorted items, or -1 if absent."""
    low, high = 0, len(items) - 1
    while low <= high:
        mid = (low + high) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def linear_search(items, target):
    """Return the index of target in items, or -1 if absent."""
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1
