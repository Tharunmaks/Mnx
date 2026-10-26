"""Sort a list with merge sort in O(n log n) and return a new list.

tags: merge sort sorting algorithm divide conquer
"""

def merge_sort(items: list) -> list:
    if len(items) <= 1:
        return list(items)
    mid = len(items) // 2
    left, right = merge_sort(items[:mid]), merge_sort(items[mid:])
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    return out + left[i:] + right[j:]


# tests
assert merge_sort([5, 3, 8, 1, 2]) == [1, 2, 3, 5, 8]
assert merge_sort([1]) == [1]
