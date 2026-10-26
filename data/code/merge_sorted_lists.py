"""Merge two sorted lists into one sorted list.

tags: merge two sorted lists
"""

def merge_sorted_lists(a: list, b: list) -> list:
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    return out + a[i:] + b[j:]


# tests
assert merge_sorted_lists([1, 4, 6], [2, 3, 7]) == [1, 2, 3, 4, 6, 7]
