"""Sort a list with insertion sort and return a new list.

tags: insertion sort sorting algorithm
"""

def insertion_sort(items: list) -> list:
    a = list(items)
    for i in range(1, len(a)):
        key, j = a[i], i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


# tests
assert insertion_sort([3, 1, 2]) == [1, 2, 3]
