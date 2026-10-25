"""Sort a list with bubble sort and return a new list.

tags: bubble sort sorting algorithm
"""

def bubble_sort(items: list) -> list:
    a = list(items)
    for i in range(len(a)):
        swapped = False
        for j in range(len(a) - 1 - i):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a


# tests
assert bubble_sort([5, 2, 9, 1]) == [1, 2, 5, 9]
assert bubble_sort([]) == []
