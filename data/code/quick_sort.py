"""Sort a list with quicksort and return a new list.

tags: quick sort quicksort sorting algorithm pivot
"""

def quick_sort(items: list) -> list:
    if len(items) <= 1:
        return list(items)
    pivot = items[len(items) // 2]
    less = [x for x in items if x < pivot]
    equal = [x for x in items if x == pivot]
    greater = [x for x in items if x > pivot]
    return quick_sort(less) + equal + quick_sort(greater)


# tests
assert quick_sort([3, 6, 1, 8, 1]) == [1, 1, 3, 6, 8]
