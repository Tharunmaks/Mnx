"""Sort a list by repeatedly taking the smallest remaining item.

tags: sort, selection, algorithm, list
"""

def selection_sort(xs):
    out = list(xs)
    for i in range(len(out)):
        j = min(range(i, len(out)), key=lambda k: out[k])
        out[i], out[j] = out[j], out[i]
    return out

# tests
assert selection_sort([3, 1, 2]) == [1, 2, 3]
assert selection_sort([]) == []
