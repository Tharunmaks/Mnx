"""Return the list with the items at indices divisible by three sorted.

tags: list, sort, third, indices, every
"""

def sort_third(xs):
    picked = sorted(xs[::3])
    out = list(xs)
    out[::3] = picked
    return out

# tests
assert sort_third([1, 2, 3]) == [1, 2, 3]
assert sort_third([5, 6, 3, 4, 8, 9, 2]) == [2, 6, 3, 4, 8, 9, 5]
