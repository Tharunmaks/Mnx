"""Return the sum of the squares of the numbers in a list.

tags: math, list, sum, squares
"""

def sum_of_squares(xs):
    return sum(x * x for x in xs)

# tests
assert sum_of_squares([1, 2, 3]) == 14
assert sum_of_squares([]) == 0
assert sum_of_squares([-2]) == 4
