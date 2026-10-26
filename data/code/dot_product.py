"""Return the dot product of two equal-length vectors.

tags: math, vector, dot, product, multiply, sum
"""

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

# tests
assert dot_product([1, 2], [3, 4]) == 11
assert dot_product([], []) == 0
