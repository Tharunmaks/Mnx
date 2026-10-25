"""Differentiate a polynomial given as a list of coefficients.

tags: math, derivative, polynomial, coefficients, differentiate
"""

def derivative_of_poly(xs):
    return [i * c for i, c in enumerate(xs)][1:]

# tests
assert derivative_of_poly([3, 1, 2, 4]) == [1, 4, 12]
assert derivative_of_poly([1]) == []
