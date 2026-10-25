"""Return the hypotenuse of a right triangle from its two sides.

tags: math, geometry, triangle, hypotenuse, pythagoras
"""

def hypotenuse(a, b):
    return (a * a + b * b) ** 0.5

# tests
assert hypotenuse(3, 4) == 5.0
