"""Return the area of a circle from its radius.

tags: math, geometry, circle, area, radius
"""

import math


def circle_area(radius):
    return math.pi * radius ** 2

# tests
assert abs(circle_area(1) - 3.14159265) < 1e-6
