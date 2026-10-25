"""Return the area of a triangle from its base and height.

tags: math, geometry, triangle, area
"""

def triangle_area(base, height):
    return base * height / 2.0

# tests
assert triangle_area(5, 3) == 7.5
