"""Return True if an integer is a perfect cube.

tags: math, cube, perfect, root, check
"""

def is_cube(a):
    r = round(abs(a) ** (1 / 3))
    return r ** 3 == abs(a)

# tests
assert is_cube(27)
assert is_cube(0)
assert not is_cube(10)
