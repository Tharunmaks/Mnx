"""Return the integer nth root of a non-negative number.

tags: math, root, nth, integer
"""

def nth_root(x, n):
    r = round(x ** (1.0 / n))
    while r ** n > x:
        r -= 1
    while (r + 1) ** n <= x:
        r += 1
    return r

# tests
assert nth_root(27, 3) == 3
assert nth_root(16, 2) == 4
