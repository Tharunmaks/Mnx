"""Return True if an integer is a perfect square.

tags: math, square, perfect, root, check
"""

def is_square(n):
    if n < 0:
        return False
    r = int(n ** 0.5)
    return r * r == n or (r + 1) * (r + 1) == n

# tests
assert is_square(16)
assert not is_square(15)
