"""Add up the numbers from 1 to n.

tags: math, sum, series, triangle, total
"""

def sum_to_n(n):
    return n * (n + 1) // 2

# tests
assert sum_to_n(5) == 15
assert sum_to_n(0) == 0
