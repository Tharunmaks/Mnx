"""Rotate a square matrix by 90 degrees clockwise.

tags: matrix, rotate, clockwise, grid, transpose
"""

def matrix_rotate(m):
    return [list(row) for row in zip(*m[::-1])]

# tests
assert matrix_rotate([[1, 2], [3, 4]]) == [[3, 1], [4, 2]]
