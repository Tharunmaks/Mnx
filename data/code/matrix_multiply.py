"""Multiply two matrices given as lists of lists.

tags: matrix multiply multiplication product
"""

def matrix_multiply(a: list, b: list) -> list:
    if not a or len(a[0]) != len(b):
        raise ValueError("incompatible shapes")
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


# tests
assert matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]
