"""Determinant of a square matrix by Gaussian elimination.

tags: determinant matrix square linear algebra
"""

def matrix_determinant(m: list) -> float:
    a = [list(map(float, row)) for row in m]
    n, det = len(a), 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        if a[p][c] == 0:
            return 0.0
        if p != c:
            a[c], a[p] = a[p], a[c]
            det = -det
        det *= a[c][c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            for k in range(c, n):
                a[r][k] -= f * a[c][k]
    return det


# tests
assert abs(matrix_determinant([[1, 2], [3, 4]]) + 2) < 1e-9
