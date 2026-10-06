def transpose(matrix):
    """Return the transpose of a matrix given as a list of rows."""
    return [list(row) for row in zip(*matrix)]


def matmul(a, b):
    """Multiply two matrices given as lists of rows."""
    rows, inner, cols = len(a), len(b), len(b[0])
    result = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            total = 0
            for k in range(inner):
                total += a[i][k] * b[k][j]
            result[i][j] = total
    return result


def identity(n):
    """Return the n by n identity matrix."""
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
