"""Transpose a matrix given as a list of lists.

tags: transpose matrix rows columns
"""

def transpose(m: list) -> list:
    return [list(row) for row in zip(*m)]


# tests
assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
