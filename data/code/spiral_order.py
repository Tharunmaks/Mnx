"""Read a matrix in spiral order.

tags: matrix, spiral, order, grid, traverse
"""

def spiral_order(m):
    out = []
    m = [list(r) for r in m]
    while m:
        out.extend(m.pop(0))
        m = [list(r) for r in zip(*m)][::-1]
    return out

# tests
assert spiral_order([[1, 2], [4, 3]]) == [1, 2, 3, 4]
assert spiral_order([]) == []
