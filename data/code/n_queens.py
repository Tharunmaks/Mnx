"""Count the solutions to the N-queens puzzle with backtracking.

tags: n queens backtracking chess puzzle
"""

def n_queens(n: int) -> int:
    count = 0
    cols, d1, d2 = set(), set(), set()

    def place(row):
        nonlocal count
        if row == n:
            count += 1
            return
        for c in range(n):
            if c in cols or row - c in d1 or row + c in d2:
                continue
            cols.add(c); d1.add(row - c); d2.add(row + c)
            place(row + 1)
            cols.discard(c); d1.discard(row - c); d2.discard(row + c)

    place(0)
    return count


# tests
assert n_queens(4) == 2 and n_queens(8) == 92
