"""Split a list into chunks of size n.

tags: chunk split list into groups batches size
"""

def chunk_list(items: list, n: int) -> list:
    if n <= 0:
        raise ValueError("n must be positive")
    return [items[i:i + n] for i in range(0, len(items), n)]


# tests
assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
