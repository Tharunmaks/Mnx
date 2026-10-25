"""0/1 knapsack: maximum total value within a weight capacity.

tags: knapsack 0/1 dynamic programming weights values capacity
"""

def knapsack(weights: list, values: list, capacity: int) -> int:
    best = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            best[c] = max(best[c], best[c - w] + v)
    return best[capacity]


# tests
assert knapsack([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9
