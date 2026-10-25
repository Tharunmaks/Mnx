"""Return the fewest coins needed to make an amount, or -1.

tags: dynamic, programming, coins, change, minimum, money
"""

def coin_change(coins, amount):
    best = [0] + [float("inf")] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                best[a] = min(best[a], best[a - c] + 1)
    return -1 if best[amount] == float("inf") else best[amount]

# tests
assert coin_change([1, 5], 7) == 3
assert coin_change([5], 3) == -1
