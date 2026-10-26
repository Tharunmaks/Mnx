"""Return True if three distinct numbers in the list add up to zero.

tags: list, triples, three, sum, zero
"""

def triples_sum_to_zero(xs):
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if xs[i] + xs[j] + xs[k] == 0:
                    return True
    return False

# tests
assert triples_sum_to_zero([1, 3, -4])
assert not triples_sum_to_zero([1, 2, 3])
