"""Return True if two distinct numbers in the list add up to zero.

tags: list, pairs, sum, zero, two
"""

def pairs_sum_to_zero(xs):
    for i, a in enumerate(xs):
        for b in xs[i + 1:]:
            if a + b == 0:
                return True
    return False

# tests
assert pairs_sum_to_zero([1, -1])
assert not pairs_sum_to_zero([1, 2])
