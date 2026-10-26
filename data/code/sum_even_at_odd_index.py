"""Add the even numbers that sit at odd positions in a list.

tags: list, even, odd, index, position, sum
"""

def sum_even_at_odd_index(xs):
    return sum(x for i, x in enumerate(xs) if i % 2 == 1 and x % 2 == 0)

# tests
assert sum_even_at_odd_index([4, 2, 6, 7]) == 2
assert sum_even_at_odd_index([]) == 0
