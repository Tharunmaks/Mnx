"""Return the arithmetic mean of a non-empty list of numbers.

tags: math, list, mean, average
"""

def average(xs):
    return sum(xs) / len(xs)

# tests
assert average([1, 2, 3]) == 2
assert average([4]) == 4
assert average([1, 2]) == 1.5
