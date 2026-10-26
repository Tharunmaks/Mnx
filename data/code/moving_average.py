"""Return the averages of every window of k consecutive numbers.

tags: list, moving, average, window, rolling, mean
"""

def moving_average(xs, k):
    return [sum(xs[i:i + k]) / k for i in range(len(xs) - k + 1)]

# tests
assert moving_average([1, 2, 3], 2) == [1.5, 2.5]
