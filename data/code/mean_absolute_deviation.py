"""Return the average distance of each number from the mean.

tags: math, statistics, mean, deviation, average, absolute
"""

def mean_absolute_deviation(numbers):
    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)

# tests
assert mean_absolute_deviation([1, 2, 3, 4]) == 1.0
