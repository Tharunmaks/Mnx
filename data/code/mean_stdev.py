"""Mean and population standard deviation of a list of numbers.

tags: mean average standard deviation statistics variance
"""

def mean_stdev(nums: list) -> tuple:
    n = len(nums)
    mean = sum(nums) / n
    var = sum((x - mean) ** 2 for x in nums) / n
    return mean, var ** 0.5


# tests
assert mean_stdev([2, 4, 4, 4, 5, 5, 7, 9]) == (5.0, 2.0)
