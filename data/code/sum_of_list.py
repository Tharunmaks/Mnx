"""Sum the numbers in a list without using sum().

tags: sum list numbers total add loop
"""

def sum_of_list(nums: list):
    total = 0
    for n in nums:
        total += n
    return total


# tests
assert sum_of_list([1, 2, 3]) == 6 and sum_of_list([]) == 0
