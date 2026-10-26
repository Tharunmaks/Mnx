"""Return the even numbers from a list.

tags: even numbers filter list
"""

def even_numbers(nums: list) -> list:
    return [n for n in nums if n % 2 == 0]


# tests
assert even_numbers([1, 2, 3, 4]) == [2, 4]
