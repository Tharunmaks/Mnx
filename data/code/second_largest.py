"""Return the second largest distinct value in a list, or None.

tags: second largest number list
"""

def second_largest(nums: list):
    distinct = sorted(set(nums), reverse=True)
    return distinct[1] if len(distinct) > 1 else None


# tests
assert second_largest([4, 1, 7, 7, 3]) == 4
assert second_largest([5, 5]) is None
