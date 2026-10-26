"""Return indices of the two numbers that add up to target (hash map, O(n)), or None.

tags: two sum target pair indices array
"""

def two_sum(nums: list, target: int):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
    return None


# tests
assert two_sum([2, 7, 11, 15], 9) == [0, 1]
assert two_sum([1, 2], 7) is None
