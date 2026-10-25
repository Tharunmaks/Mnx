"""Find the largest number in a list without using max().

tags: maximum largest number list find max
"""

def find_max(nums: list):
    if not nums:
        raise ValueError("empty list")
    best = nums[0]
    for n in nums[1:]:
        if n > best:
            best = n
    return best


# tests
assert find_max([3, 9, 2]) == 9
