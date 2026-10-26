"""Maximum sum of a contiguous subarray (Kadane's algorithm).

tags: maximum subarray sum kadane contiguous
"""

def max_subarray_sum(nums: list) -> int:
    best = cur = nums[0]
    for n in nums[1:]:
        cur = max(n, cur + n)
        best = max(best, cur)
    return best


# tests
assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
assert max_subarray_sum([-3, -1]) == -1
