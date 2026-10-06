def two_sum(numbers, target):
    """Return indices of two numbers that add up to target, or None."""
    seen = {}
    for i, value in enumerate(numbers):
        other = target - value
        if other in seen:
            return seen[other], i
        seen[value] = i
    return None


def max_subarray(numbers):
    """Return the largest sum of any contiguous subarray."""
    best = current = numbers[0]
    for value in numbers[1:]:
        current = max(value, current + value)
        best = max(best, current)
    return best
