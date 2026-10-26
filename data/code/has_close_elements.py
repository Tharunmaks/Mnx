"""Return True if any two numbers in the list are closer than a threshold.

tags: list, close, threshold, distance, pairs
"""

def has_close_elements(numbers, threshold):
    for i, a in enumerate(numbers):
        for b in numbers[i + 1:]:
            if abs(a - b) < threshold:
                return True
    return False

# tests
assert has_close_elements([1.0, 2.0, 3.0], 0.5) is False
assert has_close_elements([1.0, 1.2], 0.5) is True
