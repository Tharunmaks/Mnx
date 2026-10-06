def find_max(values):
    """Return the largest element of a non-empty list."""
    best = values[0]
    for v in values[1:]:
        if v > best:
            best = v
    return best


def find_min(values):
    """Return the smallest element of a non-empty list."""
    best = values[0]
    for v in values[1:]:
        if v < best:
            best = v
    return best


def clamp(value, low, high):
    """Clamp value into the inclusive range [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value
