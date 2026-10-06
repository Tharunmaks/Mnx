def mean(values):
    """Return the arithmetic mean of a non-empty list."""
    if not values:
        raise ValueError("mean of empty list")
    return sum(values) / len(values)


def median(values):
    """Return the median of a non-empty list."""
    if not values:
        raise ValueError("median of empty list")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[mid - 1] + ordered[mid]) / 2
    return ordered[mid]


def variance(values):
    """Return the population variance of a list."""
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)
