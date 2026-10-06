def flatten(nested):
    """Flatten one level of nesting."""
    result = []
    for item in nested:
        result.extend(item)
    return result


def chunk(items, size):
    """Split items into chunks of the given size."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def unique(items):
    """Return the unique items in order of first appearance."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def rotate(items, k):
    """Rotate items to the left by k positions."""
    if not items:
        return []
    k %= len(items)
    return items[k:] + items[:k]
