def invert(mapping):
    """Swap keys and values of a dict."""
    return {value: key for key, value in mapping.items()}


def merge_dicts(a, b):
    """Return a new dict with keys from a and b; b wins on conflicts."""
    result = dict(a)
    result.update(b)
    return result


def group_by(items, key):
    """Group items into a dict keyed by key(item)."""
    groups = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return groups
