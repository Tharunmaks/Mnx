"""Flatten a nested dictionary into dotted keys.

tags: dictionary, flatten, nested, keys, recursion
"""

def flatten_dict(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, key + "."))
        else:
            out[key] = v
    return out

# tests
assert flatten_dict({'a': {'b': 1}}) == {'a.b': 1}
assert flatten_dict({}) == {}
