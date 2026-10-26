"""Read a value out of nested dictionaries by a path of keys.

tags: dictionary, nested, get, path, keys, deep
"""

def deep_get(d, path, default=None):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d

# tests
assert deep_get({'a': {'b': 1}}, ['a', 'b']) == 1
assert deep_get({}, ['x'], 0) == 0
