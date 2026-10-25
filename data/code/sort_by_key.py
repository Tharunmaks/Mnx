"""Sort a list of dictionaries by one key.

tags: list, dictionary, sort, key, order
"""

def sort_by_key(rows, key):
    return sorted(rows, key=lambda r: r[key])

# tests
assert sort_by_key([{'a': 2}, {'a': 1}], 'a') == [{'a': 1}, {'a': 2}]
