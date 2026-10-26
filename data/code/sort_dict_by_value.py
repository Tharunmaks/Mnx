"""Return the dictionary items sorted by value, largest first.

tags: dictionary, sort, value, rank, order
"""

def sort_dict_by_value(d):
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)

# tests
assert sort_dict_by_value({'a': 1, 'b': 3}) == [('b', 3), ('a', 1)]
