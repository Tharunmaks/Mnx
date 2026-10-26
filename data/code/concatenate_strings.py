"""Join a list of strings into one string.

tags: string, join, concatenate, list
"""

def concatenate_strings(strings):
    return "".join(strings)

# tests
assert concatenate_strings(['a', 'b']) == 'ab'
assert concatenate_strings([]) == ''
