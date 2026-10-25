"""Split numbers into the positive ones and the negative ones.

tags: list, split, positive, negative, partition
"""

def split_positive_negative(xs):
    return [x for x in xs if x > 0], [x for x in xs if x < 0]

# tests
assert split_positive_negative([1, -2, 3]) == ([1, 3], [-2])
