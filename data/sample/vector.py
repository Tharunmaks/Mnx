import math


def dot(a, b):
    """Return the dot product of two equal-length vectors."""
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    """Return the Euclidean length of a vector."""
    return math.sqrt(dot(v, v))


def normalize(v):
    """Return v scaled to unit length."""
    length = norm(v)
    if length == 0:
        raise ValueError("cannot normalize zero vector")
    return [x / length for x in v]


def add_vectors(a, b):
    """Return the element-wise sum of two vectors."""
    return [x + y for x, y in zip(a, b)]
