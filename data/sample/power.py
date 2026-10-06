def power(base, exponent):
    """Return base raised to exponent using fast exponentiation."""
    result = 1
    while exponent > 0:
        if exponent % 2 == 1:
            result *= base
        base *= base
        exponent //= 2
    return result


def square(x):
    """Return x squared."""
    return x * x


def cube(x):
    """Return x cubed."""
    return x * x * x
