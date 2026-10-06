def digit_sum(n):
    """Return the sum of the decimal digits of n."""
    return sum(int(d) for d in str(abs(n)))


def reverse_number(n):
    """Return n with its digits reversed."""
    sign = -1 if n < 0 else 1
    return sign * int(str(abs(n))[::-1])


def is_armstrong(n):
    """Return True if n equals the sum of its digits raised to the digit count."""
    digits = str(n)
    k = len(digits)
    return n == sum(int(d) ** k for d in digits)
