"""Check whether a number is an Armstrong (narcissistic) number.

tags: armstrong narcissistic number digits power
"""

def armstrong_number(n: int) -> bool:
    digits = str(n)
    return n == sum(int(d) ** len(digits) for d in digits)


# tests
assert armstrong_number(153) and not armstrong_number(154)
