"""Return True if a running total of the deposits and withdrawals ever drops below zero.

tags: list, running, balance, below, zero, negative
"""

def below_zero(operations):
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False

# tests
assert not below_zero([1, 2, 3])
assert below_zero([1, -4, 2])
