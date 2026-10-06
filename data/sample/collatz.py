def collatz_steps(n):
    """Return how many steps n takes to reach 1 in the Collatz sequence."""
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps


def collatz_sequence(n):
    """Return the full Collatz sequence starting at n."""
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence
