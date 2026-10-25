"""Count the Collatz steps needed to reach 1.

tags: math, collatz, steps, sequence, hailstone
"""

def collatz_steps(n):
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps

# tests
assert collatz_steps(1) == 0
assert collatz_steps(6) == 8
