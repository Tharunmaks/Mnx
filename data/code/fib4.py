"""Return the nth number of the four-step Fibonacci sequence.

tags: math, fibonacci, sequence, recurrence, four
"""

def fib4(n):
    seq = [0, 0, 2, 0]
    if n < 4:
        return seq[n]
    for _ in range(4, n + 1):
        seq.append(sum(seq[-4:]))
        seq.pop(0)
    return seq[-1]

# tests
assert fib4(5) == 4
assert fib4(0) == 0
