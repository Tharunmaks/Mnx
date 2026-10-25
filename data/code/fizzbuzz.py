"""Return the FizzBuzz sequence from 1 to n as a list of strings.

tags: fizzbuzz fizz buzz
"""

def fizzbuzz(n: int) -> list:
    out = []
    for i in range(1, n + 1):
        s = ("Fizz" if i % 3 == 0 else "") + ("Buzz" if i % 5 == 0 else "")
        out.append(s or str(i))
    return out


# tests
assert fizzbuzz(15)[-1] == "FizzBuzz"
assert fizzbuzz(5) == ["1", "2", "Fizz", "4", "Buzz"]
