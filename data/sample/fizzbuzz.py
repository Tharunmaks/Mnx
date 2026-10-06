def fizzbuzz(n):
    """Return the FizzBuzz string for n."""
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)


def fizzbuzz_list(limit):
    """Return FizzBuzz results for 1 through limit."""
    return [fizzbuzz(i) for i in range(1, limit + 1)]
